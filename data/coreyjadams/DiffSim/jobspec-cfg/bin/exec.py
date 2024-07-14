import sys, os
import pathlib
import time

import signal
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# from logging import handlers

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log


hydra.output_subdir = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['PMI_LOCAL_RANK']
except:
    pass

import jax
from jax import random, jit, vmap
import jax.numpy as numpy
import jax.tree_util as tree_util
import optax

from flax.training import train_state

from tensorboardX import SummaryWriter


# Add the local folder to the import path:
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(src_dir)
sys.path.insert(0,src_dir)

from diffsim.config import Config

# if MPI_AVAILABLE:
#     import horovod.tensorflow as hvd

from diffsim.config.mode import ModeKind

from diffsim.simulator import init_simulator
# from diffsim.simulator import NEW_Simulator, init_NEW_simulator
from diffsim.simulator import batch_update_rng_keys

from diffsim.utils import init_mpi, discover_local_rank
from diffsim.utils import summary, model_summary
from diffsim.utils import init_checkpointer
from diffsim.utils import set_compute_parameters, configure_logger, should_do_io
from diffsim.utils import logging


from diffsim.dataloaders import build_dataloader

from diffsim.utils import comparison_plots

def interupt_handler( sig, frame):
    logger = logging.getLogger()

    logger.info("Finishing iteration and snapshoting weights...")
    global active
    active = False

@jit
def update_summary_params(metrics, params):
    # Add the diffusion:
    metrics["physics/diffusion_0"]  = params["params"]["diff"]["diffusion"][0]
    metrics["physics/diffusion_1"]  = params["params"]["diff"]["diffusion"][1]
    metrics["physics/diffusion_2"]  = params["params"]["diff"]["diffusion"][2]
    metrics["physics/el_gain"]      = params["params"]["el_gain"][0]
    # metrics["physics/el_spread"]   = params["el_spread"]["sipm_s2"]["el_spread"][0]
    # metrics["physics/lifetime"]     = params["params"]["lifetime"]["lifetime"][0]
    # metrics["physics/nn_bin_sigma"] = params["nn_bin_sigma"]["pmt_s2"]["nn_bin_sigma"][0]

    return metrics

@jit
def scale_data(input_batch, prefactor):
    # This scales the target data to whatever prefactor we are using:
    for key in input_batch.keys():
        if key in prefactor.keys():
            input_batch[key] = prefactor[key]*input_batch[key]

    return input_batch

@hydra.main(version_base = None, config_path="../diffsim/config/recipes")
def main(cfg : OmegaConf) -> None:

    # Extend the save path:
    # cfg.save_path = cfg.save_path + f"/{cfg.hamiltonian.form}/"
    # cfg.save_path = cfg.save_path + f"/{cfg.sampler.n_particles}particles/"
    # cfg.save_path = cfg.save_path + f"/{cfg.optimizer.solver}_solver/"
    cfg.save_path = cfg.save_path + f"/{cfg.run.id}/"



    # Prepare directories:
    work_dir = pathlib.Path(cfg.save_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(cfg.save_path + "/log/")
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = pathlib.Path(cfg["model_name"])

    MPI_AVAILABLE, rank, size = init_mpi(cfg.run.distributed)

    # Here we initialize the checkpointer functions:
    if should_do_io(MPI_AVAILABLE, rank):
        save_weights, restore_weights = init_checkpointer(work_dir)


    # Figure out the local rank if MPI is available:
    if MPI_AVAILABLE:
        local_rank = discover_local_rank()
    else:
        local_rank = 0
    # model_name = config.model_name



    configure_logger(log_dir, MPI_AVAILABLE, rank)

    logger = logging.getLogger()
    logger.info("")
    logger.info("\n" + OmegaConf.to_yaml(cfg))



    # Training generator_state variables:
    global active
    active = True

    # Active is a global so that we can interrupt the train loop gracefully
    signal.signal(signal.SIGINT, interupt_handler)

    target_device = set_compute_parameters(local_rank)

    # At this point, jax has already allocated some memory on the wrong device.


    # Always construct a dataloader:
    dataloader  = build_dataloader(cfg, MPI_AVAILABLE)


    # If it's just IO testing, run that here then exit:
    from diffsim.config.mode import ModeKind
    if cfg.mode.name == ModeKind.iotest:
        iotest(dataloader, cfg)
        return
    with jax.default_device(target_device):

        # Initialize the global random seed:
        if cfg.seed == -1:
            global_random_seed = int(time.time())
        else:
            global_random_seed = cfg.seed

        if MPI_AVAILABLE and size > 1:
            if rank == 0:
                # Create a single master key
                master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)
            else:
                # This isn't meaningful except as a placeholder:
                master_key = jax.device_put(random.PRNGKey(0), target_device)

            # Here, sync up all ranks to the same master key
            import mpi4jax
            from mpi4py import MPI
            master_key, token = mpi4jax.bcast(master_key, root=0, comm=MPI.COMM_WORLD)
        else:
            master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)

        # Initialize the model:
        example_data = next(dataloader)

        out_positions, mask = eg(example_data['e_deps'], cfg.physics.electron_generator)
        example_data['e_deps'] = out_positions
        example_data['mask'] = mask
    


        sim_func, sim_params, next_rng_keys, simulator_str = init_simulator(master_key, cfg, example_data)


        logger.info(simulator_str)

        n_parameters = 0
        flat_params, tree_def = tree_util.tree_flatten(sim_params)
        for p in flat_params:
            n_parameters += numpy.prod(numpy.asarray(p.shape))
        logger.info(f"Number of parameters in this network: {n_parameters}")



        # Import the right trainer and optimizer functions based on training mode:
        if cfg.mode.name == ModeKind.supervised:
            from diffsim.trainers.supervised_trainer import build_optimizer, close_over_training_step
        elif cfg.mode.name == ModeKind.unsupervised:
            from diffsim.discriminator import init_discriminator
            disc_func, disc_params, d_str = init_discriminator(master_key, cfg, example_data)
            from diffsim.trainers.unsupervised_trainer import build_optimizer, close_over_training_step
            logger.info(d_str)

        optimizer, opt_state = build_optimizer(cfg, sim_params)

        global_step = 0

        # Create a train generator_state:
        generator_state = train_state.TrainState(
            step      = global_step,
            apply_fn  = sim_func,
            params    = sim_params,
            tx        = optimizer,
            opt_state = opt_state,
        )

        # In unsupervised mode, create a second train state to hold the discriminator properties:
        if cfg.mode.name == ModeKind.unsupervised:
            disc_optimizer, disc_opt_state = build_optimizer(cfg, disc_params)
            discriminator_state = train_state.TrainState(
                step      = global_step,
                apply_fn  = disc_func,
                params    = disc_params,
                tx        = disc_optimizer,
                opt_state = disc_opt_state
            )
        else:
            # 
            discriminator_state = None

        #     # self.trainer = self.build_trainer(batch, self.simulator_fn, self.simulator_params)


        train_step = close_over_training_step(cfg, MPI_AVAILABLE)


        # Create a summary writer:
        if should_do_io(MPI_AVAILABLE, rank):
            writer = SummaryWriter(log_dir, flush_secs=20)
        else:
            writer = None

        # Restore the weights

        if should_do_io(MPI_AVAILABLE, rank):
            r_state, r_disc_state = restore_weights(generator_state, discriminator_state)

            if r_state is not None:
                generator_state  = r_state
                discriminator_state = r_disc_state
                logger.info("Loaded weights, optimizer and global step!")
            else:
                logger.info("Failed to load weights!")



        if MPI_AVAILABLE and size > 1:
            logger.info("Broadcasting initial model and opt state.")
            from diffsim.utils.mpi import broadcast_train_state
            generator_state = broadcast_train_state(generator_state)
            if discriminator_state is not None:
                discriminator_state = broadcast_train_state(discriminator_state)
            logger.info("Done broadcasting initial model and optimizer state.")


        dl_iterable = dataloader
        comp_data = next(dl_iterable)
        comp_out_positions, comp_mask = eg(comp_data['e_deps'], cfg.physics.electron_generator)
        comp_data['e_deps'] = comp_out_positions
        comp_data['mask'] = comp_mask


        prefactor = {
                "S2Pmt" : 1.,
                "S2Si"  : 1.
            }

        # for key in comp_data.keys():
        #     if key in prefactor.keys():
        #         comp_data[key] = prefactor[key]*comp_data[key]



        end = None

        while generator_state.step < cfg.run.iterations:

            if not active: break


            # Add comparison plots every N iterations
            if generator_state.step % cfg.run.image_iteration == 0:
                if should_do_io(MPI_AVAILABLE, rank):
                    save_dir = cfg.save_path / pathlib.Path(f'comp/{generator_state.step}/')
                    # jax.tree_util.tree_map( lambda x : x.shape,
                                            # generator_state.params)
                    
                    simulated_data = generator_state.apply_fn(
                        generator_state.params,
                        comp_data['e_deps'], comp_data['mask'],
                        rngs=next_rng_keys
                    )



                    # Remove the prefactor on simulated data for this:
                    # It's not applied to the comp data, but we have to scale up the output
                    # according to the prefactor
                    for key in simulated_data.keys():
                        if key in prefactor.keys():
                            simulated_data[key] = simulated_data[key] / prefactor[key]

                    comparison_plots(save_dir, simulated_data, comp_data)

            metrics = {}
            start = time.time()

            batch = next(dl_iterable)

            out_positions, mask = eg(batch['e_deps'], cfg.physics.electron_generator)
            batch['e_deps'] = out_positions
            batch['mask'] = mask

            batch = scale_data(batch, prefactor)


            
        

            if cfg.run.profile:
                if should_do_io(MPI_AVAILABLE, rank):
                    jax.profiler.start_trace(str(cfg.save_path) + "profile")


            metrics["io_time"] = time.time() - start

            # Split the keys:
            next_rng_keys = batch_update_rng_keys(next_rng_keys)

            if discriminator_state is not None:
                generator_state, discriminator_state, loss, train_metrics = train_step(
                    generator_state,
                    discriminator_state,
                    batch,
                    next_rng_keys)

            else:
                generator_state, loss, train_metrics = train_step(
                    generator_state,
                    batch,
                    next_rng_keys)

            # Remove the residual effect on the metrics:
            for key in train_metrics.keys():
                prefactor_key = key.replace("residual/", "")
                if prefactor_key in prefactor.keys():
                    train_metrics[key] = train_metrics[key] / prefactor[prefactor_key]



            metrics.update(train_metrics)
            # Add to the metrics the physical parameters:
            metrics = update_summary_params(metrics, generator_state.params)
            if cfg.run.profile:
                if should_do_io(MPI_AVAILABLE, rank):
                    x.block_until_ready()
                    jax.profiler.save_device_memory_profile(str(cfg.save_path) + f"memory{generator_state.step}.prof")



            metrics.update({"loss" : loss})





            if generator_state.step % cfg.run.checkpoint  == 0:
                if should_do_io(MPI_AVAILABLE, rank):
                    save_weights(generator_state, discriminator_state)

            if cfg.run.profile:
                if should_do_io(MPI_AVAILABLE, rank):
                    jax.profiler.stop_trace()


            end = time.time()
            metrics['time'] = time.time() - start
            metrics['img_per_sec'] = size * cfg.run.minibatch_size / metrics['time']

            if should_do_io(MPI_AVAILABLE, rank):
                summary(writer, metrics, generator_state.step)

            # Iterate:
            # generator_state.step += 1

            if generator_state.step % 1 == 0:
                logger.info(f"step = {generator_state.step}, loss = {metrics['loss']:.3f}")
                logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io) - {metrics['img_per_sec']:.3f} Img/s")

    # Save the weights at the very end:
    if should_do_io(MPI_AVAILABLE, rank):
        try:
            save_weights(generator_state)
        except:
            pass

import numpy as np

# @profile
def eg(energies_and_positions, cfg, M=120000):

    # numpy.save("test_input", energies_and_positions)
    # First, split the energy and positions apart:
    positions = energies_and_positions[:,:,0:3]
    energies  = energies_and_positions[:,:,-1]
    batch_size= energies_and_positions.shape[0]

    normal_draws = np.random.normal(size=energies.shape)

    # Get the number of electrons per position:
    n      = energies * 1000.*1000. / cfg.p1
    sigmas = np.sqrt( n * cfg.p2)

    n_electrons = (sigmas*normal_draws + n).astype(np.int32)



    n_per_batch = n_electrons.sum(axis=-1)

    out_positions = np.zeros((batch_size, M, 3))
    mask = np.zeros((batch_size, M, 1))

    for b in range(batch_size):
        start = 0;
        # Collect the postions, broadcasted to the right shape, and then concatenate them
        # into an array.  Then write the array into the output:

        broadcasted_points = []
        for i in range(n_electrons.shape[-1]):
            n = n_electrons[b][i]
            broadcasted_points.append( np.broadcast_to(positions[b,i], (n, 3)) )
            # start = start + n_electrons[b][i]

        merged_points = np.concatenate(broadcasted_points, axis=0)
        total_n = merged_points.shape[0]
        out_positions[b, 0:total_n] = merged_points                              


        # for i in range(n_electrons.shape[-1]):
        #     end = start + n_electrons[b][i]
        #     out_positions[b, start:end] = positions[b,i]
        #     start = start + n_electrons[b][i]

        mask[b][0:total_n] = 1.0


    return out_positions, mask


def iotest(dataloader, config):

    logger = logging.getLogger()

    global active
    active = True

    dl_iterable = dataloader

    global_step = 0

    while global_step < config.run.iterations:

        if not active: break

        metrics = {}
        start = time.time()

        batch = next(dl_iterable)

        out_positions, mask = eg(batch['e_deps'], config.physics.electron_generator)
        batch['e_deps'] = out_positions
        batch['mask'] = mask


        metrics["io_time"] = time.time() - start

        metrics['time'] = time.time() - start


        if global_step % 1 == 0:
            logger.info(f"step = {global_step}")
            logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

        # Iterate:
        global_step += 1



if __name__ == "__main__":
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]


    main()
