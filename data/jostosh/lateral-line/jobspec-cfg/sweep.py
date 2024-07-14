from latline.experiment_config import ExperimentConfig
from argparse import ArgumentParser
import subprocess


sweeps = [
    # dict(model='parallel', merge_at=4, logdir_base_suffix='parallel_default'),
    # dict(model='parallel', merge_at=3, logdir_base_suffix='parallel_m3'),
    # dict(model='parallel', merge_at=2, logdir_base_suffix='parallel_m2'),
    # dict(model='cross', logdir_base_suffix='cross_default'),
    # dict(model='parallel', logdir_base_suffix='parallel_multi_range', multi_range=True),
    # # dict(model='parallel', logdir_base_suffix='parallel_multi_range_tr', multi_range=True, multi_range_trainable=True),
    # dict(model='parallel', logdir_base_suffix='parallel_dense', dense=True),
    # dict(model='parallel', lr=1e-4, logdir_base_suffix='l2', loss='l2'),
    # dict(model='parallel', merge_at=1, logdir_base_suffix='parallel_m1'),
    # dict(model='parallel', merge_at=0, logdir_base_suffix='parallel_m0'),
    # dict(model='parallel', noise=0.001, logdir_base_suffix='parallel_n0001'),
    # dict(model='parallel', noise=0.1, logdir_base_suffix='parallel_n01'),
    # dict(model='parallel', noise=0.2, logdir_base_suffix='parallel_n02'),
    # # dict(model='parallel', optimizer='yellow', logdir_base_suffix='yellow1', lr=1e-3),
    # # dict(model='parallel', optimizer='yellow', logdir_base_suffix='yellow2', lr=5e-3),
    # # dict(model='parallel', optimizer='adam', logdir_base_suffix='adam2', lr=5e-3),
    # dict(model='parallel', merge_at=3, logdir_base_suffix='tau3', tau=3),
    # dict(model='parallel', merge_at=3, logdir_base_suffix='tau2', tau=2),
    # dict(model='parallel', merge_at=3, logdir_base_suffix='tau1', tau=1),
    # dict(model='parallel', logdir_base_suffix='tau4', tau=4),
    # dict(model='parallel', logdir_base_suffix='best', multi_range=True, tau=1),
    # dict(model='parallel', merge_at=5, logdir_base_suffix='parallel_m5'),
    # dict(model='parallel', merge_at=3, tau=2, logdir_base_suffix='king', multi_range=True, resolution=128, n_sensors=128),
    dict(model='parallel', log_sub='no_sharing', share=False)
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--until", type=int, default=len(sweeps))
    parser.add_argument("--subset", type=int, default=[], nargs='+')
    parser.add_argument("--fnm", type=str, default='multisphere')
    args = parser.parse_args()
    config = ExperimentConfig()

    sweep_indices = range(args.start, args.until) if not args.subset else args.subset

    for i in sweep_indices:
        config.__dict__.update(sweeps[i])
        command = ['python', './train.py', '--fnm={}'.format(args.fnm)] + \
                  ['--{}={}'.format(k, v) if not isinstance(v, bool) else '--{}'.format(k) for k, v in sweeps[i].items()]
        print("Initiating ", ' '.join(command))
        try:
            subprocess.run(command)
        except:
            subprocess.call(command)
