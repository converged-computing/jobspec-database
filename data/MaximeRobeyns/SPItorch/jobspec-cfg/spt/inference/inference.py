# SPItorch: Inference of Stellar Population Properties in PyTorch
#
# Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This file implements the full training procedure.

It then performs inference on a catalogue.
"""

import logging
import torch as t
import numpy as np

from torch.utils.data import DataLoader

from spt.types import Tensor
from spt.inference.san import SAN, PModel
from spt.load_photometry import load_simulated_data, get_norm_theta, load_real_data


if __name__ == '__main__':

    import spt.config as cfg

    ip = cfg.InferenceParams()
    fp = cfg.ForwardModelParams()

    # Maximum-likelihood training of approximate posterior --------------------

    mp = cfg.SANParams()
    Q = SAN(mp)
    logging.info(f'Initialised {Q.name} model')

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=Q.params.batch_size,
        phot_transforms=[lambda x: t.from_numpy(np.log(x))],
        theta_transforms=[get_norm_theta(fp)],
    )
    logging.info('Created data loaders')

    Q.offline_train(train_loader, ip)
    logging.info('ML training of approximate posterior complete.')

    # Maximum-likelihood training of neural likelihood ------------------------

    lp = cfg.SANLikelihoodParams()
    P = PModel(lp)
    logging.info(f'Initialised neural likelihood: {P.name}')

    ip.ident = "ML_likelihood"
    P.offline_train(train_loader, ip)
    logging.info('ML training of neural likelihood complete.')

    # =========================================================================
    # Temporarily commented out for cluster job:
    # =========================================================================

    # # HMC update procedure with simulated data -------------------------------

    # # Here we create smaller data loaders for the HMC update procedure, due to
    # # VRAM considerations.
    # sim_train_loader, _ = load_simulated_data(
    #     path=ip.dataset_loc,
    #     split_ratio=ip.split_ratio,
    #     batch_size=700,
    #     phot_transforms=[lambda x: t.from_numpy(np.log(x))],
    #     theta_transforms=[get_norm_theta(fp)],
    # )
    # logging.info('Created smaller data loaders')

    # ip.ident = ip.hmc_update_sim_ident
    # Q.hmc_retrain_procedure(sim_train_loader, ip, P=P,
    #                         epochs=ip.hmc_update_sim_epochs,
    #                         K=ip.hmc_update_sim_K, lr=3e-4, decay=1e-4,
    #                         logging_frequency=10)
    # logging.info('HMC update on sim data complete.')

    # HMC update procedure with real data -------------------------------

    real_train_loader, _ = load_real_data(
        path=ip.catalogue_loc,
        filters=fp.filters,
        split_ratio=ip.split_ratio,
        batch_size=600,
        transforms=[t.from_numpy],
        x_transforms=[np.log],
    )
    logging.info('Created real data loader')

    ip.ident = ip.hmc_update_real_ident
    Q.hmc_retrain_procedure(real_train_loader, ip, P=P,
                            epochs=ip.hmc_update_real_epochs,
                            K=ip.hmc_update_real_K, lr=3e-4, decay=1e-4,
                            logging_frequency=10)
    logging.info('HMC update on real data complete.')
