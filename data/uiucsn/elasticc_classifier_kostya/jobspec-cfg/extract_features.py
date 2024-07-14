#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lcdata
import light_curve as licu
import numpy as np
import parsnip
from astropy.coordinates import SkyCoord
from astropy.table import Table
from dustmaps import sfd
from dustmaps.config import config as dustmaps_config

from util import SNANA_TO_TAXONOMY


# From the ParSNIP paper
PARSNIP_FEATURES = [
    'color',
    'color_error',
    's1',
    's1_error',
    's2',
    's2_error',
    's3',
    's3_error',
    # 'luminosity',
    # 'luminosity_error',
    'reference_time_error',
]

BANDS_SNANA = 'ugrizY'
BANDS_PARSNIP = ['lsst' + band for band in BANDS_SNANA.lower()]
LEN_BANDS = len(BANDS_PARSNIP)

MAGERR_COEFF = 2.5 / np.log(10.0)

MJD0 = 60000.0

SATURATION_FLUX = 1e5

# All smaller redshifts are considered as unknown
MIN_REDSHIFT = 1e-2


class MetaExtractor():
    features = (
        ['abs_gal_b', 'redshift', 'mwebv']
        + list(chain.from_iterable([f'hostgal{i}_zspec', f'hostgal{i}_zspec_err', f'hostgal{i}_ellipticity',
                                    f'hostgal{i}_sqradius', f'hostgal{i}_zphot', f'hostgal{i}_zphot_err',
                                    f'hostgal{i}_snsep'] + [f'hostgal{i}_mag_{b}' for b in BANDS_SNANA]
                                   for i in ['', '2']))
    )
    size = len(features)

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is not None:
            dustmaps_config['data_dir'] = cache_dir
        sfd.fetch()
        self.sfd_query = sfd.SFDQuery()

    def _prepare(self, coord: SkyCoord, **kwargs: Dict[str, np.ndarray]):
        abs_gal_b = np.abs(coord.galactic.b.deg)
        mwebv = self.sfd_query(coord)
        return dict(abs_gal_b=abs_gal_b, mwebv=mwebv, **kwargs)

    def _extract_lcdata(self, meta: Table) -> Dict[str, np.ndarray]:
        kwargs = {'coord': SkyCoord(ra=meta['ra'], dec=meta['dec'], unit='deg'), 'redshift': meta['redshift'],
                  'hostgal_zspec': meta['HOSTGAL_SPECZ'], 'hostgal2_zspec': meta['HOSTGAL2_SPECZ'],
                  'hostgal_zspec_err': meta['HOSTGAL_SPECZ_ERR'], 'hostgal2_zspec_err': meta['HOSTGAL2_SPECZ_ERR'],
                  'hostgal_zphot': meta['HOSTGAL_PHOTOZ'], 'hostgal2_zphot': meta['HOSTGAL2_PHOTOZ'],
                  'hostgal_zphot_err': meta['HOSTGAL_PHOTOZ_ERR'], 'hostgal2_zphot_err': meta['HOSTGAL2_PHOTOZ_ERR']}
        for i in ['', '2']:
            for prop in ['ellipticity', 'sqradius', 'snsep']:
                kwargs[f'hostgal{i}_{prop}'] = meta[f'HOSTGAL{i}_{prop.upper()}']
            for b in BANDS_SNANA:
                kwargs[f'hostgal{i}_mag_{b}'] = meta[f'HOSTGAL{i}_MAG_{b}']
        return self._prepare(**kwargs)

    def __call__(self, meta: Table, *, schema:str):
        try:
            feature_dict = getattr(self, f'_extract_{schema}')(meta)
        except AttributeError as e:
            raise ValueError(f'schema {schema!r} is not supported') from e
        table = Table({k: feature_dict[k] for k in self.features})
        return table


@dataclass
class LcExtractor():
    s2n: float = 5.0

    mag_extractor = licu.Extractor(
        licu.AndersonDarlingNormal(),
        licu.Bins(
            features=[
                licu.BeyondNStd(1.0),
                licu.BeyondNStd(2.0),
                licu.EtaE(),
                licu.Kurtosis(),
                licu.LinearTrend(),
                licu.MaximumSlope(),
                licu.MinimumTimeInterval(),
                licu.ObservationCount(),
                licu.Skew(),
            ],
            window=1.0,
            offset=0.0,
        ),
        licu.Duration(),
        licu.InterPercentileRange(0.01),
        licu.LinearFit(),
        licu.MaximumTimeInterval(),
        licu.ObservationCount(),
        licu.Periodogram(
            peaks=1,
            resolution=10,
            max_freq_factor=2,
            nyquist='median',
            fast=True,
            features=[
                licu.Median(),
                licu.PercentDifferenceMagnitudePercentile(0.25)
            ],
        ),
        licu.ReducedChi2(),
        licu.StetsonK(),
        licu.WeightedMean(),
    )

    flux_extractor = licu.Extractor(
        licu.Kurtosis(),
        licu.Skew(),
    )

    full_flux_extractor = licu.Extractor(
        licu.BazinFit('mcmc-lmsder', mcmc_niter=1 << 10, lmsder_niter=20),
        licu.VillarFit('mcmc-lmsder', mcmc_niter=1 << 10, lmsder_niter=20),
    )
    
    def __post_init__(self):
        self.mag_names = [f'mag_{name}_{band}' for band in BANDS_PARSNIP for name in self.mag_extractor.names]
        self.mag_size = len(self.mag_names)
        self.flux_names = [f'flux_{name}_{band}' for band in BANDS_PARSNIP for name in self.flux_extractor.names]
        self.flux_size = len(self.flux_names)
        self.full_flux_names = [f'fullflux_{name}_{band}'
                                for band in BANDS_PARSNIP
                                for name in self.full_flux_extractor.names]
        self.full_flux_size = len(self.full_flux_names)
        self.names = self.mag_names + self.flux_names + self.full_flux_names
        self.size = len(self.names)
        assert self.mag_size + self.flux_size + self.full_flux_size == self.size

    def prepare_lc_lcdata(self, lc: Table, *, non_det: bool) -> Table:
        lc = lc[lc['flux'] <= SATURATION_FLUX]
        # we don't support mags for non_det yet
        if not non_det:
            lc = lc[lc['flux'] / lc['fluxerr'] > self.s2n]
            lc['mag'] = 27.5 - 2.5 * np.log10(lc['flux'])
            lc['magerr'] = MAGERR_COEFF * lc['fluxerr'] / lc['flux']
        lc['time'] = np.asarray(lc['time'] - MJD0, dtype=np.float32)
        return lc

    def _prepare_lc_funcs(self, schema: str):
        try:
            return getattr(self, f'prepare_lc_{schema}')
        except AttributeError as e:
            raise ValueError(f'schema {schema!r} is not supported') from e

    @staticmethod
    def lcs_mag_list_tuples(lcs: Iterable[Table]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [(lc['time'], lc['mag'], lc['magerr']) for lc in lcs]
    
    @staticmethod
    def lcs_flux_list_tuples(lcs: Iterable[Table]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [(lc['time'], lc['flux'], lc['fluxerr']) for lc in lcs]
    
    def __call__(self, light_curves: Sequence[Table],
                 *, schema: str, chunk_size: int = 1 << 10, n_jobs: int = -1, out=None) -> np.ndarray:
        prepare_lc_func = self._prepare_lc_funcs(schema)
        n_light_curves = len(light_curves)
        if out is None:
            out = np.empty((n_light_curves, self.size), dtype=np.float32)
        else:
            assert out.shape == (n_light_curves, self.size)
            assert out.dtype.type is np.float32, f'out dtype is {out.dtype!r}, but np.float32 is required'
        for i in range(0, n_light_curves, chunk_size):
            lc_idx = slice(i, i + chunk_size)
            # Split each light curve to LEN_BANDS light curves
            lcs = list(lc[lc['band'] == band]
                       for lc in map(lambda lc: prepare_lc_func(lc, non_det=False), light_curves[lc_idx])
                       for band in BANDS_PARSNIP)
            full_lcs = list(lc[lc['band'] == band]
                            for lc in map(lambda lc: prepare_lc_func(lc, non_det=True), light_curves[lc_idx])
                            for band in BANDS_PARSNIP)
            mag_slice = slice(0, self.mag_size)
            flux_slice = slice(mag_slice.stop, mag_slice.stop + self.flux_size)
            full_flux_slice = slice(flux_slice.stop, flux_slice.stop + self.full_flux_size)
            out[lc_idx, mag_slice] = self.mag_extractor.many(
                self.lcs_mag_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.mag_size)
            out[lc_idx, flux_slice] = self.flux_extractor.many(
                self.lcs_flux_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.flux_size)
            out[lc_idx, full_flux_slice] = self.full_flux_extractor.many(
                self.lcs_flux_list_tuples(full_lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.full_flux_size)
        return out
    

def parse_args(argv=None):
    parser = ArgumentParser("Generate ParSNIP features for the dataset")
    parser.add_argument('-i', '--input', required=True,
                        help='HDF5 file containing lcdata Dataset to extract features, may be used multiple times')
    parser.add_argument('-m', '--model', required=True, help='ParSNIP model file to use')
    parser.add_argument('-o', '--output', default='features', help='output directory')
    parser.add_argument('--device', default='cuda', help='PyTroch device')
    parser.add_argument('--s2n', default=LcExtractor.s2n, type=float, help='S/N threshold for light-curve features')
    args = parser.parse_args(argv)
    return args


def filename_to_snana(fname):
    s = Path(fname).stem
    s = re.sub(r'_count\d+', '', s)
    s = re.sub(r'_z[\d.]+', '', s)
    return s


def fix_dataset(dataset: lcdata.Dataset) -> lcdata.Dataset:
    dataset.meta['redshift'] = np.where(dataset.meta['redshift'] >= MIN_REDSHIFT,
                                        dataset.meta['redshift'], MIN_REDSHIFT)
    return dataset


def main(argv=None):
    args = parse_args(argv)

    dataset = lcdata.read_hdf5(args.input)
    dataset = fix_dataset(dataset)

    lc_extractor = LcExtractor(s2n=args.s2n)
    meta_extractor = MetaExtractor()

    lc_slice = slice(0, lc_extractor.size)
    parsnip_slice = slice(lc_slice.stop, lc_slice.stop + len(PARSNIP_FEATURES))
    meta_slice = slice(parsnip_slice.stop, parsnip_slice.stop + meta_extractor.size)
    all_size = meta_slice.stop
    all_features = np.empty((len(dataset), all_size), dtype=np.float32)

    lc_features = all_features[:, lc_slice]
    lc_extractor(dataset.light_curves, schema='lcdata', out=lc_features)

    parsnip_model = parsnip.load_model(args.model, device=args.device)
    predictions = parsnip_model.predict_dataset(dataset)
    parsnip_features = all_features[:, parsnip_slice]
    parsnip_features[:] = np.stack([np.asarray(predictions[f], dtype=np.float32) for f in PARSNIP_FEATURES], axis=-1)

    meta_features = all_features[:, meta_slice]
    meta_table = meta_extractor(dataset.meta, schema='lcdata')
    meta_features[:] = np.stack([np.asarray(column, dtype=np.float32) for column in meta_table.itercols()], axis=-1)

    object_ids = predictions['object_id']
    snana_model_name = filename_to_snana(args.input)
    types = np.full(len(dataset), SNANA_TO_TAXONOMY[snana_model_name])

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir.joinpath(f'{snana_model_name}_features.npy'), all_features)
    np.save(output_dir.joinpath(f'{snana_model_name}_types.npy'), types)
    np.save(output_dir.joinpath(f'{snana_model_name}_ids.npy'), object_ids)
    with open(output_dir.joinpath('names.txt'), 'w') as fh:
        for name in lc_extractor.names + PARSNIP_FEATURES + meta_extractor.features:
            fh.write(f'{name}\n')


if __name__ == '__main__':
    main()
