#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import fits
from pathlib import Path
import numpy as np

def fix_header(
    file_name: Path,
):

    print(f"Fixing header for {file_name}")
    with fits.open(file_name) as hdu_list:
        new_header = hdu_list[0].header
        data_cube = hdu_list[0].data

    tmp_header = new_header.copy()
    # Need to swap NAXIS 3 and 4 to make LINMOS happy - booo
    for a, b in ((3, 4), (4, 3)):
        new_header[f"CTYPE{a}"] = tmp_header[f"CTYPE{b}"]
        new_header[f"CRPIX{a}"] = tmp_header[f"CRPIX{b}"]
        new_header[f"CRVAL{a}"] = tmp_header[f"CRVAL{b}"]
        new_header[f"CDELT{a}"] = tmp_header[f"CDELT{b}"]
        new_header[f"CUNIT{a}"] = tmp_header[f"CUNIT{b}"]

    # Cube is currently STOKES, FREQ, RA, DEC - needs to be FREQ, STOKES, RA, DEC
    data_cube = np.moveaxis(data_cube, 1, 0)

    new_file_name = file_name.with_suffix(".fixed.fits")
    print(f"Writing to {new_file_name}")

    fits.writeto(new_file_name, data_cube, new_header, overwrite=True)

    print("Done")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix the header of a cube")
    parser.add_argument("file_name", type=Path, help="The file to fix")
    args = parser.parse_args()

    fix_header(args.file_name)

if __name__ == "__main__":
    main()