from pyuvdata import UVData

uvd = UVData()

#uvd.read("/lustre/aoc/projects/hera/H4C/postprocessing/lstbin/after_filtering_before_red_average/all-bands-long-delay-clean/zen.LST.0.00000.sum.all-bands-allbls-long-delay-clean.foreground_filled.xtalk_filtered.chunked.uvh5", freq_chans=range(180, 265))

uvd.read("viscatBC_stretch0.02.uvh5")
uvd.write_uvfits("x.uvfits", force_phase=True, spoof_nonessential=True)
