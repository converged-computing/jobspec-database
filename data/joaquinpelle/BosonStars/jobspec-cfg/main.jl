include("src/BosonStars.jl")

LBSmodels = LBS(1:3)
SBSmodels = SBS(1:3)
BHmodel = BH()

LBSrunset = CameraRunSet(models = LBSmodels,
                    inclinations = [5, 45, 85],
                    number_of_pixels_per_side = 1200,
                    observation_radius = 1000.0)

SBSrunset = CameraRunSet(models = SBSmodels,
                    inclinations = [5, 45, 85],
                    number_of_pixels_per_side = 1200,
                    observation_radius = 1000.0)

BHrunset = CameraRunSet(models = BHmodel,
                    inclinations = [5, 45, 85],
                    number_of_pixels_per_side = 1200,
                    observation_radius = 1000.0)

radiative_transfer([LBSrunset, SBSrunset, BHrunset]; reltol=1e-6, abstol=1e-6)

LBScorona_runset = CoronaRunSet(models = LBSmodels,
                    heights = [2.5, 5.0, 10.0],
                    spectral_index = 2.0,
                    number_of_packets = 5000000,
                    number_of_radial_bins = 50)

SBScorona_runset = CoronaRunSet(models = SBSmodels,
                    heights = [2.5, 5.0, 10.0],
                    spectral_index = 2.0,
                    number_of_packets = 5000000,
                    number_of_radial_bins = 50)

BHcorona_runset = CoronaRunSet(models = BHmodel,
                    heights = [2.5, 5.0, 10.0],
                    spectral_index = 2.0,
                    number_of_packets = 5000000,
                    number_of_radial_bins = 50)

radiative_transfer([LBScorona_runset, SBScorona_runset, BHcorona_runset]; reltol=1e-5, abstol=1e-5)

temperature_plot(LBSrunset, SBSrunset, BHrunset; figname="plots/temperature.pdf")

_, zmax = zextrema([LBSrunset, SBSrunset, BHrunset])
bolometric_intensity_mosaic(LBSrunset; zmax = zmax, figname = "plots/LBS_mosaic.pdf")
bolometric_intensity_mosaic(SBSrunset; zmax = zmax, figname = "plots/SBS_mosaic.pdf")
bolometric_intensity_mosaic(BHrunset; zmax = zmax, figname = "plots/BH_mosaic.pdf")

ε = eV_to_erg(1.0)
observation_energies = ε*exp10.(range(0.0, stop=4.0, length=20))
spectrum_mosaic(LBSrunset, SBSrunset, BHrunset; observation_energies = observation_energies, figname = "plots/spectrum_mosaic.pdf")

emissiviy_profile_mosaic(LBScorona_runset, SBScorona_runset, BHcorona_runset; figname = "plots/emissivity_mosaic.pdf")
emissiviy_profile_mosaic_focused(SBScorona_runset, BHcorona_runset; figname = "plots/emissivity_mosaic_focused.pdf")

line_emission_mosaic(SBSrunset, BHrunset, SBScorona_runset, BHcorona_runset; number_of_energy_bins = 40, figname = "plots/line_emission_mosaic.pdf")