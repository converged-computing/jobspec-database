#!/usr/bin/env python

import sys 
#sys.path.append('/cluster/home/kristokv/opendrift')
#sys.path.append('../opendrift')
import os
from datetime import datetime, date, time, timedelta
import calendar
from dateutil.relativedelta import relativedelta
import numpy as np
#import cartopy.io.shapereader as shpreader
#from pyproj import Proj

from opendrift.readers import reader_ROMS_native
#from opendrift.readers import reader_netCDF_CF_generic 
#from opendrift.readers import reader_shape
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.sedimentdrift import SedimentDrift

#### Define run
year = '2020'
startDay= year + '-2-1-1'
endDay=year + '-12-31-1'
#endDay=year + '-2-2-1'
startTime = datetime.strptime(startDay, '%Y-%m-%d-%H')
endTime = datetime.strptime(endDay, '%Y-%m-%d-%H')
print ("Run planned from %s to %s"%(startTime,endTime))

sinkingParticles = False

if sinkingParticles:
    run_name = 'sinkingParticles'
    terminalVelocity = -.0008  # Settling speed in m/s (.001 = 1 mm/s) 
    outfile = '/cluster/projects/nn8103k/NIVA-NOLA/results/%s_%sms_%s_to_%s.nc'%(run_name,terminalVelocity,startDay,endDay)
    o = SedimentDrift(loglevel=20)
else:
    run_name = 'neutralParticles'
    terminalVelocity = 0  # Neutral particles
    outfile = '/cluster/projects/nn8103k/NIVA-NOLA/results/%s_%s_to_%s.nc'%(run_name,startDay,endDay)
    o = OceanDrift(loglevel=20)

print ("Output will be stored in %s"%(outfile))

#### Readers

#Find forcing files needed based on days:
day_i = startTime
day_range = [startTime]
while day_i<(endTime + relativedelta(days=1)):
    day_i = day_i + relativedelta(days=1)
    day_range.append(day_i)

i=0
pattern_norkyst=[]
while i < len(day_range)-1:
	day_i = day_range[i]
	day_j = day_range[i+1]
	x='/cluster/work/support/jonal/A11/norfjords_160m_his.nc4_'+str(day_i.year)+str(day_i.month).zfill(2)+str(day_i.day).zfill(2)+'01-'+str(day_j.year)+str(day_j.month).zfill(2)+str(day_j.day).zfill(2)+'00'
	pattern_norkyst.append(x)
	i += 1

#proj = "+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=1848640 +y_0=1432320 +lon_0=70"
reader_norkyst = reader_ROMS_native.Reader(pattern_norkyst)
o.add_reader([reader_norkyst]) 

#Adding readers "lazily"
#o.add_readers_from_list(pattern_norkyst)
       
#### Configuration 
#o.set_config('general:use_auto_landmask',False) # Override default landmask
o.set_config('general:coastline_action', 'previous') # Jump back to previous position when meeting coast
o.set_config('drift:vertical_mixing',True) # Move particles vertically according to eddy diffusivity and buoyancy 
o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Sundby1983') # Vertical diffusivity is included in the Norkyst files, but giving unrealistic values, using this instead
o.set_config('drift:vertical_advection',True) # Move particles vertically according to vertical ocean currents, negligable compared to vertical diffusivity
o.set_config('drift:horizontal_diffusivity', 10) # Horizontal diffusion in m2/s, to compensate for movements not resolved by the ocean model
o.set_config('general:seafloor_action', 'previous') # ‘previous’: particles are moved back to previous location. 

if sinkingParticles:
    o.set_config('vertical_mixing:resuspension_threshold', .15) # 0.2 is now set to default

#'lift_to_seafloor': particles is moved with current direction and upward to shallower sea 
# 'previous': kept at seafloor, cannot move if meeting shallower area.

#o.set_config('environment:fallback:sea_surface_wave_stokes_drift_x_velocity',.2) #Not used
#o.set_config('drift:current_uncertainty',2) #Not necessary when adding horizontal_diffusivity
#o.set_config('drift:wind_uncertainty',2) #Not necessary when adding horizontal_diffusivity

#### Seeding setup
N = 100 # Number of particles
z = np.random.uniform(-5, -0.1, N) # Particle release depth between -5 and surface - adding depth seems to cause an error with seed_cone, particles are only released at surface. OK for now
#Seed in cone at the mouth of Måselv
lon_m_outer_e = [18.5382232] 
lat_m_outer_e = [69.3079802] 
lon_m_outer_w = [18.5056143] 
lat_m_outer_w = [69.3062551] 

months =  list(range(2, 10)) # Seed all months Feb-Oct
for month in months:
    num_days = calendar.monthrange(startTime.year, month)[1]
    days = [date(startTime.year, month, day) for day in range(1, num_days+1)]
    for day in days:
        day = datetime.combine(day, datetime.min.time())
        o.seed_cone(lon=[lon_m_outer_e, lon_m_outer_w], lat=[lat_m_outer_e, lat_m_outer_w], number=N, radius=20, time=day, origin_marker=month, terminal_velocity=terminalVelocity)

####  Running model
o.run(time_step=3600, end_time=endTime, time_step_output=3600*24, outfile=outfile, export_buffer_length=4)

# Show output
#o.plot(fast=True, linecolor='origin_marker', legend=['Målselv','Aursfjord','Nordfjord'],colorbar=False,filename='drift_plot.png')
#o.plot_property('z')
#o.plot_property('z', mean=True)
#o.animation(fast=True, color='origin_marker', legend=['Målselv','Aursfjord','Nordfjord'],colorbar=False,filename='drift_animation.mp4')
