import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import healpy as hp
import h5py
from scipy.interpolate import RegularGridInterpolator
from astropy.time import Time, TimeDelta
from garam import GalaxyElimination

file = "DIPOLE_500times_SMOOTH.h5"

SITE_LATITUDE       = 32.81444444 #/* +32.77944444   deg for Hanle */ 
SITE_LONGITUDE      = 78.85555556 #/* 78.96416667 deg for Hanle */
ELEVATION = 4500

dt = TimeDelta(np.linspace(0.,24.*3600, 500), format='sec') #Will NOT take ~5 min
obstimes = Time('2022-10-15 00:00:00') + dt

chosen_frequency = 30
nside = 16
lmax = 45
bmax = 10

param_obs = {'file': file,\
            'time': obstimes,\
            'chosen_frequency': chosen_frequency,\
            'site_latitude': SITE_LATITUDE,\
            'site_longitude': SITE_LONGITUDE,\
            'elevation': ELEVATION,\
            'nside': nside,\
            'l':lmax,\
            'b':bmax}

GE = GalaxyElimination(**param_obs)
GE.time_elimination()