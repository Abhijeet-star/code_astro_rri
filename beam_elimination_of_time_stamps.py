import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import healpy as hp
import matplotlib
import healpy as hp
from pygdsm import GlobalSkyModel 
from scipy.interpolate import RegularGridInterpolator
import h5py
import lunarsky as lss
import os
from tqdm.notebook import tqdm
import h5py
from scipy.optimize import basinhopping
cwd = os.getcwd()
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = "18"
gsm = GlobalSkyModel()

np.random.seed(0)

#########################################################################################
nside = 16 
npix = hp.nside2npix(nside)
pixels = np.arange(npix)
theta, phi = hp.pix2ang(nside, pixels, nest=True)

ll_coordinate = phi
bb_coordinate = np.pi/2. - theta

#########################################################################################
SITE_LATITUDE   = 14.93497
SITE_LONGITUDE  = 170.05050
ELEVATION       = 0.0


################# READ .h5 FILE ###############################################################
ff = h5py.File('dipole_smooth_5_50MHz_jul30.h5','r')
sig = np.loadtxt('signal_21cm_5_50MHz.txt')

LST = ff['index_map']['LST'][()]
freq = ff['index_map']['frequency'][()]
ind_sorted = np.argsort(LST)
lst = LST[ind_sorted]

beam_3D = ff['ancillary_prod']['beam'][()]   # Reading the normalized beam

T = ff['T_A'][()][ind_sorted, :]
chosen_freq = 5                 # Frequency for which time-stamp elimination to be performed
chosen_freq_idx = np.where(freq==chosen_freq)[0][0]


B = 1000000                     # 1MHz frequency resolution
tau = 31556926 * 10**3          # max avg temp ~ 5 x 10^5
const = 1/np.sqrt(B*tau)        # T_rms getting down to around 2.8 mK

for i in range(len(freq)):
    freq_range = freq
################# ADDING SIGNAL ###############################################################
    T2 = np.zeros_like(T)
    for kk in range(len(LST)):
        T2[kk] = T[kk] + sig[:,1]

    avg = []
    nn = []

    for ii in range(len(freq_range)):
        a = sum(T2[:,ii]) / len(LST)
        sigma = a * const
        gen_noise = np.random.normal(0,sigma,1)
        a1 = a + gen_noise[0]
        nn.append(gen_noise[0])
        avg.append(a1)
    av_temp = np.array(avg)
    sigma_i = av_temp * const

################# NUMBER OF TIME-STAMPS #######################################################
dt = TimeDelta(np.linspace(0.,655.2*3600, len(lst)), format='sec')
obstimes1 = Time('2019-4-12 23:00:00') + dt
obstimes = obstimes1[ind_sorted]          # Sorting the time-stamps
phi_res   = 1.0
theta_res = 1.0
theta_array = np.arange(-90, 90 + theta_res, theta_res)
theta_array = np.round(theta_array, 2) 
phi_array = np.arange(0, 360 + phi_res, phi_res)

tsky   = np.zeros((npix, len(freq)))

location = lss.MoonLocation(lat=SITE_LATITUDE*u.deg,lon=SITE_LONGITUDE*u.deg,height=ELEVATION*u.m) 
gc       = SkyCoord(l=ll_coordinate*u.radian, b=bb_coordinate*u.radian,frame='galactic')  


################# CREATING BEAM ARRAY #########################################################
beam_val = RegularGridInterpolator((theta_array, phi_array), beam_3D[chosen_freq_idx])


################# FUNCTIONS USED ##############################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def weighted_std(x, wts, p):
    wts = np.array(wts)
    x = np.array(x)
    x_bar = np.sum(wts * x) / np.sum(wts)
    N = len(x)
    ss = np.sum(wts * (x - x_bar)**2) / (np.sum(wts) * (N-p)/N)
    var_wt = ss 
    return  np.sqrt(var_wt)

# def find_idx(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx


def inside_rectangle_np(points, rect_bottom_left, rect_top_right, include_boundary=True):
    x1, y1 = rect_bottom_left
    x2, y2 = rect_top_right
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    px, py = points[:, 0], points[:, 1]
    if include_boundary:
        mask = (xmin <= px) & (px <= xmax) & (ymin <= py) & (py <= ymax)
    else:
        mask = (xmin < px) & (px < xmax) & (ymin < py) & (py < ymax)
    return mask


def fixed_radius(tt):
    mn = lss.LunarTopo(obstime=obstimes[tt], location=location)
    trans_local = gc.transform_to(mn)
    az, alt     = trans_local.az.degree, trans_local.alt.degree
    beam_gen = np.zeros_like(tsky)

    rogue_phi  = []

    for iangle, (alt_value, az_value) in enumerate(zip(alt,az)):
        if az_value > phi_array.max():
            rogue_phi.append(az_value)
            az_value = 360 - az_value
        beam_gen[iangle,0] = beam_val([alt_value, az_value])

    hf_bm = find_nearest(beam_gen[:,0], 0.5)
    idx_hf = np.where(beam_gen[:,0] == hf_bm)

    idx = np.where(beam_gen[:,0] == max(beam_gen[:,0]))
    theta_c, phi_c = hp.pix2ang(16, idx[0][0], nest=True)

    theta_hf, phi_hf = hp.pix2ang(16, idx_hf[0][0], nest=True)

    vec_c  = hp.ang2vec(theta_c, phi_c)
    vec_hf = hp.ang2vec(theta_hf, phi_hf)
    radius = np.arccos(np.clip(np.dot(vec_c, vec_hf), -1.0, 1.0))
    return radius


################# TIME-STAMP ELIMINATION #######################################################
masked_timestamps = []
good_timestamps   = []
radius_all = []
for ii in range(len(obstimes)):
    mn = lss.LunarTopo(obstime=obstimes[ii], location=location)
    trans_local = gc.transform_to(mn)
    az, alt     = trans_local.az.degree, trans_local.alt.degree
    ind_below_horizon       = alt < 0
    beam_gen = np.zeros_like(tsky)

    rogue_phi  = []

    for iangle, (alt_value, az_value) in enumerate(zip(alt,az)):
        if az_value > phi_array.max():
            rogue_phi.append(az_value)
            az_value = 360 - az_value
        beam_gen[iangle,0] = beam_val([alt_value, az_value])

    hf_bm = find_nearest(beam_gen[:,0], 0.5)
    idx_hf = np.where(beam_gen[:,0] == hf_bm)

    idx = np.where(beam_gen[:,0] == max(beam_gen[:,0]))
    theta_c, phi_c = hp.pix2ang(16, idx[0][0], nest=True)

    theta_hf, phi_hf = hp.pix2ang(16, idx_hf[0][0], nest=True)

    # vec_c  = hp.ang2vec(theta_c, phi_c)
    # vec_hf = hp.ang2vec(theta_hf, phi_hf)
    # radius = np.arccos(np.clip(np.dot(vec_c, vec_hf), -1.0, 1.0))
    radius = fixed_radius(int(len(LST)/2))

    npts = 1000
    phi = np.linspace(0, 2*np.pi, npts)

    theta_ring = np.arccos(np.cos(radius) * np.cos(theta_c) +
                        np.sin(radius) * np.sin(theta_c) * np.cos(phi))
    phi_ring = phi_c + np.arctan2(np.sin(phi) * np.sin(radius) * np.sin(theta_c),
                                np.cos(radius) - np.cos(theta_ring) * np.cos(theta_c))

    lon = np.degrees(phi_ring) 
    lat = 90 - np.degrees(theta_ring)

    l = 45
    b = 10

    rbl = (-l,-b)
    rtr = (+l,+b)

    lb = np.stack((lon,lat), axis=1)


    if inside_rectangle_np((lb), rbl, rtr).any() == True:
        masked_timestamps.append(obstimes[ii])
    else:
        good_timestamps.append(obstimes[ii])
    
    msk_tstps = np.array(masked_timestamps)      # Has Galaxy Coverage
    gd_tstps  = np.array(good_timestamps)        # Galaxy is eliminated
    radius_all.append(radius)

rad_arr = np.array(radius_all)

################# GOOD TIMES ################################################################
lst_good = []
for i in range(len(gd_tstps)):
    observing_time = lss.Time(gd_tstps[i], scale='utc', location=location)
    lst1           = observing_time.sidereal_time('mean').value
    lst_good.append(lst1)
LST_good = np.array(lst_good)/15

good_idx = []
for i in range(len(LST_good)):
    a = np.where(np.round(lst,8) == np.round(LST_good[i],8))[0][0]
    good_idx.append(a)
good_idx_arr = np.array(good_idx)


################# BAD TIMES #################################################################
lst_bad = []
for i in range(len(msk_tstps)):
    observing_time = lss.Time(msk_tstps[i], scale='utc', location=location)
    lst1           = observing_time.sidereal_time('mean').value
    lst_bad.append(lst1)
LST_bad = np.array(lst_bad)/15

bad_idx = []
for i in range(len(LST_bad)):
    a = np.where(np.round(lst,8) == np.round(LST_bad[i],8))[0][0]
    bad_idx.append(a)
bad_idx_arr = np.array(bad_idx)


################# BASIN-HOPPING #############################################################

def rescale(arr, min1=-1, max1=1, log=True):  # scales x axis to -1 to +1
    arr = np.asfarray(arr)                    # np.asfarray -- > returns to float
    if log == True:
        arr = np.log10(arr)
    min_arr = np.amin(arr)
    max_arr = np.amax(arr)
    arr_sc = ((max1 - min1)) * ((arr) - float(min_arr)) / (max_arr - min_arr) + min1
    return arr_sc


########## CASE - I : ORIGINAL FILE, NO ELIMINATION #########################################

# # CREATE MESH-GRID OF LST-FREQ 
# T_log = np.log10(T)                        # T_log is in log10 units
# plt.figure(figsize=(10,6))
# lst = np.degrees(lst) 
# T_log[:,][bad_idx_arr] = np.nan
 

# freq_mesh, lst_mesh = np.meshgrid(freq, lst)
# plt.imshow(T_log, extent=(freq[0], freq[-1], lst[0], lst[-1]), aspect='auto', origin='lower',
#            interpolation='none', cmap='viridis', vmin=np.nanmin(T_log), vmax=np.nanmax(T_log))

# plt.title("DIPOLE BEAM - 100 Timestamps (33 eliminated)")
# plt.xlabel('Frequency (MHz)', fontsize=15)
# plt.ylabel('LST (hours)', fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=12)

# np.savetxt("bad_tsps_13650_mar10.txt", bad_idx_arr)