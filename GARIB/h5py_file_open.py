import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker

f = h5py.File('SINESQ_500times_EARTH_jun26.h5','r')

LST = f['index_map']['LST'][()]
freq = f['index_map']['frequency'][()]
ind_sorted = np.argsort(LST)
lst = LST[ind_sorted]

T = f['T_A'][()][ind_sorted,:]
T = np.log10(T)

gd_times = np.loadtxt('badtimes.txt')
index = []
for i in range(len(gd_times)):
    ind = np.where(gd_times[i] == lst)[0][0]
    index.append(ind)

ind_arr = np.array(index)

T[ind_arr,:] = np.nan

#### Create meshgrid of frequencies and LST value
plt.figure(figsize=(10,6))
freq_mesh1, lst_mesh1 = np.meshgrid(freq, lst)

plt.imshow(T, extent=(freq[0], freq[-1], lst[0], lst[-1]), aspect='auto', origin='lower',
           interpolation='none', cmap='viridis', vmin=np.nanmin(T), vmax=np.nanmax(T))

plt.xlabel('Frequency (MHz)', fontsize=15)
plt.ylabel('LST (hours)', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)