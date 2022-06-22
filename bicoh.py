from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from list_new import *
from matplotlib.path import Path
from bispectral_analysis import *

#from polycoherence import _plot_signal, polycoherence, plot_polycoherence

##t = np.linspace(0, 100, N)
#dx = 0.003
#dt = 0.001
#x = np.arange(0,3,dx)
#t = np.arange(0,1,dt)
#X,T = np.meshgrid(x,t)
#fs = 1 / (t[1] - t[0]) # sampling frequency
#N = len(x)

#f1 = 20
#f2 = 5
#f3 = f1 + f2
#A1 = 1 ; A2 = 1
#kk1 = 5 ; kk2 = 0
#k1 = 2*np.pi*kk1 ; k2 = 2*np.pi*kk2
#w1 = 2*np.pi*f1 ; w2 = 2*np.pi*f2 

#s1 = A1*np.sin(w1*t)#-k1*X)
#s2 = A2*np.sin(w2*t)#-k2*X)
#signal = s1 + s2
#print(s1.shape,s2.shape)

##s1 = np.cos(2 * pi * 5 * t + 0.2)
##s2 = 3 * np.cos(2 * pi * 7 * t + 0.5)
##np.random.seed(0)
##noise = 5 * np.random.normal(0, 1, N)
### signal is combined indepedent signals (just use fieldmatrix here)
##signal = s1 + s2 + 0.5 * s1 * s2 + noise
##plt.imshow(signal,cmap='gray')
##plt.show()
##_plot_signal(t, signal)

## Plot total bicoherence spectrum
#kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))
##freq1, freq2, bicoh = polycoherence(signal, fs, **kw)
##plot_polycoherence(freq1, freq2, bicoh)

### Plot bispectrum
##freq1, fre2, bispec = polycoherence(signal, fs, norm=None, **kw)
##plot_polycoherence(freq1, fre2, bispec)

## Plot part of the bicoherence spectrum
#freq1, freq2, bicoh = polycoherence(signal, fs, flim1=(0, 250), flim2=(0, 250), **kw)
#plot_polycoherence(freq1, freq2, bicoh)


sim_loc = getSimulation('/storage/space2/phrmsf/JET_26148')
ind_lst = list_sdf(sim_loc)
signal = read_pkl('fieldmatrix_Magnetic_Field_Bz')
times = read_pkl('times')
fft = get2dTransform(signal, window='No')
#plt.figure()
#plt.imshow(np.log10(abs(fft[1:,1:])))#,extent=[-knyq,knyq,-wnyq,wnyq])
#plt.colorbar()
#plt.show()
duration = times[-1]
dt = (times[-1]-times[0])/len(times)
length = getGridlen(sdfread(0))
dx = getdxyz(sdfread(0))
nT=signal.shape[0]
nx=signal.shape[1]

file0 = sdfread(0)
filelast=sdfread(ind_lst[-1])
SIM_DATA = ind_lst, file0, filelast, times	
knyq, wnyq = batch_getDispersionlimits(SIM_DATA)

verts = [
   (0., 0.),  # left, bottom
   (0.,knyq),  # left, top
   (wnyq, knyq),  # right, top
   (wnyq, 0.),  # right, bottom
   (0., 0.),  # ignored
]

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]

area = Path(verts, codes)
nfft=nT//5
bicoh=bispectrum2D(signal, dt, length, duration, area, nfft, noverlap=2*nfft//3, norm=[1,1], window=False, bispectrum=False)
#with open('bicoh_mat.pkl', 'wb') as f:
#	pickle.dump([bicoh,dt, length, area, nfft], f, protocol=4)
#extent1 = [0, wnyq, 0, wnyq]
plot_bicoh(bicoh, smooth=True, cbar=True, clim=(None, None))
#from scipy.ndimage.filters import gaussian_filter
#bicoh = gaussian_filter(bicoh, sigma=1)
#plt.imshow(bicoh)
plt.show()












