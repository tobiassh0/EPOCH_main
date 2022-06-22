import numpy as np
from list_new import *
import my_constants as const
import os, sys
from scipy.signal import find_peaks

# quant = 'Magnetic_Field_Bz'
# # sim_files = ['/storage/space2/phrmsf/0006qp','/storage/space2/phrmsf/7_3rb']
# # sim_files = ['/storage/space2/phrmsf/7_3rb']
# # sim_files = ['0006qp','7_3rb']
# sim_files = ['/storage/space2/phrmsf/7_3rb','/storage/space2/phrmsf/He3_rb2']
# #names = ['Background', 'w/ MCI']
# #names = ['w/ MCI']
# #names = ['']

# height = 5
# width = height*const.g_ratio
# fig,ax=plt.subplots(figsize=(width,height))

# #if in_klimprime == 0:
# #	in_klimprime = klim_prime
# #if in_wlimprime == 0:
# #	in_wlimprime = wlim_prime

# maj_species = 'Protons'
# min_species = 'Helium3'

# Zmaj = getChargeNum(maj_species)
# Zmin = getChargeNum(min_species)

# t_max = 9#13.473739145030912
# k_max = 10136.787287877456
# in_tlimprime = t_max
# in_klimprime = 20


# ylims=[2,10]
	
# for s in range(len(sim_files)):
# 	fig,ax=plt.subplots(figsize=(10,5))	
# 	fig2,ax2=plt.subplots(figsize=(10,5))

# 	index_list = list_sdf(sim_files[s])
# 	os.chdir(sim_files[s])
	
# 	try:
# 		FT_1d = read_pkl('FT_1d_'+quant)
# 	except:
# 		print('## ERROR ## :: Cannot load FT_1d')
# 		raise SystemExit

# 	k_range=[2,3,4,5]
# 	k_range = np.linspace(1,200,5000)
# 	#k_range=[3]
# 	counts = []
# 	for k_min in k_range:
# 	#	print('k_min', k_min)
# 		log10_power_one_k, times = power_one_k(FT_1d,t_max,k_max,0,in_tlimprime,k_min)
# 	#	log10_power,_ = powerspectrum(FT_1d,t_max,k_max,0,in_tlimprime,0,k_max)
# 		#	log10_power, times = powerspectrum(self.FT_1d, self.tlim_prime, self.klim_prime,0,self.tlim_prime,1,15)
# 		power = 10**log10_power_one_k
# 	#	ax.plot(times,power,label='k={}'.format(k_min))
# 	#	PT_loc = find_peaks(power,height=None,threshold=None,prominence=0.15)[0]
# 	#	print(len(PT_loc)/int(t_max))
# 	#	ax.scatter(times[PT_loc],power[PT_loc])
# 		counts.append(len(find_peaks(power,height=None,threshold=None,prominence=0.15)[0])/int(t_max))

# 	counts = np.array(counts)
# 	ax2.plot(k_range,(1/counts))

# 	#ax.set_xlabel(r'$t/\tau_{cHe3}$', fontsize=18)
# 	#ax.set_ylabel('Power',fontsize=18)
# 	ax2.set_xlabel(r'$k^\prime$', fontsize=18)
# 	ax2.set_ylabel(r'$\delta t$'+'  '+'['+r'$\tau_{cD}$'+']',fontsize=18)

# 	##ax.set_yscale('log')
# 	#ax.legend(loc='best',fontsize=14)
# 	#fig.savefig('power_one_k.png',bbox_inches='tight')
# 	ax2.set_ylim(0,ylims[s])
# 	fig2.savefig('PT_count.png',bbox_inches='tight')
# 	print('Plotted 1d Power')
# 	plt.clf()
# 	os.chdir('..')


def power(klim_prime,wlim_prime,wmax,kmax,norm_omega=r'$\Omega_D$',quant='',plot=False):

	try:
		log10_power = read_pkl('log10_power')
		omegas = read_pkl('omegas_power')
	except:
		try:
			FT_2d = read_pkl('FT_2d_'+quant)
		except:
			print('## ERROR ## :: Cannot load FT_2d')
			raise SystemExit
		print('Calculating power...')
		log10_power,omegas=powerspectrum(FT_2d,wlim_prime,klim_prime,0,wmax,0,kmax)
		dumpfiles(log10_power,'log10_power')
		dumpfiles(omegas,'omegas_power')
	
	if plot:
		print('Plotting Power...')
		width,height = 8,5
		fig,ax=plt.subplots(figsize=(width,height))

		ax.plot(omegas,10**log10_power)
		for i in range(1,int(wmax)+1):
			ax.axvline(i,color='k',alpha=0.5,linestyle=':')
		ax.set_xlabel(r'$\omega$'+'/'+norm_omega,fontsize=18)
		ax.set_xlim(0,wmax)
		ax.set_ylabel('Power',fontsize=18)
		ax.set_ylim(1E4,1E11)
#		print(10**log10_power)
		ax.set_yscale('log')
		plotting(fig,ax,'power')
#		fig.savefig('power.jpeg',bbox_inches='tight')
#		plt.clf()
#		print('Done.')

	del omegas, log10_power

	return None
