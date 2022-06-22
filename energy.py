import numpy as np
from list_new import *
import my_constants as const
import os, sys

# quant = ['Magnetic_Field_Bz','Magnetic_Field_By','Electrical_Field_Ex']
# sim_files = ['/storage/space2/phrmsf/He3_rb2']
# sim_files = ['/storage/space2/phrmsf/7_3rb']


# height = 5
# width = 8 
# #height = width*(1/const.g_ratio)
# fig,ax=plt.subplots(figsize=(width,height))

# #if in_klimprime == 0:
# #	in_klimprime = klim_prime
# #if in_wlimprime == 0:
# #	in_wlimprime = wlim_prime

# maj_species = 'Deuterons'
# min_species = 'Alphas'

# Zmaj = getChargeNum(maj_species)
# Zmin = getChargeNum(min_species)

# electric_field_dens = e0*0.5 
# magnetic_field_dens = 1.0/(2.0*mu0)
# # const.mult_elec_field = 0.5*e0
# # const.mult_magn_field = 1.0/(2.0*mu0)
# energy_quant=['energy_times','Exenergy','Byenergy','Bzenergy','majKE','minKE']


# index_list = list_sdf(sim_files[0])
# os.chdir(sim_files[0])

# calc = input('Calculate or plot?\n!>>')
# if calc!='':
# 	# Calculate
# 	print('Calculating...')
# 	energy_data = plotenergies(specmin=min_species,zmin=Zmin,specmaj=maj_species,zmaj=Zmaj,ndata=len(index_list),frac=1)
# 	for i in range(1,len(energy_data)):
# 		dumpfiles(energy_data[i],energy_quant[i])
# else:
# 	# Plot
# 	print('Plotting...')
# 	mult_E=const.mult_elec_field
# 	mult_B=const.mult_magn_field
# 	Energy_mult=[0,mult_E,mult_B,mult_B,1,1]
# 	names=['',r'$\langle E_x^2\rangle\epsilon_0/2$',r'$\langle B_y^2\rangle/2\mu_0$',r'$\langle\Delta B_z^2\rangle/2\mu_0$',r'$D_2$',r'$\alpha$']
# 	colors=['','b','cyan','g','r','m']

# 	time_mult=getCyclotronFreq(sdfread(100),min_species,Zmin)/(2*const.PI)
# 	times=read_pkl('times')
# 	print('multipliers',mult_E,mult_B,time_mult)
# 	print('len times', len(times))
# 	mean_to = 10
# 	for i in range(1,len(energy_quant)):
# 		Energy=read_pkl(energy_quant[i])
# 		print('len energy',len(Energy))
# 		mean_Energy=np.mean(Energy[:mean_to])
# 		energy_plot = (Energy-mean_Energy)*Energy_mult[i]
# 		time_plot = times*time_mult
# 		frac = 1
# 		ax.plot(time_plot[::frac],energy_plot[::frac],label=names[i],color=colors[i])

# #		if energy_quant[i] in ['Exenergy','Byenergy','Bzenergy']:
# #			first = np.where(times*time_mult>0.3)[0][0]
# #			last = np.where(times*time_mult<0.6)[-1][-1]
# #			Energy_dummy = Energy[first:last]
# #			times_dummy = times[first:last]
# #			coeffs = np.polyfit(times_dummy, np.log(Energy_dummy-mean_Energy), 1, w=np.sqrt(Energy_dummy-mean_Energy))
# #			Energy_fit = np.exp(coeffs[1])*np.exp(coeffs[0]*times)
# #			ax.plot(times*time_mult,Energy_fit*Energy_mult[i],linestyle='--',color='r')
# #			print('growth rate {}:: {}'.format(energy_quant[i],str(np.around(coeffs[0],2))))
# #    >>> numpy.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))
# #    array([ 0.06009446,  1.41648096])
# #    #    y ≈ exp(1.42) * exp(0.0601 * x) = 4.12 * exp(0.0601 * x)


# 	ax.set_xlabel(r'$t\Omega_{c\alpha}/2\pi$',fontsize=18)
# 	ax.set_ylabel(r'$\Delta u$'+'  '+'['+r'$Jm^{-3}$'+']',fontsize=18)
# 	ax.legend(loc='best',fontsize=14)
# #	ax.set_ylim(-1200,1000)
# #	ax.set_ylim(1E-3,1E5)
# 	ax.set_xlim(0,5)
# #	ax.set_yscale('log')
# 	fig.savefig('energy_densities.png',bbox_inches='tight')
# 	print('Plot saved.')

def energies(sim_loc,min_species='',maj_species='',maj2_species='',frac=1,plot=False):
	width = 10
	height = 6
	#height = width*(1/const.g_ratio)
	fig,ax=plt.subplots(figsize=(width,height))
	
	quant=['Magnetic_Field_Bz','Magnetic_Field_By','Electrical_Field_Ex']

	Zmaj = getChargeNum(maj_species)
	Zmin = getChargeNum(min_species)
	
	if maj2_species == '': # checks to see if there are two majority species in the sample (e.g. D-T instead of just D)
		energy_quant=['Exenergy','Byenergy','Bzenergy',maj_species+'_KE',min_species+'_KE','Electrons_KE']
		names=[r'$\langle E_x^2\rangle\epsilon_0/2$',r'$\langle B_y^2\rangle/2\mu_0$',r'$\langle\Delta B_z^2\rangle/2\mu_0$',getIonlabel(maj_species),getIonlabel(min_species),getIonlabel('Electrons')]
	else:
		energy_quant=['Exenergy','Byenergy','Bzenergy',maj_species+'_KE',maj2_species+'_KE',min_species+'_KE','Electrons_KE']
		names=[r'$\langle E_x^2\rangle\epsilon_0/2$',r'$\langle B_y^2\rangle/2\mu_0$',r'$\langle\Delta B_z^2\rangle/2\mu_0$',getIonlabel(maj_species),getIonlabel(maj2_species),getIonlabel(min_species),getIonlabel('Electrons')]


	index_list = list_sdf(sim_loc)
	os.chdir(sim_loc)

	# Calculate
	for i in range(len(energy_quant)):
		if energy_quant[i]+'.pkl' not in os.listdir(os.getcwd()):
			print(energy_quant[i]+' .pkl file not present.')
			energy_data = getEnergies(specmin=min_species,zmin=Zmin,specmaj=maj_species,zmaj=Zmaj,specmaj2=maj2_species,ndata=len(index_list),frac=frac)
			for i in range(0,len(energy_data)):
				dumpfiles(energy_data[i],energy_quant[i])
		else:
			print(energy_quant[i]+' .pkl file present')

	if plot:
		print('Plotting energies...')
		mult_E=const.mult_elec_field
		mult_B=const.mult_magn_field
		Energy_mult=[mult_E,mult_B,mult_B,1,1,1,1] # no multiplier for KE of maj, min and elec species
		colors=['b','cyan','g','r','m','orange','k'] # will only use all of them if there are 3 +ve and 1 -ve species
		time_mult=getCyclotronFreq(sdfread(100),min_species,Zmin)/(2*const.PI)
		try:
			times=read_pkl('times')
		except:
			times = batch_getTimes(np.zeros(len(read_pkl('Exenergy'))),1,len(read_pkl('Exenergy'))-1) # janky approach, only used for testing and when times.pkl hasn't been made
		print('multipliers',mult_E,mult_B,time_mult)
		print('len times', len(times))
		mean_to = 10
		for i in range(0,len(energy_quant)):
			Energy=read_pkl(energy_quant[i])
			print('len energy',len(Energy))
			mean_Energy=np.mean(Energy[:mean_to])
			energy_plot = (Energy-mean_Energy)*Energy_mult[i]
			time_plot = times*time_mult
			frac = frac
			ax.plot(time_plot[::frac],energy_plot[::frac],label=names[i],color=colors[i])

		ax.set_xlabel(r'$t\Omega_{\alpha}/2\pi$',fontsize=18)
		ax.set_ylabel(r'$\Delta u$'+'  '+'['+r'$Jm^{-3}$'+']',fontsize=18)
		ax.legend(loc='best',fontsize=14)
		#ax.set_xlim(0,max(times/time_mult))
		plotting(fig,ax,'energy_densities')
#		fig.savefig('energy_densities.jpeg',bbox_inches='tight')
#		print('Plot saved.')
#		plt.clf()
	
		del times, time_plot, Energy, energy_plot
	
	return None
	



