## packages
import numpy as np
import sdf
import matplotlib.pyplot as plt
import pickle

## python files
import my_constants as const
from list_new import *
from power import *
from energy import *



class Simulation():
	def __init__(self):
		
		self.sim_file_loc = getSimulation('/storage/space2/phrmsf/resolve_e_gyro')
#		self.sim_file_loc = getSimulation('/home/space/phrmsf/Documents/EPOCH/rb_epoch/epoch/epoch1d/D_He3')
		self.quantity = 'Magnetic_Field_Bz'
#		self.quantity = 'Derived_Number_Density_Deuterons'
		# self.quantity = 'Electric_Field_Ex'
		# self.quantity = ''
		self.index_list = list_sdf(self.sim_file_loc)
		print(str(len(self.index_list))+' files')
		
		try: ## load 0th and last file for later use in calculation of constants
			print('Loading first and last file...')
			self.file0 = sdfread(0)
			self.filelast = sdfread(self.index_list[-1])
			print('Done.')
		except:
			print('# ERROR # : Couldn\'t load files.')
			raise SystemExit
		
		try:
			self.times = read_pkl('times')	
		except:
			print('# ERROR # : Couldn\'t load times.')
		
		# grid_length, N_grid_points, duration, N_time_points = batch_getSimRanges((self.index_list,self.file0,self.filelast,self.times))
		self.N_x = len(getGrid(self.file0)[0])

		## Creates list of available fieldmatrix pickled files in the sim directory
		lst_dir = os.listdir(self.sim_file_loc)
		pkl_list = [item for item in lst_dir if '.pkl' in item]
		lst_FT2d = [item for item in pkl_list if 'FT_2d' in item]
		lst_Fm = [item for item in pkl_list if 'fieldmatrix' in item]

		if self.quantity != '':
			if len(pkl_list) > 0: ## check if any previous pkl files
				if len(lst_FT2d) > 0:
					print('You have available FT 2d pkl file(s): {}'.format(lst_FT2d))
				if len(lst_Fm) > 0:
					print('You have available fieldmatrix pkl file(s): {}'.format(lst_Fm))
#				pkl_file = input('Type the name of the * quantity * to select a pkl file [or \'no\' to generate new]...\n!>>') # just need the quantity name
				pkl_file = self.quantity
				## change this is measuring something different 
				if pkl_file not in ['no','n','N','No','','None','NO']: # only enters this loop if want to load file
					if 'FT_2d_'+pkl_file+'.pkl' in pkl_list: # check if pkl file is in list
						FT2d_or_Fm = input('2d FT or load Fm? [choices: 2d, fm]\n!>>')
						if FT2d_or_Fm == '2d':
							try:
								self.FT_2d = read_pkl('FT_2d_'+pkl_file)
								print('shape FT2d (w,k) :: {}'.format(np.shape(self.FT_2d)))
							except:
								print('# ERROR # : Couldn\'t load 2d FFT.')
								raise SystemExit
						elif FT2d_or_Fm =='fm':
							try:
								self.fieldmatrix = read_pkl('fieldmatrix_'+pkl_file)
								print('shape Fldmat (T,X) :: {}'.format(np.shape(self.fieldmatrix)))
							except:
								print('# ERROR # : Couldn\'t load fieldmatrix.')
								raise SystemExit						
					elif 'fieldmatrix_'+pkl_file+'.pkl' in pkl_list:
						self.fieldmatrix = read_pkl('fieldmatrix_'+pkl_file)
						print('shape Fldmat (T,X) :: {}'.format(np.shape(self.fieldmatrix)))
						# self.times = read_pkl('times')
					else:
						print('# ERROR # : Please select the correct file.')
						raise SystemExit
				else: # produce fieldmatrix if don't want to load pkl file
					self.batch_size = int(input('Input Batch Size:\n!>>'))
					if self.batch_size <= 0 or self.batch_size > 1024 or self.batch_size > len(self.index_list):
						print('# ERROR # : Batch size less than 0, greater than 1024 or greater than length of simulation.')
						raise SystemExit
					self.fieldmatrix, self.times = load_tot_fieldmatrix(self.index_list,self.N_x,self.file0,self.quantity,self.batch_size)
					# Pickles and writes these arrays to files, so we can use them next time
					print('shape Fldmat (T,X) :: {}'.format(np.shape(self.fieldmatrix)))
					dumpfiles(self.fieldmatrix, 'fieldmatrix_'+self.quantity)
					dumpfiles(self.times, 'times')
			else: ## If no pkl files then will create a new one based on 'self.quantity'
				self.batch_size = int(input('Input Batch Size:\n!>>'))
				if self.batch_size <= 0 or self.batch_size > 1024 or self.batch_size > len(self.index_list):
					print('# ERROR # : Batch size less than 0, greater than 1024 or greater than length of simulation.')
					raise SystemExit
				self.fieldmatrix, self.times = load_tot_fieldmatrix(self.index_list,self.N_x,self.file0,self.quantity,self.batch_size)
				# Pickles and writes these arrays to files, so we can use them next time
				dumpfiles(self.fieldmatrix, 'fieldmatrix_'+self.quantity)
				dumpfiles(self.times, 'times')
				print('shape Fldmat (T,X) :: {}'.format(np.shape(self.fieldmatrix)))
		else:
			print('No fieldmatrix loaded.')
		
		spec_names = getIonSpecies(self.file0) # get a list of species names from scanning the 0th file's keys for an always returned value "Number_Density_" + species
		print('# Ions :: ',len(spec_names), ' :: ', spec_names)
		maj_species, maj2_species, min_species = spec_names
		Zmaj = getChargeNum(maj_species)
		Zmin = getChargeNum(min_species)

		self.wc_maj  = getCyclotronFreq(self.file0,maj_species,Z=Zmaj)
#		self.wc_min  = getCyclotronFreq(self.file0,min_species,Z=Zmin)
		self.wce  = getCyclotronFreq(self.file0,'Electrons',Z=1)
		self.va   = getAlfvenVel(self.file0)
		self.lambdaD= getDebyeLength(self.file0,'Electrons')
		self.wpe	 = getPlasmaFreq(self.file0,species='Electrons')
		self.wpi  = getPlasmaFreq(self.file0,species=maj_species)
		self.tc_maj = 2*const.PI/self.wc_maj
#		self.tc_min = 2*const.PI/self.wc_min

		self.tnorm = self.tc_maj
		self.wnorm = self.wc_maj
#		print('wc_maj: ', self.wc_maj,'wc_min: ',self.wc_min, 'wce: ', self.wce,'wpe: ', self.wpe,'va: ', self.va,'lambdaD: ', self.lambdaD)
		print('normalisation w: ', self.wnorm, ' [Hz]')
#		print('tc_min',self.tc_min)
		
		#!#!#  Wrote new script to handle this work
	### PLOT ENERGY DENSITIES OF FIELD COMPONENTS ### 
#		energies(sim_loc=self.sim_file_loc,min_species=min_species,maj_species=maj_species,maj2_species=maj2_species,frac=1,plot=True)

	### FOURIER TRANSFORMS ###
		self.klim, self.wlim = batch_getDispersionlimits((self.index_list,self.file0,self.filelast,self.times)) # non-normalised units
		DISP_DATA = self.index_list, self.file0, self.filelast, self.times, self.klim, self.wlim, self.wc_maj, self.wce, self.va, self.lambdaD, self.wpe, self.wpi, self.wnorm
		self.klim_prime, self.wlim_prime, self.tlim_prime = norm_DispersionLimits(DISP_DATA)#,species=maj_species)

#		in_klimprime = float(input('!>> klim_prime maximum on plot (0 for max): ')) ## in normalised units
#		in_wlimprime = float(input('!>> wlim_prime maximum on plot (0 for max): ')) ## in normalised units
		in_klimprime = 200 # use when lazy
		in_wlimprime = 90
		if in_klimprime == 0:
			in_klimprime = self.klim_prime
		if in_wlimprime == 0:
			in_wlimprime = self.wlim_prime
		in_tlimprime = self.tlim_prime # maximum time
		# in_tlimprime = 2.

	### FT 1d ###
		try:
			self.FT_1d = read_pkl('FT_1d_'+self.quantity)
			print('FT_1d shape (t,k):: {}'.format(self.FT_1d.shape))
			k_lim, t_lim = self.FT_1d.shape[1]*(in_klimprime/self.klim_prime), self.FT_1d.shape[0]*(in_tlimprime/self.tlim_prime)
			print('k_file_lim ',k_lim,'t_file_lim ', t_lim)
			self.FT_1d = self.FT_1d[:int(t_lim),:int(k_lim)]
			print('FT_1d plotting shape (t,k):: {}'.format(self.FT_1d.shape))			
		except:
			print('Creating 1d FFT...')
			self.FT_1d = get1dTransform(self.fieldmatrix,window=False)
			dumpfiles(self.FT_1d,'FT_1d_'+self.quantity)
			## Limits the size of the loaded array to our inputted limits so that it normalises properly ##
			k_lim, t_lim = self.FT_1d.shape[1]*(in_klimprime/self.klim_prime), self.FT_1d.shape[0]*(in_tlimprime/self.tlim_prime)
			print('k_file_lim ',k_lim,'t_file_lim ', t_lim)
			self.FT_1d = self.FT_1d[:int(t_lim),:int(k_lim)]
		fig_1, ax_1 = plot1dTransform(self.FT_1d, va_tci=[self.va,self.tnorm==self.tc_maj], klim=in_klimprime, tlim=in_tlimprime, Omega_label=getOmegaLabel(maj_species))
		plotting(fig_1,ax_1,'FT_1d_'+self.quantity)
#		fig_1.savefig('FT_1d_'+self.quantity+'.jpeg',bbox_inches='tight')
#		plt.clf()
#		print('Plotted 1d FT + Saved.')
		del self.FT_1d

#	### Find the minimum field value along a given k' stream (to look at corrugation effect)
#		fig,ax=plt.subplots(figsize=(10,5))			
#		k_range=[3,4,5,6]
#		for k_min in k_range:		
#			print('k_min', k_min)
#			log10_power_one_k, times = power_one_k(self.FT_1d,self.tlim_prime, self.klim_prime,0,self.tlim_prime,k_min)
#			#	log10_power, times = powerspectrum(self.FT_1d, self.tlim_prime, self.klim_prime,0,self.tlim_prime,1,15)
#			power = 10**log10_power_one_k
#			ax.plot(times,power,label='k={}'.format(k_min))
#			#	grad_power = np.gradient(power)
#			#nth = 100
#			#times_nth, grad_nth = times[::nth], grad_power[::nth]
#			# print(times, log10_power)
#			#ax.set_ylabel('Power', fontsize=18)
#			#ax2=ax.twinx()
#			#ax.plot(times_nth, grad_nth,label='grad : nth ={}'.format(nth),linewidth=1,color='r') # only plot every nth data point
#		ax.set_xlabel(r'$t/\tau_{cD}$', fontsize=18)
#		ax.set_ylabel('Power',fontsize=18)
#		#ax.set_yscale('log')
#		ax.legend(loc='best',fontsize=14)
#		fig.savefig('power_one_k.jpeg',bbox_inches='tight')
#		print('Plotted 1d Power')
	
#	### Mean Bz ### 
#		print(self.fieldmatrix)
#		mean_field = getMeanquantity(self.file0,self.quantity)
#		self.fieldmatrix = self.fieldmatrix-mean_field
#		print(mean_field,self.fieldmatrix)
		
	### FT 2d ###
		try:
			self.FT_2d = read_pkl('FT_2d_'+self.quantity)
			w_lim, k_lim = self.FT_2d.shape[0]*(in_wlimprime/self.wlim_prime), self.FT_2d.shape[1]*(in_klimprime/self.klim_prime)
			self.FT_2d = self.FT_2d[:int(w_lim),:int(k_lim)]
		except:
			print('Creating FT_2d...')
			self.FT_2d = get2dTransform(self.fieldmatrix,window=True)
			dumpfiles(self.FT_2d,'FT_2d_'+self.quantity)
			## Limits the size of the loaded array to our inputted limits so that it normalises properly ##
			w_lim, k_lim = self.FT_2d.shape[0]*(in_wlimprime/self.wlim_prime), self.FT_2d.shape[1]*(in_klimprime/self.klim_prime)
			self.FT_2d = self.FT_2d[:int(w_lim),:int(k_lim)]
		print('plotting shape: ',np.shape(self.FT_2d))
		fig, ax = plot2dTransform(self.FT_2d, va_wci=[self.va,self.wnorm==self.wc_maj], klim=in_klimprime, wlim=in_wlimprime, Omega_label=getOmegaLabel(maj_species))

	### COLD PLASMA DISPERSION ###
		omegas = self.wnorm*np.linspace(0,in_wlimprime,10000) # TODO: find out why doesn't go through 0
		if maj2_species == '':
			k1,k2,k3=coldplasmadispersion(self.file0,maj_species,'Electrons',z1=Zmaj,z2=Zmin,omegas=omegas) # two solutions to the cold plasma dispersion
		else:
			k1,k2,k3=coldplasmadispersion_twomaj(file0=self.file0,species1=maj_species,species2=maj2_species,species3='Electrons',z1=Zmaj,z2=getChargeNum(maj2_species),z3=Zmaj, omegas=omegas)
		
		knorm = (self.va/self.wc_maj)
		k1,k2,k3 = k1*knorm, k2*knorm, k3*knorm
		thresh = k2 > 0 # threshold the array so it only plots the FAW and not the horizontal line to the 0 parts of the dispersion
		ax.plot(k2[thresh],omegas[thresh]/self.wnorm,color='k',linestyle='-',alpha=0.75)
#		ax.plot(k1,omegas/self.wnorm,color='k',linestyle='-',alpha=0.75)
#		ax.plot(k3,omegas/self.wnorm,color='k',linestyle='-',alpha=0.75)
		ax = ColdWaveModes(ax,[self.wpi, self.wpe, self.wc_maj, self.wce],self.va,self.wnorm,OMODE=False,FAW=False,UH=False,LH=True,CA=False,CYC=False,LGHT=False) ## turn this on if you want the classical modes plotted
		ax.set_ylim(0,in_wlimprime)
		ax.set_xlim(0,in_klimprime)
	### DOPPLER SHIFT DISPERSION ###
#		v_thD = getThermalVelocity(self.file0,'Deuterons')
#		U = -0.00018384528786131768*v_thD
#		U = U*(10**(4))
#		k2_rl = k2/knorm
#		w_prime = omegas - U*k2_rl
#		print(w_prime/self.wnorm)
#		ax.plot(np.real(k2_rl)*knorm, w_prime/self.wnorm, color='k',linestyle='--',alpha=0.75)
		plotting(fig,ax,'FT_2d_'+self.quantity)

	### Finding the size of the k_diff between CA and FAW dispersions
#		plt.clf()
#		plt.figure()
#		plt.subplot(131)
#		plt.plot(kA,omegas/self.wnorm,linestyle='--',label='Alfven')
#		plt.plot(k2,omegas/self.wnorm,label='FAW')
#		plt.ylim(2,25)
#		plt.xlim(0,150)
#		plt.xlabel(r'$kv_A/\Omega_p$',fontsize=18)
#		plt.ylabel(r'$\omega/\Omega_p$',fontsize=18)
#		
#		plt.subplot(132)
#		plt.plot(abs(k2-kA),omegas/self.wnorm)
#		plt.ylim(2,25)
#		plt.xlim(0,150)
#		plt.xlabel(r'$|k_{FAW}-k_A|$',fontsize=18)

#		plt.subplot(133)
#		kA, k2 = kA/knorm, k2/knorm # SI units
#		kdiff = abs(k2-kA) # SI
#		Ldiff = 2*const.PI/kdiff # wavelengths of difference
#		rlalpha = getLarmorRadius(self.file0, 'Alphas')
#		rlp = getLarmorRadius(self.file0, 'Protons')
#		plt.plot(Ldiff/rlp, omegas/self.wnorm,color='orange',label=r'$r_{Lp}$')
#		plt.plot(Ldiff/rlalpha, omegas/self.wnorm,color='r',label=r'$r_{L\alpha}$')
#		plt.xlabel(r'$L_{|k_{diff}|}/r_{L\sigma}$',fontsize=18)
#		plt.ylim(2,25)
#		plt.xlim(0,70)
#		plt.legend(loc='best')
#		plt.show()

#	### MCI GROWTH RATES ###
		# use k2 values as kall in growth_rates
#		v0 = 12982938.44
#		ud = 6491467.721
#		vd = 11243551.91
#		vr = 13660.64338
#		ur = 1873606.4338
#		theta = 90.
#		posomegas, posgammas = get_MCIgrowth(b=sdfread(100),minions=min_species,majions=maj_species,v0=v0,ud=ud,ur=ur,vd=vd,vr=vr,omegas=omegas,kall=k2,theta=90.)		
##		posomegas,posgammas = growth_rate_man(minions=minions, majions=majions, theta=theta, b=sdfread(100), u=v0, vd=vd, vr=vr, kall=k2, omegaall=omegas)
#		fig,ax=plt.subplots(figsize=(6,6))
#		ax.plot(posomegas,posgammas)	
#		ax.set_xlabel(r'$\omega/\Omega_{c\alpha}$',fontsize=18)	
#		ax.set_ylabel(r'$\gamma/\Omega_{c\alpha}$',fontsize=18)
#		ax.set_yscale('log')
#		fig.savefig('MCI_gammas.jpeg',bbox_inches='tight')
#		plt.clf()

	### POWER SPECTRUM ###
		power(klim_prime=self.klim_prime,wlim_prime=self.wlim_prime,wmax=30,kmax=in_klimprime,norm_omega=getOmegaLabel(maj_species),quant=self.quantity,plot=True)
		 
#	### BICOHERENCE SPECTRA ###
		# TODO; Figure out how to get this to work
#		bicoherence_theory_taverage(field=self.fieldmatrix, klim=self.klim_prime, ftspacing=175, ftwidth=200, smooth=False)
#		fs = 1/(self.times[0]-self.times[1])
#		N = self.times.shape[0]
#		kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))		
#		freq1, freq2, bicoh = polycoherence(self.FT_2d, fs, **kw)
#		plot_polycoherence(freq1, freq2, bicoh)



if __name__ == '__main__':
	Simulation()
