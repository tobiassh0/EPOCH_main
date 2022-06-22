import numpy as np
import os, sys
import sdf
# import seaborn as sns
from itertools import cycle
import scipy.fftpack
from scipy import special as spec
from scipy import stats
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import my_constants as const
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D 

plt.style.use('classic')
plt.tight_layout()

# ===============  CONSTANTS  =============== # 
#PI=np.pi
#e0=8.85E-12 			# F m^-1
#mu0=1.25666E-6			# N A^-2
#me=9.10938356E-31 		# kg // make this more precise
#mp=1.6726E-27 			# kg // proper mass ratio 
#qe=1.60217662E-19		# C
#kb=1.3806E-23			# J K^-1
#c=3E8					# m s^-1
#me_to_mp = 1836.2		# Ratio
#me_to_alpha = 7294.3	# ""
#me_to_He3 = 5497.885	# ""
#me_to_D2 = 3671.5		# ""
## home_path = '/mnt/c/Users/twsh2/Documents/PHD/newnew/' #chnage this to wherever you want to save your plots etc.
#home_path = ''
# =========================================== #


# Returns the file name/loc of the simulation being analysed
def getSimulation(loc=''):
	cwd = os.getcwd()
	if loc == '':
		sim_file = input('Input Simulation Directory:\n!>>')
	else:
		sim_file = loc

	try:
		os.chdir(sim_file)
	except:
		print('# ERROR # : Simulation file does not exist.')
		raise SystemExit
	# if os.path.isdir(sim_file) == False:
	return os.path.join(cwd,sim_file)

#def getSpecies(specmin='',specmaj=''): # redundant after 
#	if specmin == '' and specmaj =='':
#		maj_species = input('Input majority +ve species name: (Protons,Deuterons,...)\n!>>')
#		min_species = input('Input minority +ve species name: (Protons,Alphas,...)\n!>>')
#	elif specmin == '' and specmaj != '':	
#		print('maj species : {}'.format(specmaj))
#		min_species = input('Input minority +ve species name: (Protons,Alphas,...)\n!>>')
#	elif specmaj == '' and specmin != '':	
#		print('min species : {}'.format(specmin))
#		maj_species = input('Input minority +ve species name: (Protons,Deuterons,...)\n!>>')		
#	else:
#		print('min species : {}  , maj species : {}'.format(specmin,specmaj))
#		min_species, maj_species = specmin, specmaj

#	return min_species, maj_species

def getIonSpecies(file0):
	keys = file0.__dict__.keys() # returns all the names of quantities in the simulation sdf file
	# check something which is always dumped like derived charge dens
	names = []
	for k in keys:
		if 'Derived_Number_Density_' in k and 'Electrons' not in k:
			names.append(k)
		else:
			continue
	names = [s.replace('Derived_Number_Density_','') for s in names]
	if len(names) <= 2:
		maj2spec = ''
		minspec = ''
		if len(names) == 1:
			return names[0], maj2spec, minspec # case when there is only one ion species
		elif len(names) == 2:
			return names[0], maj2spec, names[1] # case when there are two ions total (maj & min)
		else:
			print('# ERROR # : Can\'t count number of species in sim.')
			raise SystemExit
	elif len(names) == 3:
		return names[0], names[1], names[2] # maj1, maj2, min # i.e. D, T, alphas
	else:
		print('# ERROR # : More than three ion species in sim.')
		raise SystemExit

# Scans and returns a list of files as objs, readable in the form "files[0], files[1] , ..." . NEEDS TO BE IN THE SIMULATION DIRECTORY
def filelist(lim):
	#Scans cwd for all sdf files. Reads and loads these sdf obj into list called "files"
	#Accessing "files[i]" gives the 'i th' sdf object and hence the data for the 'i th' time step.
	files=[] 
	limall = 9999 	 #Assumes no more than 9999 files (TODO: If need more then change %04d part)
	if lim==-1:
		lim=limall
	for i in range(lim):
		try:
			next=sdf.read(('%04d'%i)+'.sdf')  #Assumes no less than 1000 files, otherwise change to 3
			files.append(next)
		except: 
			print('Failed to load: {}.sdf'.format('%04d'%i))
			files.append(None)
	for i in range(limall-lim, limall):  #Work backwards to remove "None" objs in "files"
		if files[limall-1-i]==None:
			files.pop(limall-1-i)
		else:
			break
	# files = [val for val in files if val != None]  #Removes all "None"s from files	 #TODO see if this can be made to work
	return files
	
def batch_filelist(start,stop,len04d):
	# Scans cwd for all sdf files. Reads and loads these sdf obj into list called "files"
	# Accessing "files[i]" gives the 'i th' sdf object and hence the data for the 'i th' time step.
	# Assumes there are more than 9999 files to read
	if len04d: ## TODO; This is an ugly solution, want to change this
		file_list=[]
		if start != stop:
			for i in range(start,stop+1):
				try:
					next=sdf.read(('%04d'%i)+'.sdf')
					file_list.append(next)
				except: 
					print('Failed to load: {}.sdf'.format('%04d'%i))
					file_list.append(None)
		else:
			next=sdf.read(('%04d'%stop)+'.sdf')
			file_list.append(next)
		return file_list
	else:
		file_list=[]
		if start != stop: # makes sure we aren't dealing with a remainder file
			for i in range(start,stop+1):
				try:
					next=sdf.read(('%05d'%i)+'.sdf')
					file_list.append(next)
				except: 
					print('Failed to load: {}.sdf'.format('%05d'%i))
					file_list.append(None)
		else:
			next=sdf.read(('%05d'%stop)+'.sdf')
			file_list.append(next)
		return file_list

# Lists all the sdf files in the given simulation, converts this to a list of indices and returns it (as list of numbers)
def list_sdf(sim_file_loc):
	sdf_list = [i for i in os.listdir(sim_file_loc) if ".sdf" in i]
	index_list = np.sort(np.asarray([s.replace('.sdf','') for s in sdf_list], dtype=int))
	return index_list

# Reads a given sdf file as per the index
def sdfread(index):
#	print('====================='+str(index))
	try:
		d = sdf.read(('%04d'%index)+'.sdf')
	except:
		d = sdf.read(('%05d'%index)+'.sdf')
	
#	keys = d.__dict__.keys() # returns all the names of quantities in the simulation sdf file
#	for k in keys: print(k)
	return d

# Gives info about the simulation grid (used for dxdydz and more later)
def getGrid(d):
	return d.__dict__["Grid_Grid_mid"].data

# Takes in the grid shape and size then returns the total length across x,y and z (depending on dimensionality)
def getGridlen(d):
	# TODO: Make more general so that we can define lx,ly,lz and then return them all based on dimensions?
	l = getGrid(d)
	if len(l) == 1:
		lx = l[0]
		del l
		return lx[-1] - lx[0]
	elif len(l)==2:
		lx = l[0] ; ly = l[1]
		del l
		return lx[-1]-lx[0] , ly[-1]-ly[0]
	elif len(l)==3:
		lx = l[0] ; ly = l[1] ; lz = l[2]
		del l
		return lx[-1]-lx[0] , ly[-1]-ly[0], lz[-1]-lz[0]

# Returns the physical size of domain steps in x,y,z (depending on dimension of sim)
def getdxyz(d):
	l=getGrid(d)
	if (len(l)==1):
		lx = l[0]
		dx = (lx[-1]-lx[0])/(len(lx)-1)
		del lx ,l
		return dx
	elif (len(l)==2):
		lx = l[0]
		dx = (lx[-1]-lx[0])/(len(lx)-1)
		ly = l[1]
		dy = (ly[-1]-ly[0])/(len(ly)-1)
		del lx, ly , l
		return dx, dy
	elif (len(l)==3):
		lx = l[0]
		dx = (lx[-1]-lx[0])/(len(lx)-1)
		ly = l[1]
		dy = (ly[-1]-ly[0])/(len(ly)-1)
		lz = l[2]
		dz = (lz[-1]-lz[0])/(len(lz)-1)
		del lx, ly, lz , l
		return dx, dy, dz

# Returns a list of times (seconds) from each sdf file
def getTimes(files):
	# Scans all sdf objs in the "files", returns the times and stores into an array
	times=np.zeros((len(files)))
	for i in range(0,len(files)):
		times[i] = files[i].__dict__['Header']['time']
		# print(files[i].__dict__['Header']['time'])
	# times[0] =float(0) #if you start from t=0, first timestep be small but not zero, so make it
			#0. Remove if not starting from 0. There is probably a better way of doing this
	# could re-code this to not use the file list
	return times

# Appends an array of times batch-wise
def batch_getTimes(times,start,stop):
	if start == stop:
		file = sdfread(len(index_list)-1) 
		times[-1] = file.__dict__['Header']['time']
	for i in range(start,stop+1):
		file = sdfread(i)
		times[i] = file.__dict__['Header']['time']
		# print(files[i].__dict__['Header']['time'])
	# times[0] =float(0) #if you start from t=0, first timestep be small but not zero, so make it
			#0. Remove if not starting from 0. There is probably a better way of doing this
	# could re-code this to not use the file list, also depends on index_list which is not passed but presumably a variable
	del file
	return times

# Gets the total simulation ranges for a len(filelist) < 1024
def getSimRanges(files):
	grid_length = getGridlen(files[0])
	N_grid_points = len(getGrid(files[-1])[0])
	times = getTimes(files)
	duration = times[-1] - times[0]
	N_time_points = len(files)
	return grid_length, N_grid_points, duration, N_time_points

# Gets the simulation ranges from a batch-wise loaded array
def batch_getSimRanges(SIM_DATA):
	index_list,file0,filelast,times = SIM_DATA
	grid_length = getGridlen(file0)
	N_grid_points = len(getGrid(filelast)[0])
	duration = times[-1] - times[0]
	N_time_points = len(index_list)
	return grid_length, N_grid_points, duration, N_time_points

# Gets dispersion limits of a simulation for a len(filelist) < 1024
def getDispersionlimits(files):
	grid_length, N_grid_points, duration, N_time_points = getSimRanges(files)
	wfrac = 2.0*const.PI/duration 		#2pi/time of sim		(2pi/T)	
	kfrac = 2.0*const.PI/grid_length 		#2pi/length of grid 	(2pi/L)
	klim = kfrac*0.5*N_grid_points
	wlim = wfrac*0.5*N_time_points
	# highest freq we can resolve # see OneNote lab-book for more info (utilises Nyquist theorem)
	return  klim, wlim #,kfrac, wfrac

# Gets dispersion limits of a simulation for a batch-wise loaded array
def batch_getDispersionlimits(SIM_DATA):
	grid_length, N_grid_points, duration, N_time_points = batch_getSimRanges(SIM_DATA)
	wfrac = 2.0*const.PI/duration 		#2pi/time of sim		(2pi/T)	
	kfrac = 2.0*const.PI/grid_length 		#2pi/length of grid 	(2pi/L)
	klim = kfrac*0.5*N_grid_points
	wlim = wfrac*0.5*N_time_points
	# highest freq we can resolve # see OneNote lab-book for more info (utilises Nyquist theorem)
	del grid_length, N_grid_points, duration, N_time_points, wfrac, kfrac
	return  klim, wlim

def norm_DispersionLimits(DISP_DATA):
	index_list, file0, filelast, times, klim, wlim, wc_maj, wce, va, lambdaD, wpe, wpi, wnorm = DISP_DATA
	grid_length, N_grid_points, duration, N_time_points = batch_getSimRanges((index_list,file0,filelast,times))
	# klim,wlim = batch_getDispersionlimits((self.index_list,self.file0,self.filelast,self.times)) # non-normalised units
	_,_,klim_prime,wlim_prime,tlim_prime = norm_PlottingLimits((index_list,file0,filelast,times),klim,wlim,wc_maj,wce,va,lambdaD,wpe,wpi,wnorm)
	## k,w,t_prime are now normalised according to whether there is a alfven wave present (hence B field)
	## TODO; change this so that I can choose the normalisation
	return klim_prime, wlim_prime, tlim_prime

# Returns the quantity of a given time-step
def getQuantity1d(d, quantity):
	quan = d.__dict__[quantity]
	# quan_name, quan_units = quan.name, quan.units
	# x = quan.grid.data[0]
	# x = np.array(x[1:])
	quan_array = np.array(quan.data)
	return quan_array

# Returns the mean of a quantity
def getMeanquantity(d,quantity): 
	mean_quan = getQuantity1d(d,quantity)
	mean_quan = np.mean(mean_quan)
	return mean_quan

# Returns the mean of a given field: e.g. quantity = 'Electric_Field_E' or 'Magnetic_Field_B'
def getMeanField3D(d,quantity): #include all of the header apart from last charchter (x,y,z) which is the coord
	try:
		mean_x = getMeanquantity(d,quantity+'x')
	except:
		mean_x = 0
	
	try:
		mean_y = getMeanquantity(d,quantity+'y')
	except:
		mean_y = 0

	try:
		mean_z = getMeanquantity(d,quantity+'z')
	except:
		mean_z = 0

	#doesn't account for sign so will return lower even if amplitude is large and negative
	# modified so that it will try to read the component, returning 0 otherwise
	return (mean_x**2+mean_y**2+mean_z**2)**0.5

# Gets the mass (m) of a species defined by a dictionary
def getMass(species):
	masses = {'Electrons': const.me,
	'Left_Electrons': const.me,
	'Right_Electrons': const.me, 
	'Protons': const.me*const.me_to_mp, 
	'PProtons': const.me*const.me_to_mp, 
	'Alphas': const.me*const.me_to_malpha, 
	'Alpha': const.me*const.me_to_malpha, 
	'Deuterons': const.me*const.me_to_D2,
	'Deutrons': const.me*const.me_to_D2, # sometimes misspelled
	'Tritium' : const.me*const.me_to_mT,
	'Tritons' : const.me*const.me_to_mT,
	'Helium3': const.me*const.me_to_He3, 
	'He3': const.me*const.me_to_He3,
	'Ions': const.me*const.me_to_mp} # assuming by Ions the user means Protons

	if species not in masses:
		if species == '':
			None
		else:
			print('Species [{}] mass is not in dictionary, check name passed for spelling mistakes'.format(species))
		raise SystemExit
	else: 
		return masses.get(species)

# Gets the charge number (Z) of a species defined by a dictionary
def getChargeNum(species):
	charges = {'Electrons': 1,
	'Left_Electrons': 1,
	'Right_Electrons': 1, 
	'Protons': 1, 
	'PProtons': 1, 
	'Alphas': 2, 
	'Alpha': 2,
	'Deuterons': 1,
	'Deutrons': 1, # sometimes misspelled
	'Tritium': 1,
	'Tritons': 1,
	'Helium3': 2, 
	'He3': 2,
	'Ions': 1,
	'': 0}

	if species not in charges:
		if species == '':
			print('No species provided, Z=0')
		else:
			print('Species [{}] charge number is not in dictionary, check name passed for spelling mistakes'.format(species))
		raise SystemExit
	else: 
		return charges.get(species)

def getIonlabel(species):
	labels = {'Electrons': r'$e$',
	'Left_Electrons': r'$e$',
	'Right_Electrons': r'$e$', 
	'Protons': r'$p$', 
	'PProtons': r'$p$', # sometimes have two of the same species
	'Alphas': r'$\alpha$', 
	'Alpha': r'$\alpha$', 	
	'Deuterons': r'$D_2$',
	'Deutrons': r'$D_2$', # sometimes misspelled
	'Tritium': r'$T_3$',
	'Tritons': r'$T_3$',
	'Helium3': r'$He_3$', 
	'He3': r'$He_3$',
	'Ions': 'Ions'}
	
	if species not in labels:
		print('Species [{}] label is not in dictionary, check name passed for spelling mistakes'.format(species))
		raise SystemExit
	else: 
		return labels.get(species)
	
def getOmegaLabel(species):
	labels = {'Electrons': r'$\Omega_e$',
	'Left_Electrons': r'$\Omega_e$',
	'Right_Electrons': r'$\Omega_e$', 
	'Protons': r'$\Omega_p$', 
	'PProtons': r'$\Omega_p$', 
	'Alphas': r'$\Omega_\alpha$', 
	'Alpha': r'$\Omega_\alpha$', 	
	'Deuterons': r'$\Omega_D$',
	'Deutrons': r'$\Omega_D$', # sometimes misspelled
	'Tritium': r'$\Omega_T$',
	'Tritons': r'$\Omega_T$',
	'Helium3': r'$\Omega_He3$', 
	'He3': r'$\Omega_He$',
	'Ions': r'$\Omega_i$',
	'Ion': r'$\Omega_i$'}
	
	if species not in labels:
		print('Species [{}] label is not in dictionary, check name passed for spelling mistakes'.format(species))
		raise SystemExit
	else: 
		return labels.get(species)


# Gets the cyclotron frequency of a species. Is generalised so can work for positive and negatively charged species
def getCyclotronFreq(d,species,Z=0):
	Z = getChargeNum(species)
	return Z*const.qe*getMeanField3D(d,"Magnetic_Field_B")/getMass(species)

# Returns the plasma freq (w_pe), could add a species check here on the mass we need to consider, right now this is harcoded as mass of electron
def getPlasmaFreq(d,species):
	# This 0.5 only works if the plasma is quasi-neutral, TODO: add a check for this (measure density of both species?)
	# return ((0.5*getMeanquantity(d,'Derived_Number_Density')*(const.qe**2))/(getMass('Electrons')*const.e0))**0.5
	return ((getMeanquantity(d,'Derived_Number_Density_'+species)*(const.qe**2))/(getMass(species)*const.e0))**0.5

# Calculates thermal velocity, species and mass are hardcoded to an electron.
def getThermalVelocity(d,species):
	temp = getTemperature(species) 
	return (const.kb*temp/getMass(species))**0.5

# Returns the Alfven Velocity of a simulation assuming one species (given) dominates
def getAlfvenVel(d):
	spec_lst = getIonSpecies(d)
	totden = 0
	for spec in spec_lst:
		try:
			totden += getMass(spec)*getMeanquantity(d,'Derived_Number_Density_'+spec) # should loop through all species in the sim including minority (makes more accurate), doesnt include electrons though...
		except:
			continue
#	# Assumes mass density contribution from electrons is negligible
#	totden = getMass(species)*getMeanquantity(d,'Derived_Number_Density_'+species)
#	totden += getMass('Tritons')*getMeanquantity(d,'Derived_Number_Density_Tritons')
	return getMeanField3D(d,'Magnetic_Field_B')/(const.mu0*totden)**0.5

# Returns the Debye Length at a given time-step (careful to use this in loops as temperature and density will change)
def getDebyeLength(d, species): # Species is supurfulous as simulation should be quasi-neutral
	temp = getTemperature(species)
	return (const.e0*const.kb*temp/(getMeanquantity(d,"Derived_Number_Density_"+species)*const.qe**2))**0.5 #as ne=ni just take half of total, should use just electron density

def getTemperature(species):
	temp = 0
	for i in range(0,1000):  #assuming within the first 1000 files there is temperature within a file 
		try:
			temp = getMeanquantity(sdfread(i),'Derived_Temperature_'+species)
			print('THIS IS THE TEMPERATURE in eV ::', temp)
			return (temp*1/(const.eV_to_K))*1E3
		except:
			continue
	
	if temp == 0:
		print('# ERROR # : Can\'t get temperature from files...')
		temp = float(input('Temperature of {}? [keV] :: '.format(species)))

	return temp*const.eV_to_K*1E3 # in keV
	
	
# Returns the Larmor radius of a species depending on the mean magnetic field, and their average energy
def getLarmorRadius(d,species):
	try:
		ek = getMeanquantity(d, 'Derived_EkBar_'+species)		
	except:
		ek = getMeanquantity(d, 'Derived_Average_Particle_Energy_'+species)

	return np.sqrt(2*getMass(species)*ek)/(const.qe*getChargeNum(species)*getMeanField3D(d,'Magnetic_Field_B'))

# Returns the plotting limits of a 2d FT fieldmatrix which looks to see if there is a magnetic field present (va flag) or not
def PlottingLimits(files, species, Z):
	wci   = getCyclotronFreq(files[0],species,Z)
	va   = getAlfvenVel(files[0])
	lambD= getDebyeLength(files[0],'Electrons')
	wpe	 = getPlasmaFreq(files[0],species='Electrons')
	wpp  = getPlasmaFreq(files[0],species=species)
	# wp   = np.sqrt(wpp*2+wpe*2)
	# u0x  = getThermalVelocity(files[0],'Electrons')# abs(getMeanquantity(files[0],'Particles_Vx_Electrons')) # TODO: Change this form being hardcoded to just that of Electrons
	# print('Check here:',u0x, wpe, u0x/wpe, lambD)
	
	if wci or va != 0:
		print('Using alfven & wci normalisation...')
		klim = getDispersionlimits(files)[0]*(va/wci)
		wlim = getDispersionlimits(files)[1]*(1/wci)
		tlim = getSimRanges(files)[2]*(wci/(2*const.PI))
	else:
		print('Using lambda & wpe normalisation...')
		klim = getDispersionlimits(files)[0]*lambD
		wlim = getDispersionlimits(files)[1]*(1/wpe)
		tlim = getSimRanges(files)[2]*(wpe/(2*const.PI))
	# klim = getDispersionlimits(files)[0]*(u0x/wpe)
	# klim = getDispersionlimits(files)[0]*(va/wci)
	# wlim = getDispersionlimits(files)[1]*(1/wci)
	# tlim = getSimRanges(files)[2]*(wci/(2*const.PI))

	print('duration: ', getSimRanges(files)[2])
	print('va=',va,'wci=',wci,'wpe=',wpe)
	print('klim=',klim,'wlim=',wlim,'tlim=',tlim)
	return wci, va, klim, wlim, tlim

# Normalises plotting limits to the klim and wlim for a batch-wise loaded array
def norm_PlottingLimits(SIM_DATA,klim,wlim,wci,wce,va,lambD,wpe,wpp,wnorm):
	#index_list, file0, filelast, times = SIM_DATA
	if wci or va != 0:
		print('Using alfven & wci normalisation...')
		klim_prime = batch_getDispersionlimits(SIM_DATA)[0]*(va/wci)
		wlim_prime = batch_getDispersionlimits(SIM_DATA)[1]*(1/wnorm)
		tlim_prime = batch_getSimRanges(SIM_DATA)[2]*(wnorm/(2*const.PI))
	else:
		print('Using lambda & wpe normalisation...')
		klim_prime = batch_getDispersionlimits(SIM_DATA)[0]*lambD
		wlim_prime = batch_getDispersionlimits(SIM_DATA)[1]*(1/wnorm)
		tlim_prime = batch_getSimRanges(SIM_DATA)[2]*(wnorm/(2*const.PI))
	# klim = getDispersionlimits(files)[0]*(u0x/wpe)
	# klim = getDispersionlimits(files)[0]*(va/wci)
	# wlim = getDispersionlimits(files)[1]*(1/wci)
	# tlim = getSimRanges(files)[2]*(wci/(2*const.PI))

	print('va=',va,'wci=',wci,'wpe=',wpe)
	print('klim_prime=',klim_prime,'wlim_prime=',wlim_prime,'tlim_prime=',tlim_prime)
	return wci, va, klim_prime, wlim_prime, tlim_prime

# Returns a matrix of t vs x of a quantity. Can be used to get an EM field before an FT and/or visualise an entire sim in one plot
def getfieldmatrix(files,quantity): 
	fieldmatrix = np.zeros([len(files),len(getGrid(files[0])[0])]) # note it relies on another function
	for i in range(0,len(files)): #loop over all files/timesteps
		fieldmatrix[i][:] = getQuantity1d(files[i],quantity)	#populates each row with the field values at that time
	return fieldmatrix #returns matrix with field values at each t,x

# Plots a variable "varname" through x-space at a single time-step 
def plot1d(d,varname):
	fig,ax=plt.subplots(figsize=(5,5))
	var=d.__dict__[varname] #pull data we want
	x=var.grid.data[0] #without the '[0]' this is a tuple with only 1 element
	ax.plot(x[1:], var.data, 'r+-') #staggered grid, the length has one more value than all quantities found so far, throw 1st away...
	ax.set_xlabel(var.grid.labels[0] + r' $(' + var.grid.units[0] + ')$',fontsize='12') #labels are in the sdf file, may need to reconsider x units
	ax.set_ylabel(var.name + r' $(' + var.units + ')$',fontsize='12') #as are the units]
	# fig.savefig(var.name+'.jpeg')
	fig.savefig(home_path+varname+'.jpeg',bbox_inches='tight')
	ax.clear()

# Moving average calculation of a quantity
def temporal_average(index_list,quantity,points=np.linspace(-1,1,3,dtype=int)):
	# TODO : Make this use the file_list rather than index_list and then phase this index_list out from the whole code
	# Moving average calculation of a quantity over N-points. Evenly weighted in time (backwards and forwards) and discards points outside of range of
	# index list. e.g. points=[-1,0,1] will carry out moving average either side of data point (including itself '0').
	valtspec=[]
	time=[]
	n = max(index_list)
	for index in index_list:
		# time.append(getQuantity1d(sdfread(index),'Wall_time'))
		vald = 0.
		counter = 0.
		for p in points:
			if index+p < 0 or index+p > n:
				continue
			else:
				dp = sdfread(index+p)
				vald += abs(getMeanquantity(dp,quantity))
				counter += 1
				del dp
		valtspec.append(vald/counter)
	del n, vald

	return valtspec, time


# Plots the velocity vs. real-space diagram (Phase space) using the list of files found in setup(). Then saves them to home dir  
def plotPhaseSpace(files,species):
	# TODO: Need to make this more general so we can plot the phase space of just one beam or species
	print('PLOTTING PHASE SPACE')
	if len(species) == 3: colors = cycle(['g','r','b'])
	elif len(species) == 2: colors = cycle(['r','b'])
	else: colors = cycle(['r'])
	next(colors)
	print(len(species))
	times = getTimes(files)
	d0 = files[0]
	LambdaD = getDebyeLength(d0,species[0])
	print('LambdaD = ',LambdaD)
	w_p = getPlasmaFreq(d0)
	Omega_cp = getCyclotronFreq(d0,species,1)
	times = (times*w_p)/(2*const.PI)
	v_th = getThermalVelocity(d0,species[0])
	del d0

	files = files[0::int(len(files)/100)]
	fig, ax = plt.subplots(figsize=(10,5))
	for i in range(len(files)):
		for name in species:
			print(i)
			# pos_x_elec = getQuantity1d(d,'Grid_Particles_electron')[0]  
			# x_left = getQuantity1d(d,'Grid_Particles_Left')[0]  
			vx_name = files[i].__dict__['Particles_Vx_'+name]
			# vx_2 = files[i].__dict__['Particles_Vx_'+species2]
			x_name = vx_name.grid.data[0]/LambdaD 		# Normalises distance to units of Debye length
			# x2 = vx_2.grid.data[0]/LambdaD
			if i == 0: # set plot limits
				xmax = max(x_name)
				xmin = min(x_name)
				# xmax = max(x1)+100
				# xmin = min(x1)-100
			color = next(colors)
			ax.scatter(x_name,vx_name.data/v_th,color=color,s=1.) 		# Normalises speed to v_thermal at t=0
		# ax.scatter(x2,vx_2.data/v_th,color='r',s=1.)	
		ax.set_xlabel(r'$x/\lambda_D$',fontsize=18)
		ax.set_ylabel(r'$v_x/v_{th}$', fontsize=18)
		# ax.set_yticks([-180,-135,-90,-45,0,45,90,135,180])
		# ax.set_ylim(-180,180)
		# ax.set_ylim(-120,120)
		# ax.set_xlim(xmin,xmax)
		# ax.set_xticks([0,500,1000,1500,2000,2500])
		# ax.set_xticklabels(['0','500','1000','1500','2000','2500'])

		ax.set_title(r'$t\omega_{pe}/2\pi = $'+str(np.around(times[i],3)), fontsize=18)
		fig.savefig(home_path+'velocity{}.jpeg'.format(i), bbox_inches="tight")
		ax.clear()

# Plots the electric field potential in the x direction using the numpy.gradient() function so as to maintain array size.
def plotPhi(files,species='Electrons'):
	print('PLOTTING PHI THROUGH TIME')	
	d0 = files[0]
	LambdaD = getDebyeLength(d0,species=species)
	w_p = getPlasmaFreq(d0)
	Omega_cp = getCyclotronFreq(d0,'Protons',1)
	times = getTimes(files)*(w_p)/(2*const.PI)
	del d0

	fig,ax=plt.subplots(figsize=(10,5))
	i=0
	for file in files:
		EX = file.__dict__['Electric_Field_Ex']
		x = EX.grid.data[0]
		dx = getdxyz(file)
		# phi = -1*np.gradient(EX, dx)
		EX = np.array(EX.data)
		phi = -1*EX*dx
		
		ax.plot(x[1:]/LambdaD,phi)

		ax.set_ylabel(r'$\phi$' +' ' +r'$ [V]$',fontsize=18)
		ax.set_xlabel(r'$x/\lambda_D$',fontsize=18)
		# ax.set_ylim(-200,200)
		# ax.set_yticks([-7.5,-6.0,-4.5,-3.0,-1.5,0,1.5,3.0,4.5,6.0,7.5])
		# ax.set_yticklabels([r'$-7.5$',r'$-6.0$',r'$-4.5$',r'$-3.0$',r'$-1.5$',r'$0.0$',r'$1.5$',r'$3.0$',r'$4.5$',r'$6.0$',r'$7.5$'])
		ax.set_title(r'$t\omega_{pe}/2\pi = $'+str(np.around(times[i],3)), fontsize=18)
		
		fig.savefig(home_path+'phi{}.jpeg'.format(i), bbox_inches='tight')
		ax.clear()
		print(i)
		i+=1

# Plots the mean of the absolute electric field at each time step. From this we can plot the instability growth rate (characterised by gamma)  
def plotAveEX(files,species):
	print('PLOTTING MEAN OF ABSOLUTE EX')
	Etot=[]
	d0 = files[0]
	w_p = getPlasmaFreq(d0)
	gamma = w_p/2
	LambdaD = getDebyeLength(d0,species)
	print('gamma: '+str(gamma))
	OFFSET = 15.7
	times = getTimes(files)
	dt = times[1] - times[0]
	GAMMA = np.exp((gamma*times)-OFFSET)
	# times = (times*w_p)/(2*const.PI)
	del d0
	
	for i in range(len(files)): 
		EX = getQuantity1d(files[i],'Electric_Field_Ex')
		EXmean = np.mean(np.abs(EX))
		Etot.append(EXmean)

	EXunits = files[0].__dict__['Electric_Field_Ex'].units
	fig, ax = plt.subplots(figsize=(6,6))

	## Finding optimum offset
	# first = np.where(times>0.01)[0][0]
	# last = np.where(times<0.1)[-1][-1]
	# print(first,last)
	# Etot_dummy = Etot[first:last]
	# time_dummy = np.linspace(0.025,0.1,len(Etot_dummy))
	# pars, cov = curve_fit(f=exponential, xdata=time_dummy, ydata=Etot_dummy, p0=[0], bounds=(-np.inf, np.inf))
	# print(pars)
	# ax.plot(time_dummy,exponential(time_dummy,pars[0]),linestyle='--',color='b')

	ax.plot(times,Etot,color='b',linestyle='-',alpha=1.)
	# ax.plot(times[first[0][0]:last[-1][-1]],GAMMA[first[0][0]:last[-1][-1]],linestyle='--',color='r')
	ax.plot(times,GAMMA,linestyle='--',color='r')

	left, bottom, width, height = [0.41, 0.22, 0.45, 0.2]
	ax2 = fig.add_axes([left, bottom, width, height])
	ax2.plot(times,Etot,color='b',linestyle='-')
	ax2.plot(times,GAMMA,linestyle='--',color='r')
	ax2.set_xlim(0,0.08)
	ax2.set_ylim(1E-6,1E-3)
	ax2.set_yscale('log')
	ax2.tick_params(axis='both', which='major', labelsize=10)
	# ax2.set_ylabel(r'$\langle|E_x|\rangle $'+'  '+r'$ [ $'+ EXunits +r'$ ]$',fontsize=14)
	# ax2.set_xlabel(r'$t$' + ' ' +r'$ [s] $',fontsize=14)


	ax.set_ylim(1E-6,1E-2)
	ax.set_yscale('log')
	ax.set_ylabel(r'$\langle|E_x|\rangle $'+'  '+r'$ [ $'+ EXunits +r'$ ]$',fontsize=18)
	ax.set_xlabel(r'$t$' + ' ' +r'$ [s] $',fontsize=18)
	ax.annotate(r'$\gamma = {}$'.format(np.around(gamma,1)), xy=(0.25,0.85), xycoords='axes fraction', color='r')

	fig.savefig(home_path+'EXmean_t.jpeg',bbox_inches='tight')

# Plots any general quantity that covers time and space then plots a normalised heat map of this value (if norm='on') and saves it accordingly.
def plotNormHeatmap(files,quantity,species,norm='on'):
	print('PLOTTING NORMALISED HEATMAP')
	quanMatrix = getfieldmatrix(files,quantity)
	w_pe = getPlasmaFreq(files[0])
	LambdaD = getDebyeLength(files[0],species)  # assumes this to be electrons but can be generalised e.g. 'Left' and 'Right' for two stream case
	dx = getdxyz(files[0])
	quan_norm=np.ndarray(shape=quanMatrix.shape)
	for t in range(len(files)):
		# quan_hold = -1*quanMatrix[t][0:]*dx 		# used for phi (electrostatic potential) rather than purely Ex
		quan_hold = quanMatrix[t][0:]
		if norm == 'on': 
			max_quan = max(abs(np.array(quan_hold)))
		else: 
			max_quan = 1.
		quan_norm[t][0:] = np.array(quan_hold)/max_quan
		del quan_hold
	
	fig,ax=plt.subplots(figsize=(8,6))
	x = files[0].__dict__[quantity].grid.data[0]/LambdaD
	extent = [0,(getTimes(files)[-1]*(w_pe/(2*const.PI))),0,max(x)]

	quan_norm = quan_norm[1:][:] # removes empty column in time domain
	im = plt.imshow(quan_norm.transpose(), interpolation='nearest',extent=extent,cmap='jet',origin='lower',aspect='auto')
	cbar = plt.colorbar(label=r'$\hat{\phi}$')
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(12)
	ax.set_xlabel(r'$t\omega_p/2\pi$',fontsize=18)
	ax.set_ylabel(r'$x/\lambda_D$',fontsize=18)

	fig.savefig(home_path+'heatmap_{}.jpeg'.format(quantity),bbox_inches='tight')

# =================================================
# =================================================

def HanningWindowT(fieldmatrix):
	# look at the "np.hanning documentation"
	new_field = np.zeros((fieldmatrix.shape[0],fieldmatrix.shape[1]))
	han = np.hanning(fieldmatrix.shape[0]) #window with length of x dim
	for i in range(0, fieldmatrix.shape[1]): #loop over t vignal.spectrogramalues
		new_field[:,i] = fieldmatrix[:,i]*han
	del han
	return new_field

def HanningWindow2D(field):
	#new_field = np.zeros((field.shape[0],field.shape[1]))
	hant = np.hanning(field.shape[0]) #window with length of x dimHanningWindowT
	hanx = np.hanning(field.shape[1])
	han2D = np.outer(hant,hanx)
	new_field = field*han2D
	return new_field

def HanningWindowK(field):
	new_field = np.zeros((field.shape[0],field.shape[1]))
	han = np.hanning(field.shape[1]) #window with length of x dim
	for i in range(0, field.shape[0]): #loop over t values
		new_field[i,:] = field[i,:]*han
	return new_field

## TODO; make a new function where the user can choose what type of normalisation they want (by default this is just read in atm using va as a trigger)
# def Norm_Type():

# Get the 1d FT of field data
def get1dTransform(fieldmatrix, window=False, start=0): #takes log of absolute shifted 1D FT (x -> k) of the matrix from getfieldmatrix()
	# read the "numpy.fft" documentation
	if window:
		fieldmatrix = HanningWindowT(fieldmatrix)
	else:
		print('!# Warning #! : 1d FFT is not being Hanning windowed.')

	preshiftFT = np.zeros((fieldmatrix.shape[0], fieldmatrix.shape[1]),complex)
	for t in range(0, fieldmatrix.shape[0]):
		preshiftFT[t][:]=np.fft.fft(fieldmatrix[t][:])
	
	shift = np.fft.fftshift(preshiftFT[start:,start:],1)
	FT = np.abs(shift) # Take modulus as result is complex
	FT_half = FT[:,int(FT.shape[1]/2):] # All time, but only k > 0
	del preshiftFT, FT
	return FT_half


# Plot the 1d FT of field data (using the above get1d function) and plots it with time on the y-axis
def plot1dTransform(FT_matrix, va_tci, klim, tlim, Omega_label = r'$\Omega_D$', cbar=False, cmap='magma'): # Plots t vs k as a heat map of a field quantity
	# Pass it the matrix from "get1dTranform". Also pass it klim and tlim from "PlottingLimits()"
	trFT = np.log10(FT_matrix[:][1:])
	extent=[0, klim, 0, tlim]
	fig, ax = plt.subplots(figsize=(8,8))
	im = plt.imshow(trFT, interpolation='nearest',extent=extent,cmap=cmap,origin='lower',aspect='auto')
	
	va, tci = va_tci
	if va == 0:
		ax.set_xlabel(r'$k\lambda_D$', fontsize=18) #need to change when normalising 
		ax.set_ylabel(r'$t\omega_{pe}/2\pi$',fontsize=18)
	else:
		ax.set_xlabel(r'$kv_A/$'+Omega_label,fontsize=18)
		if tci:
			ax.set_ylabel(r'$t$'+Omega_label+r'$/2\pi$',fontsize=18)
#		else: # what else would be plotted here?
#			ax.set_ylabel(r'$t\omega_{pe}/2\pi$',fontsize=18)

	if (cbar): # off by default
		plt.colorbar()
	del trFT
	# ax.axhline(np.sqrt(3)/2,linestyle='--',color='k')
#	fig.savefig(home_path+'k_t',bbox_inches='tight')
	return fig, ax


# Get the 2d FT of field data
def get2dTransform(fieldmatrix,window=True):
	if window: #windowing enabled by default
		preshift = np.fft.fft2(HanningWindowT(fieldmatrix))[:,:]
	else:
		preshift = np.fft.fft2(fieldmatrix)[:,:] 
	shift = np.fft.fftshift(preshift) # shift to zero-freq so is symmetrical (read numpy fft.fftshift documentation)
	shiftchopped = shift[int(fieldmatrix.shape[0]/2):,int(fieldmatrix.shape[1]/2):] # only take positive frequencies (w) and wavenumbers (k)
	# shiftchopped = shift[:,:] # plots the whole FFT space
	transmatrix= np.abs(shiftchopped) 
	del shiftchopped, preshift, shift
	return transmatrix

# Plot the 2d FT of field data using the above get2d function
def plot2dTransform(FFT_matrix, va_wci, klim, wlim , Omega_label=r'$\Omega_i$' ,cbar=False, clim=(None,None), cmap='magma'):
	# In:
	#	FFT_matrix , 2d FFT matrix of a field quantity e.g. Magnetic_Field_Bz
	#	va_wci , 2d array with the actual value and then a boolean to see whether the normalisation is in fact the core ion cyclotron frequency (True) or something else (False) 
	#		e.g. va_wci = [9E6,True]
	#	klim , the extent to want to plot and show the heatmap
	#	wlim , same as above but for the frequency component
	#	cbar , colour bar boolean
	# Out:
	#	returns the figure and axis plotted so can add own annotations etc
	tr = np.log10(FFT_matrix)[1:,1:]
	extent=[0,klim,0,wlim]
	print(extent)
	fig, ax = plt.subplots(figsize=(8,4))
	va, wci = va_wci
	if va == 0:
		ax.set_xlabel(r'$k\lambda_D$', fontsize=18) #need to change when normalising 
		ax.set_ylabel(r'$\omega/\omega_{pe}$',fontsize=18)
	else:
		ax.set_xlabel(r'$kv_A/$'+Omega_label,fontsize=18)
		if wci:
			ax.set_ylabel(r'$\omega/$'+Omega_label,fontsize=18)
#		else: #what else would be plotted here?
#			ax.set_ylabel(r'$\omega/\Omega_{e}$',fontsize=18)

	# k = np.linspace(-FT_lims[2],FT_lims[2],71)
	# ax.set_xlabel(r'$ku_{0x}/\omega_{pe}$',fontsize=18))
	# w1 = np.sqrt(k**2+1+np.sqrt(4*k**2+1))
	# w2 = np.sqrt(k**2+1-np.sqrt(4*k**2+1))
	# ax.plot(k,w1,k,w2, color='k')
	# ax.plot(k,abs(k),color='k',linestyle='--')
	# ax.axvline(x=np.sqrt(3)/2, ymin=0, linestyle='-.', color='k')
	# ax.axvline(x=-np.sqrt(3)/2, ymin=0, linestyle='-.', color='k')
	
	# ax.set_xlim(0,klim)
	# ax.set_ylim(0,wlim)
	im = plt.imshow(tr, interpolation='nearest',extent=extent ,cmap=cmap, origin='lower', aspect='auto',clim=clim)
	del tr
	if (cbar):
		plt.colorbar()
	# fig.savefig(home_path+'k_w',bbox_inches='tight')
	return fig, ax


def plotting(fig,ax,name):
	try:
		fig.savefig(name+'.jpeg',bbox_inches='tight')
	except:
		fig.savefig(name+'.png',bbox_inches='tight')
		print('plotted using .png format')
	plt.clf()
	return None

def plotDispersion(transmatrix, klimlow, klimup, wlimlow, wlimup, cbar=False, clim = (None,None),  labels=False): 
	tr = np.log10(transmatrix)[1:,1:] # "[1:,1:]" gets rid of the first row and coloumn. change to [0:,0:] and you'll see why this should be done
	# often want to chop to get better colour contrast. So pass in the extent. Usually this will be [0, klim, 0, wlim] where klim and wlim come from "plotting_vals"
	extent=[klimlow,klimup,wlimlow,wlimup]
	# experiment with clim, "(None, None)" means it will choose it for you, usually what you want. Sometimes I alter it to get better contrast
	if (labels):
		plt.xlabel('Wavenumber' + r' $[\omega_{c}/V_{A}]$',fontsize='15') #need to change when normalising 
		plt.ylabel('Frequency' + r' $[\omega_{c}]$',fontsize='15') #need to change when normalising 
	im = plt.imshow(tr, interpolation='nearest',extent=extent ,cmap='jet',origin='lower', aspect='auto',clim=clim)#, clim = (-4.0,None))
	if (cbar): # off by default
		plt.colorbar()
	del tr
	return im
	
# =================================================

# Finds the limits of a batch size with its start and stop parameters
def batchlims(n,batch_size,index_list,remainder):
	batch_ini, batch_fin = n*batch_size+1, (n+1)*batch_size # assures no overlap between start and stop positions between iterations
	if n == 0: # should account for if it is the first one, it includes 0th file and hence needs to be changed
		batch_ini = 0
	if remainder: # accounts for remainder, needs to be passed new argument which will be True or False
		if remainder==1:
			print('Remainder Handled with 0th file.')
		elif remainder==2:
			print('Remainder x1...')
			batch_ini = index_list[-1]
			batch_fin = batch_ini
			print('Handled.')
		else:
			print('Remainder...')		
			batch_ini = index_list[-1]+2-n
			batch_fin = index_list[-1]
			print('Handled.')
	return batch_ini, batch_fin

# Loads a section of a fieldmatrix batch wise
def batch_fieldmatrix(start,stop,quantity,file0,len04d=False):
	if start != stop:
		files = batch_filelist(start,stop,len04d)
		field_matrix = np.zeros([(stop-start+1),len(getGrid(file0)[0])])
		# print('2d batch size: ', np.shape(field_matrix))	
		for j in range(0,(stop-start)+1):
			# print('j {}'.format(j))
			field_matrix[j][:] = getQuantity1d(files[j], quantity)
		return field_matrix
	else:
		files = batch_filelist(start,stop,len04d)
		field_matrix = np.empty((1,len(getGrid(file0)[0])), float)
		print('start {}, stop {}'.format(start,stop))
		# print('2d batch size: ', np.shape(field_matrix))
		field_matrix[0][:] = getQuantity1d(files, quantity)
		return field_matrix

# Loads a total fieldmatrix according to individual batches
def load_tot_fieldmatrix(index_list,N_x,file0,quantity='',batch_size=10,times_load=True):
	## TODO; Check to see how many files there are, will alter how they are loaded (using %04d or %05d depending on trailing zero's)
	len04d = False
	if len(index_list) <= 9999:
		len04d = True
	
	nbatches, remainder = divmod(len(index_list), batch_size)
	tot_field = np.zeros((len(index_list),N_x),float)
	times = np.zeros(len(index_list))

	for i in range(nbatches):
		start,stop=batchlims(i,batch_size,index_list,remainder=False)
		print('batch #: {} , start : {}, stop : {}'.format(i,start,stop))
		fm = batch_fieldmatrix(start,stop,quantity,file0,len04d)
		tot_field = np.append(tot_field,fm,axis=0)
		if times_load:
			times=batch_getTimes(times,start,stop)
	if remainder:
		if remainder == 1:
			print('Remainder handled already.')
		else:
			print('Remaining files: {}'.format(remainder))
			start,stop=batchlims(remainder,batch_size,index_list,remainder=True)
			fm = batch_fieldmatrix(start,stop,quantity,file0,len04d)
			tot_field = np.append(tot_field,fm,axis=0)
			if times_load:
				times=batch_getTimes(times,start,stop)
	
	fieldmatrix = np.array(tot_field[len(index_list):][:]) 	# removes empty first half of the array
	times = np.array(times)				# just to make sure it is in the correct format
	del tot_field, fm, start, stop

	return fieldmatrix, times

# Plots the cold plasma dispersion for 2 species and for an arbitrary angle (theta) of B field 
def coldplasmadispersion(file0, species1, species2, z1, z2, omegas, theta=''):#assumes one of the species (2) is always electrons
	if theta == '':
		theta = float(input('Please input the angle between the equilibrium B field and simulation domain (in degrees).\n!>>'))

	d = file0 ; sin = np.sin(theta*(const.PI/180.0)) ; cos = np.cos(theta*(const.PI/180.0)) ; l = len(omegas)
#	if theta==90:
#		sin=1.0
#		cos=0.0
	print(theta, sin, cos)
	wpe = getPlasmaFreq(d,species2)
	wce = getCyclotronFreq(d,species2,z1)
	wpf = [wpe, getPlasmaFreq(d,species1)]
	wcf = [wce , getCyclotronFreq(d,species1,z1)]

	R = np.ones(l) ; P = np.ones(l) ; L = np.ones(l) ; S = np.zeros(l) ; D = np.zeros(l) 
	B = np.zeros(l) ; F = np.zeros(l) ; A = np.zeros(l) ; C = np.zeros(l) 
	
	R = R - ((wpf[0]**2)/(omegas*(omegas + wcf[0]))) - ((wpf[1]**2)/(omegas*(omegas + wcf[1]))) #neglecting contribution from min_ions
	L = L - ((wpf[0]**2)/(omegas*(omegas - wcf[0]))) - ((wpf[1]**2)/(omegas*(omegas - wcf[1])))
	P = P -  ((wpf[0]**2)/omegas**2)-  ((wpf[1]**2)/omegas**2)

	#for i in range(0,2): #loop over species
		#print (wpf[i]**2)/(omegas*(omegas + wcf[i]))
		#print wpf[i], omegas[0], wcf[i]
		#R = R + (wpf[i]**2)/(omegas*(omegas + wcf[i]))
		#P = P + (wpf[i]**2)/omegas**2
		#L = L + (wpf[i]**2)/(omegas*(omegas - wcf[i]))
	#R = 1.0 - R ; P = 1.0 -P ; L = 1.0 -L

	S = 0.5*(R+L) ; D = 0.5*(R-L)
	C = P*R*L
	B = R*L*(sin**2) + P*S*(1.0 +cos**2)
	F=  (((R*L - P*S)**2)*(sin**4) + 4.0*(P**2)*(D**2)*(cos**2))**0.5
	A = S*(sin**2) + P*(cos**2)
	n1 = np.zeros(l, dtype=complex) ; n2 = np.zeros(l, dtype=complex) ; n3 = np.zeros(l, dtype=complex) ;# n4 = np.zeros(l, dtype=complex) 
	n3 = np.lib.scimath.sqrt((R*L)/S)
	n1 =  np.lib.scimath.sqrt((B+F)/(2.0*A))
	n2 = np.lib.scimath.sqrt((B-F)/(2.0*A))
	del R, P, L, S, D

	#n1 =  np.sqrt((B+F)/(2.0*A))
	#n2 = np.sqrt((B-F)/(2.0*A))
	#n3 = -((B+F)/2.0*A)**0.5 #negative k soloutions, usually not interested in
	#n4 = -((B-F)/2.0*A)**0.5
	del B, F, A
	omegas = np.linspace(0,max(omegas),len(omegas))
	# return n1,n2,n3#,n4,omegas
	return (np.real(n1)*omegas)/const.c , (np.real(n2)*omegas)/const.c , np.real((n3*omegas)/const.c) #, (n4*omegas)/c, omegas


def coldplasmadispersion_twomaj(file0, species1, species2, species3, z1, z2, z3, omegas, theta=''):#assumes one of the species (3) is always electrons
	if theta == '':
		theta = float(input('Please input the angle between the equilibrium B field and simulation domain (in degrees).\n!>>'))

	d = file0 ; sin = np.sin(theta*(const.PI/180.0)) ; cos = np.cos(theta*(const.PI/180.0)) ; l = len(omegas)
#	if theta==90:
#		sin=1.0
#		cos=0.0
	print(theta, sin, cos)
	wpe = getPlasmaFreq(d,species3)
	wce = getCyclotronFreq(d,species3,z3)
	wpf = [wpe, getPlasmaFreq(d,species1), getPlasmaFreq(d,species2)]
	wcf = [wce , getCyclotronFreq(d,species1,z1),getCyclotronFreq(d,species2,z2)]

	R = np.ones(l) ; P = np.ones(l) ; L = np.ones(l) ; S = np.zeros(l) ; D = np.zeros(l) 
	B = np.zeros(l) ; F = np.zeros(l) ; A = np.zeros(l) ; C = np.zeros(l) 
	
	R = R - ((wpf[0]**2)/(omegas*(omegas + wcf[0]))) - ((wpf[1]**2)/(omegas*(omegas + wcf[1]))) - ((wpf[2]**2)/(omegas*(omegas + wcf[2])))#neglecting contribution from min_ions
	L = L - ((wpf[0]**2)/(omegas*(omegas - wcf[0]))) - ((wpf[1]**2)/(omegas*(omegas - wcf[1]))) - ((wpf[2]**2)/(omegas*(omegas - wcf[2])))
	P = P -  ((wpf[0]**2)/omegas**2) -  ((wpf[1]**2)/omegas**2) - ((wpf[2]**2)/omegas**2)

	#for i in range(0,2): #loop over species
		#print (wpf[i]**2)/(omegas*(omegas + wcf[i]))
		#print wpf[i], omegas[0], wcf[i]
		#R = R + (wpf[i]**2)/(omegas*(omegas + wcf[i]))
		#P = P + (wpf[i]**2)/omegas**2
		#L = L + (wpf[i]**2)/(omegas*(omegas - wcf[i]))
	#R = 1.0 - R ; P = 1.0 -P ; L = 1.0 -L

	S = 0.5*(R+L) ; D = 0.5*(R-L)
	C = P*R*L
	B = R*L*(sin**2) + P*S*(1.0 +cos**2)
	F=  (((R*L - P*S)**2)*(sin**4) + 4.0*(P**2)*(D**2)*(cos**2))**0.5
	A = S*(sin**2) + P*(cos**2)
	n1 = np.zeros(l, dtype=complex) ; n2 = np.zeros(l, dtype=complex) ; n3 = np.zeros(l, dtype=complex) ;# n4 = np.zeros(l, dtype=complex) 
	n3 = np.lib.scimath.sqrt((R*L)/S)
	n1 =  np.lib.scimath.sqrt((B+F)/(2.0*A))
	n2 = np.lib.scimath.sqrt((B-F)/(2.0*A))
	del R, P, L, S, D

	#n1 =  np.sqrt((B+F)/(2.0*A))
	#n2 = np.sqrt((B-F)/(2.0*A))
	#n3 = -((B+F)/2.0*A)**0.5 #negative k soloutions, usually not interested in
	#n4 = -((B-F)/2.0*A)**0.5
	del B, F, A
	omegas = np.linspace(0,max(omegas),len(omegas))
	# return n1,n2,n3#,n4,omegas
	return (np.real(n1)*omegas)/const.c , (np.real(n2)*omegas)/const.c , np.real((n3*omegas)/const.c) #, (n4*omegas)/c, omegas


# Plots the power spectrum of a signal from a 2d FFT (trans) matrix
def powerspectrum(trans,wlim,klim,harmonicmin,harmonicmax,kmodelow,kmodehigh):
	# plots fourier power as a function of frequency. For ICE this is what they observe in experiment. very interested in this.
	# trans - input matrix that has been 2D FT'd
	# wlim, klim - opbtain from "getDispersion" limits
	# harmonicmin,harmonicmax,kmodelow,kmodehigh - these are normalised.
	# useful because you can look at a dispersion with normalised axis and see what range of omega and k you want the pwoer for.
	# bearing in mind for ICE your frequency will be normalised to "w_cp" and k normalised to "w_cp/ v_a" 
	# so if you want the power between the 1st and 10th harmonics, harmonicmin = 1,harmonicmax = 10
	# as long as "wlim" has been passed in properly, this function will find the relevent "slice" of "trans" to calculate the power for
	wstart = int(trans.shape[0]*(harmonicmin/wlim)) #determins which minimum harmonic we want to look at
	wstop = int(trans.shape[0]*(harmonicmax/wlim))
	print(wstart,wstop)

	kstop = int(((kmodehigh/klim))*trans.shape[1])
	kstart = int(((kmodelow/klim))*trans.shape[1])
	print(kstart,kstop)

	#kstop = int((trans.shape[1]/2.0) + (kmodehigh/klim)*(trans.shape[1]/2.0))
	#kstart = int((trans.shape[1]/2.0) - (kmodelow/klim)*(trans.shape[1]/2.0))
	power = np.zeros((wstop-wstart))
	#powert0 = np.sum((trans[0,kstart:kstop])**2)
	for i in range(wstart,wstop):
		power[i-wstart] = np.sum((trans[i,kstart:kstop])**2) #positive k only
	omegas = np.linspace((wstart*wlim/trans.shape[0]),(wstop*wlim/trans.shape[0]),len(power))
	return np.log10(power), omegas #notice it returns the log of the power (what we want) - up to you to plot it properly

def power_one_k(trans,wlim,klim,harmonicmin,harmonicmax,kmodelow):
	wstart = int(trans.shape[0]*(harmonicmin/wlim))
	wstop = int(trans.shape[0]*(harmonicmax/wlim))
	
	kstart = 	int(trans.shape[1]*(kmodelow/klim))
	
	power_one_k = np.zeros((wstop-wstart))
	
	for i in range(wstart,wstop):
		power_one_k[i-wstart] =  (trans[i][kstart])**2
	omegas = np.linspace((wstart*wlim/trans.shape[0]),(wstop*wlim/trans.shape[0]),len(power_one_k))

	return np.log10(power_one_k), omegas

# Plots the change in field energy densities from their mean value
def getEnergies(specmin,zmin,specmaj,zmaj,specmaj2,ndata, frac = 1, leg='on',labels=True):
	# inputs: name of minority species, z number of this sopecies, same for majority, same for majority, number of files/data points you want
	# if you make it more than what you have, it will plot them all, in which case just set it to any large number. If ndata is less than the data 
	# that you have it will only plot as many as you specify.
	# frac plots every "frac"th data point. Just set to one (default) 
	# leg plots a legend, default is off
	# gets and plots all energies as a function of time.
	# you need to comment out relevent sections if you don;t want to get that energy.
	#  for ICE you will usually be interested in B_z, E_x, majKE, and minKe
	# Works in nD. Ignore any reference to "1D", such functions are actually general.
	# Summations are done over all axis.
	# ndata is final sdf number+1
	# TODO; could do a moving average on the data witha window equal to the releant gyro period or somethign like that	
	# TODO; make it read in the field components and then try and except each in turn later in the code so that I don't have to comment/uncomment them
	n = int(ndata/frac)

	# n = -1
	# for i in range(ndata):
	# 	try:
	# 		next=sdf.read(('%05d'%i)+'.sdf')
	# 		n = n +1
	# 	except: 
	# 		pass
	# n = int(n/frac) + 1 #now know how many data files we have

#	multiply_by_electric_field = const.e0*0.5 
#	multiply_by_magnetic_field = 1.0/(2.0*const.mu0)
	maj2 = True
	if specmaj2	== '':
		print('Two ion species...')
		maj2 = False
		maj2Ke = 0
	else:
		print('Three ion species...')
		zmaj2 = getChargeNum(specmaj2)
		maj2Ke = np.zeros(n)

	try:
		ion_gp = (2.0*const.PI)/getCyclotronFreq(sdfread(100),specmin,zmin) 
		# tries to see if there is a species names "min_ions" - this is the required name for a species initialised using the ring beam
		# if it find it, normalise to this ion frequency and subsequently try plot its enegry
		minority = True
	except TypeError:
		# otherwise just normalise to the majority species which you specified.
		ion_gp = (2.0*const.PI)/getCyclotronFreq(sdfread(100),specmaj,zmaj)
		minority = False

	multiply_by_time = 1.0/ion_gp #change divisor as appropriate (1.0 is no normalisation)
	# initialise arrays. comment out ones that wont be used to make the code more efficient
	Exenergy = np.zeros(n) #; Eyenergy = np.zeros(n); Ezenergy = np.zeros(n) 
	Bzenergy = np.zeros(n) ; Byenergy = np.zeros(n) #; Bxenergy = np.zeros(n) ;
	majKe = np.zeros(n) ; electronKe = np.zeros(n) ; minKe = np.zeros(n)
	times = np.zeros(n)
	# this is because for ICE we impose a background b field and we are only interested in the fluctuating part.
	# for other applications change/comment out as appropriate
	Bz_t0 = getQuantity1d(sdfread(0), "Magnetic_Field_Bz")

	print('Calculating field energies [may take time to print progress].')
	for i in range(0, n):
		# os.system('cls')
		d = sdfread(i*frac)
		Exenergy[i] = np.mean(getQuantity1d(d, "Electric_Field_Ex")**2)
		#Eyenergy[i] = np.mean(getQuantity1d(d, "Electric_Field_Ey")**2)
		#Ezenergy[i] = np.mean(getQuantity1d(d, "Electric_Field_Ez")**2)
		#Bxenergy[i] = np.mean(getQuantity1d(d, "Magnetic_Field_Bx")**2)
		Byenergy[i] = np.mean(getQuantity1d(d, "Magnetic_Field_By")**2)
		Bzenergy[i] = np.mean((getQuantity1d(d, "Magnetic_Field_Bz")-Bz_t0)**2)
		majKe[i] = getTotalKineticEnergyDen(d,species = specmaj)
		electronKe[i] = getTotalKineticEnergyDen(d,species = 'Electrons')
	
		if(minority):
			minKe[i] = getTotalKineticEnergyDen(d,species = specmin)
		if maj2:
			maj2Ke[i] = getTotalKineticEnergyDen(d,species = specmaj2)
		# try:
		# 	electronKe[i] = getTotalKineticEnergyDen(d,species = "Electrons")
		# 	pic=True # using pic version of epoch
		# except KeyError:
		# 	pic=False #using hybrid version of epoch, don;t need to worry about telling this function which

		times[i] = (d.__dict__['Header']['time'])
		if np.around(100*(i/n),2)%5==0: print('file: {}/{}, Progress: {}%'.format(i,n,np.around(100*i/n,2))) ##print every 5% completion rather than EVERY step

	
	first_en = 0
	startfrom = 0
	take_mean_to = 10

	# wnat to find change in energy density
	meanenergyex = np.mean(Exenergy[:take_mean_to])
	#meanenergyey = np.mean(Eyenergy[:take_mean_to])
	#meanenergyez = np.mean(Ezenergy[:take_mean_to])
	#meanenergybx = np.mean(Bxenergy[first_en])	
	meanenergyby = np.mean(Byenergy[:take_mean_to])
	meanenergybz = np.mean(Bzenergy[:take_mean_to])
	meanelectronKe = np.mean(electronKe[:take_mean_to])
	meanmajKe = np.mean(majKe[:take_mean_to])
	meanminKe = np.mean(minKe[:take_mean_to])

#	print('Plotting energies...')
#	
#	plt.plot(times[startfrom:]*multiply_by_time, (Exenergy[startfrom:]-meanenergyex)*multiply_by_electric_field,'b',label = "$E_{x}$")
#	#plt.plot(times[startfrom:]*multiply_by_time, (Eyenergy[startfrom:]-meanenergyey)*multiply_by_electric_field,'m',label = "$E_{y}$")
#	#plt.plot(times[startfrom:]*multiply_by_time, (Ezenergy[startfrom:]-meanenergyez)*multiply_by_electric_field,'y',label = "$E_{z}$")
#	#plt.plot(times[startfrom:]*multiply_by_time, (Bxenergy[startfrom:]-meanenergybx)*multiply_by_magnetic_field, label = "$B_{x}$")
#	plt.plot(times[startfrom:]*multiply_by_time, (Byenergy[startfrom:]-meanenergyby)*multiply_by_magnetic_field, 'cyan' ,label = "$B_{y}$")
#	plt.plot(times[startfrom:]*multiply_by_time, (Bzenergy[startfrom:]-meanenergybz)*multiply_by_magnetic_field,'g', label = "$B_{z}$")

#	plt.plot(times[startfrom:]*multiply_by_time, (majKe[startfrom:]-meanmajKe),'r', label = "Maj Ion KE")
#	if(minority): #if have a species named "min_ions"
#		plt.plot(times[startfrom:]*multiply_by_time, (minKe[startfrom:]-meanminKe),'cyan', label = "Min Ion KE") #"$\\alpha$")
#	# if (pic): #if have electrons
#	# 	plt.plot(times[startfrom:]*multiply_by_time, (electronKe[startfrom:]-meanelectronKe), 'k', label = "$e^{-}$")


#	if(labels): #if labels=true is given as input		
#		plt.xlabel('Time' + r' $[\tau_{cD}]$',fontsize=18)
#		plt.ylabel('Change in Energy Density' + r' $[' + 'J/m^{3}' + ']$',fontsize=18) 

#	if (leg=='on'):
#		plt.legend(loc = "best", fontsize=15)

#	#plt.tick_params(axis='y', which='major', labelsize=ticksize)
#	#xticken = np.linspace(0,10,11) ; plt.xticks(xticken, fontsize = ticksize) #assumes 10 gyroperiods
#	#plt.tick_params(axis='y', which='major', labelsize=10)
#	#plt.xlim(xticken[0], xticken[-1])
#	plt.tick_params(direction='out', pad=5)
#	#del Exenergy,Eyenergy,Ezenergy,Byenergy,Bzenergy,majKe,minKe,electronKe, times
#	plt.savefig('energy_dens.jpeg')
#	print('Figure saved here [{}].'.format(os.getcwd()))
#	dumpfiles(electronKe,'Electron_KE')
	if maj2:
		return Exenergy, Byenergy, Bzenergy, majKe, maj2Ke, minKe, electronKe#,# Eyenergy, Ezenergy, Bxenergy
	else:
		return Exenergy, Byenergy, Bzenergy, majKe, minKe, electronKe#,# Eyenergy, Ezenergy, Bxenergy
	
	# del Exenergy,Bzenergy,majKe,minKe,electronKe, times
	# use a "plt.show()" or "plt.savefig(...)" after this function.

# Gets the total kinetic energy density to be used in plotenergies()
def getTotalKineticEnergyDen(d,species):
	# Works in nD. Ignore any reference to "1D", such functions are actually general.
	# Means are computed from flattened array.

	#"Derived_EkBar" gives the KE per grid cell avergaed over all particles. In this function I then multiply by the density of particles to get the total
	#EK density. Ieally, would have KE for each particle, sum, then divide by "volume". Can calculate this fromn EPOCH momenta output 
	#(see function in "possibly_useful_funcs.py", needs modifying), but uses a lot of disc space and RAM hungry. Will probably have to do this for publication though...
	try:
		ek = getQuantity1d(d, "Derived_EkBar_"+species)
	except:
		ek = getQuantity1d(d, "Derived_Average_Particle_Energy_"+species)

	den = getQuantity1d(d, "Derived_Number_Density_"+species)
	mean_energy_density = np.mean(ek*den)

	return mean_energy_density


## Dump pkl files
def dumpfiles(array, quant):
	print('Pickling '+quant+'...')
	with open(quant+'.pkl', 'wb') as f:
		pickle.dump(array,f)

## Read pkl files
def read_pkl(quant):
	print('Loading '+quant+'...')
	with open(quant+'.pkl', 'rb') as f:
		array = pickle.load(f)
	print('Done.')
	# automatically closed when loaded due to "with" statement
	return array

## Scans dir for pkl files with "name" and returns boolean if it's in dir
def scan_pkl(names):
	# could implement a removal of "name" based off of returned boolean array: 
	# https://stackoverflow.com/questions/14537223/remove-items-from-a-list-using-a-boolean-array
	# names can be a single str or an array of files to load
	try: #check size of array
		np.shape(names)[0] 
		names=names
	except:
		names=[names]
	indir=[] # boolean array
	for name in names:
		try:
			with open(name+'.pkl', 'rb') as f: dummy = True # if can open then its in the dir
			indir.append(True)
		except:
			indir.append(False)
	if False in indir: #pkl file not in dir therefore need to run check again
		return False
	else: # all pkl files given in dir so can continue
		return True

## Calculates the cold wave modes for a given plasma (LH, UH, Omode, Xmode, Light, Cyclotron harmonics, CA) # doesnt actually calculate FAW, this is done separately
def ColdWaveModes(ax,omega_lst,va,wnorm,OMODE=True,FAW=True,UH=True,LH=True,CA=True,CYC=True,LGHT=True):
	wpi, wpe, wci, wce = omega_lst
	w = wnorm*np.linspace(0,50,10000) # normalised in units of wci and va ## TODO; change this to be more general
	k = (wnorm/va)*np.linspace(0,200,10000)

	## O-mode
	if OMODE:
		wom = np.sqrt(wpe**2+(const.c**2)*(k**2))
		plt.plot(k*va/wnorm,wom/wnorm,color='k',linestyle='--')#,label='O-mode')

	## Upper-hybrid
	if UH:
		wuh = np.sqrt(wce**2+wpe**2)
		plt.axhline(wuh/wnorm,color='k',linestyle='-.')#,label='UH')

	## Lower-hybrid
	if LH:
		wlh = np.sqrt(((wpi**2)+(wci**2))/(1+((wpe**2)/(wce**2))))
		plt.axhline(wlh/wnorm,color='k',linestyle='--')#,label='LH') ## handled in the FAW calculation

	## Compressional Alfven
	if CA:
		wA = k*va
		plt.plot(k*va/wnorm,wA/wnorm,color='k',linestyle='--')#,alpha=0.5,label='Comp Alf')

	## Light travelling in a vacuum
	if LGHT:
		w = k*const.c
		plt.plot(k*va/wnorm,w/wnorm,linestyle='--',color='k')#label='light')
	
	## Cyclotron Harmonics
	if CYC:
		if wnorm == wci:
			for i in range(0,16):
				plt.axhline(i,color='k',linestyle=':',alpha=0.5)
		elif wnorm == wce:
			for i in range(0,4):
				plt.axhline(i,color='k',linestyle=':',alpha=0.5)
		else:
			None
	return ax

## Returns all normalisation values used later (frequencies and lengths)
def getFreq_Wavenum(file0,maj_species,min_species,Zmaj,Zmin):
	wci  = getCyclotronFreq(file0,maj_species,Z=Zmaj)
	wce  = getCyclotronFreq(file0,'Electrons',Z=1)
	va   = getAlfvenVel(file0)
	lambdaD= getDebyeLength(file0,'Electrons')
	wpe	 = getPlasmaFreq(file0,species='Electrons')
	wpi  = getPlasmaFreq(file0,species=maj_species)

	return wci,wce,va,lambdaD,wpe,wpi
	
## Find the growth rates of the MCI in its linear phase based off of drift and spread velocities
def growth_rate_man(minions, majions, theta, b, u, vd, vr, kall, omegaall):
	# vr is para drift
	# vd is perp drift
	# emin is beam energy in eV

	# THIS GROWTH RATE CORRESPONDS TO EQUATIONS (8)-(10) OF MCCLEMENTS ET AL. POP 3 (2) 1996
	# I DON'T USE EQ. (11) TO GET THE FREQUENCY, RATHER I USE THE COLD PALSMA DISPERSION RELATION
	
	Zmin = getChargeNum(minions)
	Zmaj = getChargeNum(majions)
	wcycb = getCyclotronFreq(b,minions,Zmin) # cyc freq for beam ions
	wcyci = getCyclotronFreq(b,majions,Zmaj) # cyc freq for bulk ions

	wpb = getPlasmaFreq(b, minions) #square of beam ion plasma frequency, z=1 for tritons
	wpi = getPlasmaFreq(b, majions) #square of bulk ion plasma frequency, z=1 for deuterons
	va = getAlfvenVel(b)

	theta = theta*(const.PI/180.0) #radians
	gammas = np.zeros(omegaall.shape[0]) #growth rates

	for i in range(0, omegaall.shape[0]):
		l = round(omegaall[i]/wcycb) #l closest to the omega
		k = kall[i]
		kpara = kall[i]*np.cos(theta)
		kperp = kall[i]*np.sin(theta)

		Npara = (kpara*va)/omegaall[i]
		Nperp = (kperp*va)/omegaall[i]

		eetal = (omegaall[i] - kpara*vd - l*wcycb)/(kpara*vr) # vd=0
		za = kperp*u/wcycb
		############## M_l ###############################################################
		mlterm1 = 2.0*l*(omegaall[i]/wcyci)*((spec.jvp(l,za)**2) + ((1.0/za**2)*(l**2 - za**2)*spec.jv(l,za)**2))
		mlterm2 = -2.0*((omegaall[i]**2 - wcyci**2)/wcyci**2)*((spec.jv(l,za)*spec.jvp(l,za))/za)*((l**2)*Nperp**2 - (za**2 - 2.0*(l**2))*(Npara**2))
		mlterm3 = (2.0*spec.jv(l,za)*spec.jvp(l,za)/za)*(za**2 - 2.0*(l**2))
		ml = mlterm1 + mlterm2 + mlterm3
		##################################################################################
						#
						#
		############# N_l #################################################################
		nlterm1 = -2.0*l*(omegaall[i]/wcyci)*(spec.jv(l,za)*spec.jvp(l,za)/za)
		nlterm2pre = (omegaall[i]**2 - wcyci**2)/wcyci**2
		nlterm2 = (Npara**2)*((l*spec.jv(l,za)/za)**2 + spec.jvp(l,za)**2) + (Nperp*l*spec.jv(l,za)/za)**2
		nlterm3 = (l*spec.jv(l,za)/za)**2 + spec.jvp(l,za)**2
		nl = nlterm1 + nlterm2pre*nlterm2 + nlterm3
		#################################################################################		
						#
						#
		########## Gamma ################################################################
		pre = (((wpb*(wcyci**2))/wpi)**2)*((const.PI**0.5)/(2.0*omegaall[i]))*np.exp(-(eetal**2))
		term1 = 1.0/((wcyci + (omegaall[i] - wcyci)*(Npara**2))*(wcyci - (omegaall[i] + wcyci)*(Npara**2)))
		term2 = ((l*wcycb*ml)/(kpara*vr)) - (2.0*eetal*nl)*((u/vr)**2)
		gammas[i] = pre*term1*term2
		#print gammas[i]
	
	####### Want gamma > 1 only ###########
	posomega = [] ; posgamma = []	

	for i in range(0,gammas.shape[0]):
		if (gammas[i] >= 0):
			posomega.append(omegaall[i]/wcycb)
			posgamma.append(gammas[i]/wcycb)

	return posomega, posgamma

## Find and plot the distribution function of the species specified
def dist_fn(index_list,xyz,species):
	# Returns the distribution function plotted as a KDE from the binned data of the particles momenta, converted to velocity then normalised in units of the species thermal velocity
	# index_list :: same input as usual
	# xyz :: dimension 'x' , 'y' or 'z' depending on which component you want to measure
	# species :: name of the species you want to measure, typically take the minority ion
	# returns a 2d array of velocities through time and space for each file which contains the correct information
	if xyz == 'x':
		mom = 'Particles_Px_'+species
		vel = r'$v_x$'
		vname = 'Vx_'
	elif xyz == 'y':
		mom = 'Particles_Py_'+species
		vel = r'$v_y$'
		vname = 'Vy_'
	elif xyz == 'z':
		mom = 'Particles_Pz_'+species
		vel = r'$v_z$'
		vname = 'Vz_'
	else:
		print('# ERROR # : Invalid velocity option.')
		raise SystemExit

	v = []
	vth = getThermalVelocity(0,species) # assumes 0th file has temperature
	mass = getMass(species)
	for i in index_list:
		d = sdfread(i)
		try:
			v.append((getQuantity1d(d,mom)/mass)/vth)
		except:
			continue
	del d

	nval = 1000
#	bins = 750
	fig,ax=plt.subplots(figsize=(8,5))
	print('Plotting KDEs...')
	for v_arr in v:
		print('vel file {}...'.format(i))
		kde = stats.gaussian_kde(v_arr)
		vv = np.linspace(min(v_arr),max(v_arr),nval) # nval is just the length of the plotted array in v and f(v)
		#	ax.hist(v_arr,bins=bins,density=True)    # change number of bins accordingly, density=True normalises the histogram
		ax.plot(vv,kde(vv))

	ax.set_xlabel(vel+r'$/v_{th}$',fontsize=18)
	ax.set_ylabel(r'$f(\mathbf{v})$',fontsize=18)
	fig.savefig(vname+species+'_dist_fn.jpeg',bbox_inches='tight') # plots all KDEs on the same axis
	
	dumpfiles(v,vname+species)
	return v

## Bicoherence plot of the 
def bicoherence_theory_taverage(field, klim, ftspacing, ftwidth, smooth=False):

	#ftspacing is the number of t-steps to advance when compouting successive FT's, this will usually be a fraction of the gyro period
	#ft width is the width of each FT in time	

	s = int(ftwidth)
	numerator = np.zeros((s, s), dtype='complex') 
	denominator = np.zeros((s, s)) 
	indneg = np.arange(0,s/2 + 1) - (s/2)
	indpos = np.arange(1,(s/2)+1)
	nx = field.shape[1]

	print("About to start loop. I have "+str((((nx-ftwidth)/ftspacing)+1))+" iterations to do.")
	for i in range(0, int(((nx-ftwidth)/ftspacing)+1)):
		field_portion = field[:,i*ftspacing:ftwidth + i*ftspacing] 
		#trans=np.fft.fftshift(np.fft.fft(field_portion, axis=1), axes=1)
		transchop=np.fft.fftshift(np.fft.fft(field_portion, axis=1), axes=1)#[:,s/2:]
		#transchop[:,:transchop.shape[1]/2] = trans[:,:trans.shape[1]/2-1] #chopping blue line out. Could write a little function to compact this
		#transchop[:,transchop.shape[1]/2:] = trans[:,(trans.shape[1]/2)+1:]

		bicoh = np.zeros([s, s], dtype='complex')
		
		for k in range(0, int(s/2)): # horizontal
			for j in range(k, int(s/2)): #vertical
				if (indneg[j] + indneg[k] < -s/2 ):
					bicoh[j,k] = 0.0
				else:
					bicoh[j,k] = np.mean(transchop[:,j]*transchop[:,k]*np.conj(transchop[:,int(j+k-(s/2))]), axis=0)
					#bicoh[j,k] = 1.0
		for k in range(0, int(s/2)):
			for j in range(k, int(s/2)):
				if (indpos[j] + indpos[k] > s/2):
					bicoh[int(j+(s/2)),int(k+(s/2))] = 0.0
					#print indpos[j] , indpos[k], j+(s/2)
				else:
					#print j, indpos[j], k, indpos[k], j+k, indpos[j+k]
					bicoh[int(j+(s/2)),int(k+(s/2))] = np.mean(transchop[:,int(j+(s/2))]*transchop[:,int(k+(s/2))]*np.conj(transchop[:,int(j+k+(s/2))]), axis=0)
					#bicoh[j+(s/2),k+(s/2)] = 1.0
		
		for j in range(0, int(transchop.shape[1]/2)):
			for k in range(0, int(transchop.shape[1]/2)):
				tot = int(indneg[j] + indpos[k] + s/2)
				bicoh[int(j+(s/2)),k] = np.mean(transchop[:,int(j+(s/2))]*transchop[:,k]*np.conj(transchop[:,tot]), axis=0)
				#bicoh[j+(s/2),k] = float(tot)
		
		numerator = numerator + bicoh
		denominator = denominator + np.abs(bicoh)
		if i%100 == 0: print(i)

		bi = np.abs(numerator)/denominator
	bi = np.nan_to_num(bi)
	del numerator, denominator
	#unit = np.ones(bi.shape)
	#tri = -np.tril(unit)
	#nan = np.sqrt(tri*unit)*0.0
	#bi = bi+nan

	###########################################
	#cutoff = 0.0
	#for i in range(0, bi.shape[0]):
	#	for j in range(0, bi.shape[1]):
	#		if (bi[i,j] < cutoff):
	#			bi[i,j] = cutoff
	###########################################

	bi[int(s/2),:] = 0.0 ; 	bi[:,int(s/2)] = 0.0

	if (smooth):
		bi = gaussian_filter(bi, sigma=1)

	dumpfiles(bi, 'bicoherence_mat')
	ax = plt.gca()
	l1 = Line2D([-klim,0],[0,0], linestyle="dashed", color="white", linewidth="2") ; ax.add_line(l1)
	l2 = Line2D([0,0],[0,klim], linestyle="dashed", color="white", linewidth="2") ; ax.add_line(l2)
	l3 = Line2D([0,-klim],[0,klim], linestyle="dashed", color="white", linewidth="2") ; ax.add_line(l3)     

	extent = [-klim, klim, -klim, klim]
	im = plt.imshow(bi, interpolation='nearest', origin='lower',aspect='auto', extent=extent, clim = (0.0,0.2))	
	del bi	
	plt.xlabel('$k_{1}$' + r' $[\omega_{cp}/V_{A}]$', fontsize=15)
	plt.ylabel('$k_{2}$' + r' $[\omega_{cp}/V_{A}]$', fontsize=15)
	plt.colorbar()
#	plt.ylim(-15.0,30.0) ; plt.xlim(-30.0,15.0)
	plt.show()

	
