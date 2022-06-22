import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from list_new import *
import my_constants as const
from power import *

def dist_fn(index_list,xyz,species):
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
	hr = []
	for i in index_list:
		d = sdfread(i)
		try:
			v.append((getQuantity1d(d,mom)/mass))
			hr.append(i)
		except:
			continue
	del d
	print(hr)

#	i=0
#	nval = 1000
#	fig,ax=plt.subplots(figsize=(8,5))
#	print('Plotting KDEs...')
#	for v_arr in v:
#		print('vel file {}...'.format(i))
#		kde = stats.gaussian_kde(v_arr)
#		vv = np.linspace(min(v_arr),max(v_arr),nval) # nval is just given as length of array
#		#	ax.hist(v_arr,bins=750,density=True)    # change number of bins accordingly, density=True normalises the histogram
#		ax.plot(vv,kde(vv))
#		i += 1
	
#	ax.set_xlabel(vel+r'$/v_{th}$',fontsize=18)
#	ax.set_ylabel(r'$f(\mathbf{v})$',fontsize=18)
#	fig.savefig(vname+species+'_dist_fn.jpg',bbox_inches='tight') # plots all KDEs on the same axis
	dumpfiles(v,vname+species)
	return v # stored as already normalised to thermal velocity


#sim = 'JET_26148'
#home = os.getcwd()
#sim_loc = getSimulation('/home/space/phrmsf/Documents/EPOCH/rb_epoch/epoch/epoch1d/'+sim)
##sim_loc = getSimulation('/storage/space2/phrmsf/JET_26148')
#ind_lst = list_sdf(sim_loc)

#ndata = len(ind_lst)
#electronKe = np.zeros(ndata,dtype='float')
#for i in range(0, ndata):
#	d = sdfread(i)
#	electronKe[i] = getTotalKineticEnergyDen(d,species = 'Electrons')
#	if np.around(100*(i/ndata),2)%5==0: print('file: {}/{}, Progress: {}%'.format(i,ndata,np.around(100*i/ndata,2))) ##print every 5% completion rather than EVERY step
#dumpfiles(electronKe,'Electrons_KE')

#r_LHe3 = getLarmorRadius(sdfread(0),'Helium3')
#r_LD = getLarmorRadius(sdfread(0),'Deuterons')
#print(r_LD, r_LHe3)

##########################============================###############################
##########################============================###############################

#	d = sdfread(0)
#	keys = d.__dict__.keys() # returns all the names of quantities in the simulation sdf file
#	for k in keys: print(k)

##########################============================###############################
##########################============================###############################

#xyz = ['x','y','z']
#species = ['Alphas']
#for spec in species:
#	for dim in xyz:
#		dist_fn(ind_lst,dim,spec)

#species = ['Deuterons','Helium3','Alphas']
#species = ['Deuterons','Helium3']
hist, scatter, fv_KDE = False, False, False
colors = ['r','b','g'] # same len as species
colors = ['b']
alphas = [1,0.5,0.25] # smae len as species

if hist:
	fig, axs = plt.subplots(1,3,figsize=(18,5),sharey=True)
	freq = 1000
	bins = 200
	vname = [r'$v_x$',r'$v_y$',r'$v_z$']
	for j in range(len(species)):
		if species[j] == 'Deuterons':
			vth_D = getThermalVelocity(sdfread(0),species[j])
		for i in range(len(xyz)):
			v = read_pkl('V'+xyz[i]+'_'+species[j])
			ax = axs[i]
			ax.hist(v[0][::freq]*(1/vth_D),bins=bins,density=True,label=spec,color=colors[j],alpha=alphas[j]) # normalise all to the v_th of Deuterons
			ax.axvline(np.mean(v[0]*(1/vth_D)),linestyle='--')
			print(species[j],xyz[i],'vmean ',str(np.mean(v[0]*(1/vth_D))),'skew ',stats.moment(v[0]/vth_D,moment=3))
			if species[j] == 'Deuterons': # assumes at least one of the species will be named 'Deuterons'
				ax.set_title(vname[i],fontsize=20)
				if i == 1:
					ax.set_xlabel(r'$v/v_{th_D}$',fontsize=18)		
				elif i == 0:
					ax.set_ylabel(r'$f(\mathbf{v})$',fontsize=18)
			else:
				None
	fig_name = 'V_histcomp.jpeg' # save this figure later when all species are done
	fig.savefig(fig_name,bbox_inches='tight')
	plt.clf()

if scatter:
	freq = 1000
	for j in range(len(species)):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		vx = read_pkl('Vx_'+species[j]); vy = read_pkl('Vy_'+species[j]); vz = read_pkl('Vz_'+species[j])
		vth = getThermalVelocity(0,species[j])
		ax.scatter(vx[0][::freq]/vth,vy[0][::freq]/vth,vz[0][::freq]/vth)
		ax.set_xlabel(r'$v_x/v_{th}$',fontsize=18)
		ax.set_ylabel(r'$v_y/v_{th}$',fontsize=18)
		ax.set_zlabel(r'$v_z/v_{th}$',fontsize=18)
		ax.set_title(species[j],fontsize=20)
		fig.savefig('VxVyVz_dist_'+species[j]+'.jpeg')
		plt.clf()

if fv_KDE:
	fig,ax=plt.subplots(figsize=(8,5))
	print('Plotting KDEs...')
	nval = 1000
	v = read_pkl('Vx_'+species[0])
	ind = 0, int(np.shape(v)[0]/2), int(np.shape(v)[0]) 
	v = np.zeros(np.shape(v)[1])
	print(np.shape(v), ind)
	for j in range(len(species)):
		for t in range(len(ind)-1):
			print('vel file {}...'.format(ind[t]))
			for dim in xyz:
				v += np.array(read_pkl('V'+dim+'_'+species[j])[ind[t]])
			v=v/len(xyz)
			kde = stats.gaussian_kde(v)
			vv = np.linspace(min(v),max(v),nval) # nval is just given as a length of number of points to calculate
			ax.plot(vv,kde(vv))

	fig_name = 'fv_KDE.jpeg'
	fig.savefig(fig_name,bbox_inches='tight')
	plt.clf()


sims = ['rb1','rb2','rb3','rb4','rb5','rb6']
i=0
species = 'Alphas'
freq = 1000
titles = [r'$-2,-2,-\pi/3$',r'$-2,-3,-\pi/3$',r'$-3,-2,-\pi/3$',r'$-2,-2,-\pi/6$',r'$-2,-3,-\pi/6$',r'$-3,-2,-\pi/6$']
fig = plt.figure()
for s in sims:
	print('here')
	sim_loc = getSimulation('/home/space/phrmsf/Documents/EPOCH/rb_epoch/epoch/epoch1d/'+s)
	ind_lst = list_sdf(sim_loc)
	vth = getThermalVelocity(sdfread(0),'Alphas')
	vA = getAlfvenVel(sdfread(0))
	print(i)
	i+=1
	ax = fig.add_subplot(2, 3, i, projection='3d')	
	ax.ticklabel_format(useOffset=False)
	vx = read_pkl('Vx_'+species); vy = read_pkl('Vy_'+species); vz = read_pkl('Vz_'+species)
	ax.scatter(vx[0][::freq]/vA,vy[0][::freq]/vA,vz[0][::freq]/vA)
	ax.set_xlabel(r'$v_x [v_{A}]$',fontsize=18)
	ax.set_ylabel(r'$v_y [v_{A}]$',fontsize=18)
	ax.set_zlabel(r'$v_z [v_{A}]$',fontsize=18)
	ax.set_title(titles[i-1],fontsize=18)

plt.show()

