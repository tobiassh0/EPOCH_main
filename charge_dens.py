import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from list_new import *
import my_constants as const


def update_particle(px,py,pz,dt,pos=[0,0,0]):
	return pos[0]+px*dt, pos[1]+py*dt, pos[2]+pz*dt
#	x,y = pos
#	x = x + px*dt
#	y = y + py*dt
#	return x, y


def particle(px_arr,py_arr,pz_arr,t_arr,inipos=[0,0,0]):
	dt = (t_arr[-1]-t_arr[0])/len(t_arr) # average dt
	x0, y0, z0 = inipos
	if len(px_arr) != len(py_arr) or len(px_arr) != len(t_arr):
		print('## ERROR ## : Px, Py and time arrays are not same size')
		raise SystemExit
	else:
		x,y,z = [], [], []
		x.append(x0)
		y.append(y0)
		z.append(z0)
		for i in range(len(t_arr)):
			x_t, y_t, z_t = update_particle(px_arr[i],py_arr[i],pz_arr[i],dt,pos=[x[-1],y[-1],z[-1]])
			x.append(x_t)
			y.append(y_t)
			z.append(z_t)
	return x, y, z, dt
	
sim_lst = ['0006qp','7_3rb','He3_rb','He3_rb2','JET_26148','JET_26148_long','JET_26148_short']
sim_lst = ['T_1_0','DT_1_2','JET_26148','p_1_0']
sim_lst = ['resolve_e_gyro']


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
fig, ax = plt.subplots(figsize=(8,6))
species_lst = ['Deuterons','Tritons']

## current or momentum plots
current = False
mom = not current
	
for species in species_lst:
	for sim in sim_lst:	
	#	sim_loc = getSimulation('/home/space/phrmsf/Documents/EPOCH/rb_epoch/epoch/epoch1d/'+sim)
		sim_loc = getSimulation('/storage/space2/phrmsf/'+sim)
		ind_lst = list_sdf(sim_loc)
		L = getGridlen(sdfread(0))
		d = sdfread(0)
		keys = d.__dict__.keys()  ##returns all the names of quantities in the simulation sdf file
		for k in keys: print(k)
		frac = 1
		wc_alpha = getCyclotronFreq(d,'Alphas',getChargeNum('Alphas'))	
		tc_alpha = 2*const.PI/wc_alpha
		t_ind=[]
		jx=[] ; jy=[] ; jz=[]
		px=[] ; py=[] ; pz=[]
		times = read_pkl('times')
		for i in range(len(ind_lst)):
			d = sdfread(i)
			try:
				if current:
					jx.append(getMeanquantity(d,'Current_Jx')/1E6)
					jy.append(getMeanquantity(d,'Current_Jy')/1E6)
					jz.append(getMeanquantity(d,'Current_Jz')/1E6)
				if mom:
					px.append(getMeanquantity(d,'Particles_Px_'+species))
					py.append(getMeanquantity(d,'Particles_Py_'+species))
					pz.append(getMeanquantity(d,'Particles_Pz_'+species))
				t_ind.append(i)
				print(i)
			except:
				continue
		t = [times[i]/tc_alpha for i in t_ind] # create array of times as unit sof tc_alpha from the t_ind list of available points
		print(len(t),len(jx),len(jy),len(jz))
		if current:
			ax.plot(t,jx,color='r',linestyle='--')
			ax.plot(t,jy,color='b',linestyle='-')
			ax.plot(t,jz,color='g',linestyle=':')
		if mom:
			ax.plot(t,px,color='r',linestyle='--')
			ax.plot(t,py,color='b',linestyle='-')
			ax.plot(t,pz,color='g',linestyle=':')
		
	print((t[1]-t[0]))
	t=np.array(t)*tc_alpha

	if current:
		try:
			jx = read_pkl('Jx') ; jy = read_pkl('Jy') ; jz = read_pkl('Jz') 
		except:
			dumpfiles(jx,'Jx')
			dumpfiles(jy,'Jy')
			dumpfiles(jz,'Jz')
		ax.legend([r'$J_x$',r'$J_y$',r'$J_z$'],loc='lower center')
		ax.set_xlabel(r'$t/\tau_{c\alpha}$',fontsize=18)
		ax.set_ylabel('Current density [MA/'+r'$m^{-2}$'+']',fontsize=18)
		fig.savefig('current_density.jpeg',bbox_inches='tight')
		plt.cla()
		
	if mom:
		try:
			px = read_pkl('Px_'+species) ; py = read_pkl('Py_'+species) ; pz = read_pkl('Pz_'+species) 
		except:
			dumpfiles(px,'Px_'+species)
			dumpfiles(py,'Py_'+species)
			dumpfiles(pz,'Pz_'+species)
		ax.legend([r'$P_x$',r'$P_y$',r'$P_z$'],loc='best')
		ax.set_title(species)
		ax.set_xlabel(r'$t/\tau_{c\alpha}$',fontsize=18)
		ax.set_ylabel('Momentum ['+r'$kg ms^{-1}}$'+']',fontsize=18)
		fig.savefig('momentum_evolution_'+species+'.jpeg',bbox_inches='tight')
		plt.cla()


#	mass = getMass(species)
#	jx, jy, jz = np.array(jx)/mass, np.array(jy)/mass, np.array(jz)/mass
#	jx, jy, jz = np.array(jx), np.array(jy), np.array(jz)

#	x_pos, y_pos, z_pos, dt = particle(px_arr=jx,py_arr=jy,pz_arr=jz,t_arr=t)
	
#	plt.plot(x_pos/L,y_pos/L,label=species)
#	plt.plot(t/tc_alpha,x_pos[1:]/L,label=species+' s_x')
#	plt.plot(t/tc_alpha,y_pos[1:]/L,label=species+' s_y',linestyle='--')
#	ax.plot(x_pos/L,y_pos/L,z_pos/L,label=species)

#ax.set_xlabel(r'$s_x/L$',fontsize=18)
#ax.set_ylabel(r'$s_y/L$',fontsize=18)
#ax.set_zlabel(r'$s_z/L$',fontsize=18)
#plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
#plt.ylabel(r'$s/L$',fontsize=18)
#plt.xlabel(r'$t/\tau_{c\alpha}$',fontsize=18)
#plt.legend(loc='best')
#plt.show()


#	cdens = np.zeros(n)
#	species = ['Alphas','Deuterons','Helium3','Electrons']
##	cdens = getQuantity1d(d,'Derived_Charge_Density')
#	for i in range(len(species)):
#		denss = getQuantity1d(d,'Derived_Charge_Density_'+species[i])
#		cdens += denss
#		x = np.linspace(0,n,n)
#		plt.plot(x,denss)
#	cavg = np.mean(cdens)
##	plt.axhline(cavg,color='k')
#	print(sim, cavg)

#plt.plot(x,cdens)
#plt.show()

#	plt.plot(x[::frac],cdens[::frac],label=species[i])
#plt.legend(loc='best')
#plt.show()

