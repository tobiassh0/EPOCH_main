import numpy, scipy.io
import pickle


def read_pkl(quant):
	print('Loading '+quant+'...')
	with open(quant+'.pkl', 'rb') as f:
		array = pickle.load(f)
	print('Done.')
	# automatically closed when loaded due to "with" statement
	return array

data = read_pkl('fieldmatrix_Magnetic_Field_Bz')
scipy.io.savemat('fieldmat.mat', mdict={'data': data})


