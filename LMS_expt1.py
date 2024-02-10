import numpy as np 
import pickle as p 

# Each Row is a seperate experiment
d,u = p.load(open('expt1.bin','rb'))
u = np.concatenate((np.zeros((300,3)),u), axis = 1)

# Weights to right are the coefficients of lower powers of z^-1
w = np.zeros((300,4))

# Each row is a seperate experiement's error
e = np.zeros((300,600))

# Learning Rate
mu = 0.01

for i in range(600):
	# w is the same as w^H
	# u[:, i:i+4] is the same as u
	# dpred same as uw
	dpred = np.sum(w * u[:,i:i+4],axis=1,keepdims=True)
	e[:,i:i+1] = d[:,i:i+1] - dpred
	# so here u*e[:,i:i+4] does the same as (u^H)*e(i)
	# since numpy multiplication is pointwise multiplication
	w = w + mu*u[:,i:i+4]*e[:,i:i+1]

ensemble_e = np.sum((e*e), axis = 0, keepdims=True)/300

with open('LMS_expt1.bin', 'wb') as f:
	p.dump(ensemble_e, f)


from matplotlib import pyplot as plt
plt.plot(np.arange(0,600,1),ensemble_e.flatten())
plt.show()
print(w)