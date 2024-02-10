import numpy as np 
import pickle as p 

d,u = p.load(open('expt1.bin','rb'))
u = np.concatenate((np.zeros((300,3)),u), axis = 1)
w = np.zeros((300,4))
e = np.zeros((300,600))

mu = 0.2
epsilon = 0.001

for i in range(600):

	dpred = np.sum(w * u[:,i:i+4],axis=1,keepdims=True)

	e[:,i:i+1] = d[:,i:i+1] - dpred

	unorm = np.sum(u[:,i:i+4] * u[:,i:i+4],axis=1,keepdims=True)

	w = w + (mu*u[:,i:i+4]*e[:,i:i+1])/(epsilon + unorm)

ensemble_e = np.sum((e*e), axis = 0, keepdims=True)/300

with open('e-NLMS_expt1.bin', 'wb') as f:
	p.dump(ensemble_e, f)


from matplotlib import pyplot as plt
plt.plot(np.arange(0,600,1),ensemble_e.flatten())
plt.show()
print(w)