import numpy as np 
import pickle as p 

d,u = p.load(open('expt1.bin','rb'))
u = np.concatenate((np.zeros((300,3)),u), axis = 1)

e = np.zeros((300,600))

Lambda = 0.995

for exp in range(300):

	P = np.array([[1000,0,0,0],
	[0,1000,0,0],
	[0,0,1000,0],
	[0,0,0,1000]
	])

	w = np.array([0.5,0.5,0.5,0.5]).reshape((-1,1))

	d_exp = d[exp]
	u_exp = u[exp]

	for i in range(600):

		u_mat = np.array(u_exp[i:i+4]).reshape((1,4))

		d_mat = d_exp[i].reshape((1,1))

		e[exp,i] = (d_mat - (u_mat @ w))[0,0]

		A = P @ np.transpose(u_mat)

		P = (1/Lambda)*(P - (A @(u_mat @ P))/(Lambda + (u_mat @ A)))

		w = w + P@(np.transpose(u_mat) @ (d_mat - (u_mat @ w)))	

ensemble_e = np.sum((e*e), axis = 0, keepdims=True)/300

with open('RLS_expt1.bin', 'wb') as f:
	p.dump(ensemble_e, f)
