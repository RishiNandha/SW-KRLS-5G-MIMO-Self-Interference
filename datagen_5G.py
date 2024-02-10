import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

def generate_random_signal(power_dB, duration, sampling_rate):
    power = 10**(power_dB / 20)
    signal = np.random.normal(0, power, int(duration * sampling_rate))
    return signal

power_dB = -20 
duration = 0.02  
sampling_rate = 1000000
split = 7700


t = np.arange(0,0.02,1/1000000)

RX_ideal = np.sin(5002*t + 0.4*np.cos(1000*t))
noise = generate_random_signal(power_dB-10, duration, sampling_rate) 
TX = np.sin(10132*t + 0.02*np.cos(t))
f_RX = 9*10**(6)

f_TX1 = np.array([1.7*10**(6),])+np.zeros(split)
f_TX2 = np.array([4.5*10**(6),])+np.zeros(20000 - split)
f_TX = np.concatenate((f_TX1,f_TX2))

TxH2_TX1 = np.array([0.06,])+np.zeros(split)
TxH2_TX2 = np.array([0.03,])+np.zeros(20000 - split)
TxH2 = np.concatenate((TxH2_TX1,TxH2_TX2))

IMD2_TX1 = np.array([0.1,])+np.zeros(split)
IMD2_TX2 = np.array([0.23,])+np.zeros(20000 - split)
IMD2 = np.concatenate((IMD2_TX1,IMD2_TX2))

RX = np.zeros(20000)
RX = RX_ideal+noise+TxH2*TX*TX*np.cos(2*np.pi*(2*f_TX-f_RX)*t) + IMD2*np.power(TX*np.cos(4*np.pi*t*f_TX),2) 

plt.plot(t,RX)
plt.show()

def kappa(x1,x2):
	return (x1.T @ x2 + 1)*(x1.T @ x2 + 1)

def batch(x,n):
	M=1000
	return x[n:n+M].reshape(-1,1)

def KRLS_SW(X,Y):

	M = 1000

	x = X.copy()
	y = Y.copy()
	ypred=np.zeros(20000)
	e = np.zeros(20000)
	lamb=0.95
	
	P = np.eye(M)/(1+lamb*M)
	
	for n in range(1000, 20000):
		Pdash = P[1:,1:] - (P[1:,0] @ P[0, 1:])/P[0,0]
		k = np.array([kappa(batch(x,n-M+1+i), batch(x,n-M)) for i in range(M-1)]).reshape(-1,1)
		a = Pdash @ k
		g = 1/(kappa(batch(x,n-1),batch(x,n-1))+lamb*M + k.T @ a)
		P = np.block([[Pdash + (a@a.T)*g, -a*g],[-a.T*g, g]])
		alpha = P@y[n:n+M]
		ypred[n:n+1] =  np.array([kappa(batch(x,n-M+i), batch(x,n-M)) for i in range(M)]).reshape(-1,1).T @ alpha
		print(ypred[n])
	e = y - ypred
	return ypred, e

KRLS_SW(TX, RX)
plt.semilogy(t, e^2)
plt.show()
'''D, U = torch.reshape(torch.tensor([]),(0,400000)),torch.reshape(torch.tensor([]),(0,400009))
for i in range(100):
	d,u = pickle.load(open("expt3_data/expt3_"+str(i)+".bin","rb"))
	print(d.shape, u.shape, d, u)
	U = torch.cat((U, torch.reshape(torch.from_numpy(u),(1,400009))),0)
	D = torch.cat((D, torch.reshape(torch.from_numpy(d),(1,400000))),0)
	print(u)
	print(d)

pickle.dump((D,U), open("Expt3_data.bin","wb"))'''