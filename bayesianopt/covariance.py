import numpy as np
import torch

class Covariance_func:
	
	def SEkernel(a, b, param):
	    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
	    return np.exp(-.5 * (1/param) * sqdist) #+ 1e-8*np.eye(len)


	def SEkernel_torch(a, b, param):
		sqdist = torch.sum(a**2, 1).reshape(-1,1) + torch.sum(b**2, 1) - 2*torch.mm(a, b.t()) 

		return torch.exp(-0.5*(1/param)*sqdist)  
