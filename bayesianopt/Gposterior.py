import numpy as np 
from covariance import Covariance_func as cov
import torch 

class Posterior_func:

	def posterior(X_test, X_train, Y_train, param):
	    k = cov.SEkernel(X_train, X_train, param)
	    k_s = cov.SEkernel(X_test, X_train, param)
	    k_inv = np.linalg.inv(k)
	    k_ss = cov.SEkernel(X_test, X_test, param)
	    
	    mu = k_s.dot(k_inv).dot(Y_train)
	    
	    post_cov = k_ss - np.dot(k_s, np.dot(k_inv,k_s.T))
	    return mu, post_cov

	def posterior_torch(X_test, X_train, Y_train, param):
		
		k = cov.SEkernel_torch(X_train, X_train, param)
		k_s = cov.SEkernel_torch(X_test, X_train, param)
		k_inv = torch.inverse(k)
		k_ss = cov.SEkernel_torch(X_test, X_test, param)

		mu = torch.mm(k_s, torch.mm(k_inv, Y_train))

		post_cov = k_ss - torch.mm(k_s, torch.mm(k_inv,k_s.t()))
		return mu, post_cov
