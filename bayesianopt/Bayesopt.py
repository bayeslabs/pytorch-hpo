import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats
import numpy as np 
import torch
from covariance import Covariance_func
from Gposterior import Posterior_func 
from aquisition import Aquisition_func

def func_lamda(x):
	return x*x#torch.sin(5*x)#x*x
func = func_lamda

class Bayesian:
	def __init__(self,func, num_hyperparameter, device, range_lamda):
		self.num_hyp = num_hyperparameter
		# self.budget  = budget 
		self.range = range_lamda 
		self.device = device
		self.obj_func = func 	
		#if lamda_train == None:
		self.lamda_train = (range_lamda[1] - range_lamda[0])*torch.rand(3, num_hyperparameter) + range_lamda[0]
		# else:
		# 	self.lamda_train = lamda_train.reshape(-1,1) 
		# 	self.lamda_train = torch.tensor(lamda_train, device = self.device)
		# else:
		# 	self.lamda_train = lamda_train
		self.f_lamda_train = self.obj_func(self.lamda_train)

	def steps(self):
		num = 100				#number of sampling points in the range
		param = 0.1
		z = torch.randn(num,10, device = self.device)

		# lamda_train = ((self.range[1] - self.range[0])*torch.rand(3, self.num_hyp) + self.range[0]).reshape(-1,1)
		lamda_test = torch.linspace(self.range[0],self.range[1], steps = num, device = self.device).reshape(-1,1)
		f_lamda_test = self.obj_func(lamda_test)
		
		f_dash = min(self.f_lamda_train )

		cov_matrix 	= Covariance_func.SEkernel_torch(lamda_test, lamda_test, param)
		mu_post, cov_post = Posterior_func.posterior_torch(lamda_test, self.lamda_train, self.f_lamda_train, param)
		f_post = mu_post + torch.mm(cov_post, z)
		f_post_mean = torch.mean(f_post, 1)

		aquis_arr, lamda_new = Aquisition_func.ExpectedImprovement_torch(f_post_mean, cov_post, f_dash)
		
		lamda_train_new = np.append(torch.Tensor.numpy(self.lamda_train).flatten(), torch.Tensor.numpy(lamda_test[lamda_new]))
		lamda_train_new = torch.tensor(lamda_train_new).reshape(-1,1)
		
		return lamda_train_new#, mu_post, cov_post

	def optim(self, N_batch):
		for iteration in range(N_batch):
			self.lamda_train = self.steps()
			self.f_lamda_train = self.obj_func(self.lamda_train)
		return self.lamda_train, self.f_lamda_train

# hyp = Bayesian(1, device='cpu', range_lamda=[-1,1])
# print(hyp.optim())
hyper_para = Bayesian(func,1, device='cpu', range_lamda=[-1,1])#, lamda_train=torch.tensor([0.1,0.2,0.4]))#torch.tensor([1,2,3]))
# print(Bayesian(func,1, device='cpu', range_lamda=[-1,1]).steps())
lamda, f_lamda = hyper_para.optim(10)
print(lamda, f_lamda)
plt.scatter(torch.Tensor.numpy(lamda), torch.Tensor.numpy(f_lamda))
plt.show()

print(torch.randn(3))
print(torch.tensor([1,4,2]))