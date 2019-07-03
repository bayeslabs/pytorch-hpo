import numpy as np 
import scipy.stats
import torch

# class Expected_Improvement():
class Aquisition_func:
	
	def ExpectedImprovement(f_post_mean, cov_post, f_min):
		x_a = []
		pdf_all = []	
		for i in range(len(f_post_mean)):
			temp_x = np.linspace(float(f_post_mean[i]) - 10*float(cov_post[i][i]), float(f_post_mean[i]) + 10*float(cov_post[i][i]), 1000)
			temp_pdf = scipy.stats.norm.pdf(temp_x, f_post_mean[i], cov_post[i][i])
			pdf_all.append(temp_pdf) 	#storing the distribution of each value of f_post_mean
			x_a.append(temp_x)			#storing all values of f_x following abovie distribution with f_pos_mean as its mean value

		pdf_all = np.array(pdf_all)
		x_a = np.array(x_a) 	#stores the f_x values at x
		# print(x_a.shape, pdf_all.shape)

		def utility_func(f_min, f_x):
			zeros = np.zeros(len(f_x - f_min))
			u_x = np.maximum(zeros, (f_min - f_x))
			# u_x[u_x!=0] = 1 
				
			return u_x 

		def aquisition_func(f_x, pdf_f, f_min):
			u_x = utility_func(f_min, f_x)
			a_x = np.trapz(u_x*pdf_f)/np.trapz(pdf_f)
			a_x = np.where(~np.isnan(a_x), a_x, 0)
			return a_x 

		aquis_arr = [aquisition_func(x_a[i], pdf_all[i], f_min) for i in range(len(pdf_all))]
		candidate = np.argmin(aquis_arr)

		return aquis_arr, candidate

	def ExpectedImprovement_torch(f_post_mean, cov_post, f_min):
		
		def utility_func_torch(f_min, f_x):
			zeros = torch.zeros(len(f_x)).double()
			diff = f_x - f_min
			u_x = torch.min(diff, zeros)
			return u_x

		def aquisition_func_torch(f_x, pdf_f, f_min):
			u_x = utility_func_torch(f_min, f_x)
			a_x = (torch.sum(u_x*pdf_f)*(f_x[0] - f_x[-1]))/(len(pdf_f)+1)#*torch.sum(pdf_f))
			a_x[a_x != a_x] = 0 
			return a_x

		x_a = []
		pdf_all = []	
		for i in range(len(f_post_mean)):
			temp_x = np.linspace(float(f_post_mean[i]) - 4*float(cov_post[i][i]), float(f_post_mean[i]) + 4*float(cov_post[i][i]), 1000)
			temp_pdf = scipy.stats.norm.pdf(temp_x, f_post_mean[i], cov_post[i][i])
			pdf_all.append(temp_pdf) 	#storing the distribution of each value of f_post_mean
			x_a.append(temp_x)			#storing all values of f_x following abovie distribution with f_pos_mean as its mean value
					
		pdf_all = torch.tensor(pdf_all, dtype=torch.double)
		x_a = torch.tensor(x_a, dtype=torch.double) 	#stores the f_x values at x
		print(x_a[2].dtype, x_a[2].shape)
		aquis_arr = [aquisition_func_torch(x_a[i], pdf_all[i], f_min.double()) for i in range(len(pdf_all))]
		aquis_arr = torch.tensor(aquis_arr)
		candidate = torch.argmax(aquis_arr)

		return aquis_arr, candidate