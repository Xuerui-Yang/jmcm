import numpy as np
import pandas as pd



def poly(para,ipt):
	opt=0
	for i in range(len(para)):
		opt+=para[i]*(ipt**i)
	return opt

def simulate(bta,lmd,gma,m=500,ni_max=10):
	vec_y=np.empty(0)
	vec_id=np.empty(0)
	vec_t=np.empty(0)
	for i in range(m):
		ni=np.random.randint(1,ni_max,1)[0]
		vec_id=np.append(vec_id,[i]*ni)

		ti=np.sort(np.random.uniform(-1,1,ni))
		vec_t=np.append(vec_t,ti)

		yi=np.zeros(ni)
		mu_i=poly(bta, ti)
		for j in range(ni):
			sigma_ij=np.sqrt(np.exp(poly(lmd,ti[j])))
			y_ij=mu_i[j]+np.random.normal(0,sigma_ij)
			for k in range(j):
				phi_ijk=poly(gma, ti[j]-ti[k])
				y_ij+=phi_ijk*(yi[k]-mu_i[k])
			yi[j]=y_ij
		vec_y=np.append(vec_y,yi)
	df = pd.DataFrame({'y': vec_y, 'id': vec_id, 't': vec_t})
	return df

def sim_par(num_bta,num_lmd,num_gma):
	bta=np.random.uniform(-1,1,(num_bta,))
	lmd=np.random.uniform(-1,1,(num_lmd,))
	gma=np.random.uniform(-1,1,(num_gma,))
	return bta,lmd,gma
