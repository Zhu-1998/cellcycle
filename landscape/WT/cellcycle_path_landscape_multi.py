# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 02:03:39 2022

@author: USER
"""

from joblib import Parallel, delayed
import anndata
import numpy as np
# from multiprocessing import Pool, Lock
import pandas as pd
import seaborn as sns
import sys
import os
import time
import matplotlib.pyplot as plt
import dynamo as dyn
dyn.dynamo_logger.main_silence()

import warnings
warnings.filterwarnings('ignore')


#帮助您调试与版本相关的错误（如果有的话）
dyn.get_all_dependencies_version()

#模拟带有白色背景的 ggplot2 绘图样式
dyn.configuration.set_figure_params('dynamo', background='white')


#adata_labeling = anndata.read("/home/wj/datadisk/zlg/singlecell/HSC/hematopoiesis_v1.h5ad")
adata = anndata.read("/home/wj/datadisk/zlg/singlecell/cellcycle/cell_cycle.h5ad")
dyn.pp.recipe_monocle(adata)

dyn.tl.dynamics(adata, model='stochastic', cores=20)


#dyn.tl.reduceDimension(adata)
#dyn.pl.umap(adata, color='cell_cycle_phase')

dyn.tl.cell_velocities(adata, method='pearson', other_kernels_dict={'transform': 'sqrt'})

dyn.tl.cell_wise_confidence(adata)

#dyn.pl.streamline_plot(adata, color=['cell_cycle_phase'], basis='umap', show_legend='on data', show_arrowed_spines=True)

dyn.vf.VectorField(adata, basis='umap', M=1000, pot_curl_div=True)

dyn.vf.topography(adata, n=250, basis='umap');

#dyn.pl.topography(adata, basis='umap', background='white', color=['ntr', 'cell_cycle_phase'], streamline_color='black', show_legend='on data', frontier=True)

dyn.tl.cell_velocities(adata, basis='pca')
dyn.vf.VectorField(adata, basis='pca')
dyn.vf.speed(adata, basis='pca')
dyn.vf.curl(adata, basis='umap')
dyn.vf.divergence(adata, basis='pca')
dyn.vf.acceleration(adata, basis='pca')
dyn.vf.curvature(adata, basis='pca')

VecFld = adata.uns['VecFld_umap']

def vector_field_function(x, VecFld=VecFld, dim=None, kernel="full", X_ctrl_ind=None):
    """Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.

		Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
		"""

    x = np.array(x).reshape((1, -1))
    if np.size(x) == 1:
        x = x[None, :]
    K = dyn.vf.utils.con_K(x, VecFld["X_ctrl"], VecFld["beta"])
    
    if X_ctrl_ind is not None:
        C = np.zeros_like(VecFld["C"])
        C[X_ctrl_ind, :] = VecFld["C"][X_ctrl_ind, :]
    else:
        C = VecFld["C"]

    K = K.dot(C)
    return K
#################################################################################

##################################LHS参数抽样###################################
def LHSample( D,bounds,N):#直接输出抽样  
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''

    result = np.empty([N, D]) #产生一个N*D的数组,元素任意
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0]           

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return result #直接输出抽样 
#########################################################################################

VecFnc = vector_field_function

x_lim=[-4, 15]
y_lim=[-1, 12]
Dim = 2       #参数维数
bounds =[x_lim, y_lim]
N = 400
LHS_of_paras = LHSample(Dim, bounds, N)

numTimeSteps=5000000
starttime = 1000000
Tra_grid = 100


def path_function(D, dt, i):    
    x_path = []
    y_path = []
    num_tra = np.zeros((Tra_grid, Tra_grid))
    total_Fx = np.zeros((Tra_grid, Tra_grid))
    total_Fy = np.zeros((Tra_grid, Tra_grid))

    init_xy = LHS_of_paras[i, :]
    x0 = init_xy[0]
    y0 = init_xy[1]
 
    # Initialize "path" variables
    x_p = x0
    y_p = y0
    # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
    for n_steps in np.arange(1, numTimeSteps):
        # update dxdt, dydt
        dxdt, dydt = VecFnc([x_p, y_p])
        
        # update x, y
        dx = dxdt * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dy = dydt * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        
        # dx = dxdt * dt
        # dy = dydt * dt

        x_p = x_p + dx
        y_p = y_p + dy
        
        if x_p < x_lim[0]:
            x_p = 2 * x_lim[0] - x_p
        if y_p < x_lim[0]:
            y_p = 2 * y_lim[0] - y_p
            
        if x_p > x_lim[1]:
            x_p = 2 * x_lim[1] - x_p
        if y_p > y_lim[1]:
            y_p = 2 * y_lim[1] - y_p
            
        dxdt, dydt = VecFnc([x_p, y_p])  
        x_path.append(x_p)
        y_path.append(y_p)
        
        if n_steps > starttime:
            A = int((x_p - x_lim[0]) * Tra_grid / (x_lim[1] - x_lim[0]))   
            B = int((y_p - y_lim[0]) * Tra_grid / (y_lim[1] - y_lim[0]))
            if A < Tra_grid and B<Tra_grid:
                num_tra[A, B] = num_tra[A, B] + 1;
                total_Fx[A, B] = total_Fx[A, B] + dxdt
                total_Fy[A, B] = total_Fy[A, B] + dydt
        
    np.savetxt('num_tra_' + np.str(i) + '.csv', num_tra, delimiter=",") 
    np.savetxt('total_Fx_' + np.str(i) + '.csv', total_Fx, delimiter=",") 
    np.savetxt('total_Fy_' + np.str(i) + '.csv', total_Fy, delimiter=",") 
    

        
    print("Saving path_time to txt")
    np.savetxt('path_time' + np.str(i) + '.txt', list(zip(x_path, y_path)), fmt='%.5f %.5f',
               header='{:<8} {:<25}'.format('umap1', 'umap2'))
               
    plt.plot(np.linspace(0, 1, int(numTimeSteps-1)), x_path, color='blue', label='umap1') 
    plt.plot(np.linspace(0, 1, int(numTimeSteps-1)), y_path, color='green', label='umap2')   
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('X_umap')
    print("Saving path_time to png")
    plt.savefig('path_time' + np.str(i) + '.png')
    plt.close()




start = time.time()
# 并行
Parallel(n_jobs=150)(
    delayed(path_function)(0.00001, 5e-1, i)
    for i in range(N))


num_tra = np.zeros((Tra_grid, Tra_grid))
total_Fx = np.zeros((Tra_grid, Tra_grid))
total_Fy = np.zeros((Tra_grid, Tra_grid))

for i in range(N):
    num_tra_i = np.loadtxt(open('num_tra_' + np.str(i) + '.csv',"rb"),delimiter=",")
    total_Fx_i = np.loadtxt(open('total_Fx_' + np.str(i) + '.csv',"rb"),delimiter=",")
    total_Fy_i = np.loadtxt(open('total_Fy_' + np.str(i) + '.csv',"rb"),delimiter=",")
    num_tra = num_tra + num_tra_i
    total_Fx = total_Fx + total_Fx_i
    total_Fy = total_Fy + total_Fy_i
    
p_tra = num_tra / (sum(sum(num_tra)))
print([sum(sum(num_tra)), N*(numTimeSteps-starttime)])
pot_U = -np.log(p_tra)
mean_Fx = total_Fx / num_tra
mean_Fy = total_Fy / num_tra

xlin = np.linspace(x_lim[0], x_lim[1], Tra_grid)
ylin = np.linspace(y_lim[0], y_lim[1], Tra_grid)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)

np.savetxt("num_tra.csv", num_tra, delimiter=",") 
np.savetxt("total_Fx.csv", total_Fx, delimiter=",") 
np.savetxt("total_Fy.csv", total_Fy, delimiter=",") 
np.savetxt("p_tra.csv", p_tra, delimiter=",") 
np.savetxt("pot_U.csv", pot_U, delimiter=",") 
np.savetxt("mean_Fx.csv", mean_Fx, delimiter=",") 
np.savetxt("mean_Fy.csv", mean_Fy, delimiter=",") 
np.savetxt("Xgrid.csv", Xgrid, delimiter=",") 
np.savetxt("Ygrid.csv", Ygrid, delimiter=",") 

Time = time.time() - start
print(str(Time) + 's')

