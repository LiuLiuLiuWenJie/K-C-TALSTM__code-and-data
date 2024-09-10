# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:10:26 2024

@author: Wenjie Liu
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def LNXS(data, center, cluster_label):
    total_result = []
    for i in range(cluster_num):
        arg = np.where(cluster_label == i)
        cal_result  = np.sqrt(np.sum((data[arg] - center[i])**2, axis=1))
        total_result.append(np.mean(cal_result))
    return np.mean(total_result)

def kmeans(mydata, center, iters):  
    cluster_label1 = np.zeros(mydata.shape[0])  
    C = np.zeros(center.shape)  
    for i in range(center.shape[0]):
        C[i] = center[i]
    for i in range(iters):
        distances = np.sqrt(np.sum((mydata - C[:, np.newaxis])**2, axis=2))
        cluster_label1 = np.argmin(distances, axis=0)
        for j in range(cluster_num):
            C[j] = np.mean(mydata[cluster_label1 == j], axis=0)       
    distances = np.sqrt(np.sum((mydata - C[:, np.newaxis])**2, axis=2))
    cluster_label1 = np.argmin(distances, axis=0) 
    return cluster_label1, C

def kmeans1(mydata, center, iters):
    cluster_label1 = np.zeros(mydata.shape[0])  
    C = np.zeros(center.shape)  
    for i in range(center.shape[0]):
        C[i] = center[i]  
    for i in range(iters):
        distances = np.sqrt(np.sum((mydata - C[:, np.newaxis])**2, axis=2))
        cluster_label1 = np.argmin(distances, axis=0)
        for j in range(cluster_num):
            C[j] = np.mean(mydata[cluster_label1 == j], axis=0)       
    distances = np.sqrt(np.sum((mydata - C[:, np.newaxis])**2, axis=2))
    cluster_label1 = np.argmin(distances, axis=0)
    value = LNXS(mydata, C, cluster_label1)  
    return value

def CSA_kmeans(mydata, n_dim1, n_dim2, xmax, xmin, size_pop, max_gen, iters): 
    a = 0.1
    b = 0.15
    M = 3
    pop = np.zeros((size_pop, n_dim1, n_dim2))
    fitness = np.zeros((size_pop,1))
    u_fitness = np.zeros((size_pop,1))
    v_fitness = np.zeros((size_pop,1))
    for j in range(size_pop):   
        pop[j] = mydata[np.random.choice(mydata.shape[0], cluster_num, replace=False), :]
    for i in range(size_pop):
        fitness[i] = kmeans1(mydata, pop[i], iters)
    px = np.argsort(fitness)  
    gbest = np.zeros((M, n_dim1, n_dim2)) 
    fitness_gbest = np.zeros(M)
    for i in range(0, M):
        gbest[i] = pop[px[i]] 
        fitness_gbest[i] = fitness[px[i]]
    
    pbest = pop
    fitness_pbest = fitness 
    
    generation_y = np.zeros(max_gen)
    generation_x = np.zeros((max_gen, n_dim1, n_dim2)) 
    t = 0
    U=np.zeros((size_pop,n_dim1, n_dim2))
    v=np.zeros((size_pop,n_dim1, n_dim2))
    while t < max_gen:
        c = 0.5 * np.sum(np.vstack((xmax,xmin)),axis = 0)*np.random.uniform(0.8, 1.2)
        for i in range(0, size_pop):
            for j in range(0, n_dim1):
                for k in range(0, n_dim2):   
                    ind = np.random.randint(0, M)
                    A = math.log(1 / np.random.uniform(0, 1)) * (gbest[ind][j][k] - pop[i][j][k])
                    B = a * np.random.uniform(0, 1) * (np.mean(gbest, 0)[j][k] - pop[i][j][k])
                    C = b * np.random.uniform(0, 1) * (np.mean(pbest, 0)[j][k] - pop[i][j][k])
                    suiji = (10**-3)*np.exp(-t)*(xmax[k]-xmin[k])
                    U[i][j][k] = pop[i][j][k] + A + B + C + suiji
                    U[i][j][k] = np.clip(U[i][j][k], xmin[k], xmax[k])

        for i in range(0, size_pop):
            for j in range(0, n_dim1):
                for k in range(0, n_dim2):
                  if U[i][j][k] >= c[k]:
                      if abs(U[i][j][k] - c[k]) < np.random.uniform(0, 1) * abs(xmax[k] - xmin[k]):
                          v[i][j][k] = np.random.uniform(xmax[k] + xmin[k] - U[i][j][k], c[k])
                      else:
                          v[i][j][k] = np.random.uniform(xmin[k], xmax[k] + xmin[k] - U[i][j][k])
                  else:
                      if abs(U[i][j][k] - c[k]) < np.random.uniform(0, 1) * abs(xmax[k] - xmin[k]):
                          v[i][j][k] = np.random.uniform(c[k], xmax[k] + xmin[k] - U[i][j][k])
                      else:
                          v[i][j][k] =np.random.uniform(xmax[k] + xmin[k] - U[i][j][k], xmax[k])
                  v[i][j][k] = np.clip(v[i][j][k], xmin[k], xmax[k])  
        for i in range(size_pop):
            u_fitness[i] = kmeans1(mydata, U[i], iters)
            v_fitness[i] = kmeans1(mydata, v[i], iters)
        for i in range(size_pop):
            if u_fitness[i]<=v_fitness[i]:
                pop[i] = U[i]
                fitness[i] = u_fitness[i]
            else:
                pop[i] = v[i]
                fitness[i] = v_fitness[i]
        for i in range(size_pop):
            if fitness_pbest[i] > fitness[i]:
                fitness_pbest[i] = fitness[i]
                pbest[i] = pop[i]
            for j in range(0, M):
                if fitness_gbest[j] > fitness_pbest[i]:
                    fitness_gbest[j] = fitness_pbest[i]
                    gbest[j] = pbest[i] 
                    break
        generation_y[t] = fitness_gbest[0]
        generation_x[t] = gbest[0]
        t = t + 1
    return generation_x[-1], generation_y, generation_x

def kmeans_classification(mydata, k_num, size_pop, max_gen, iters):
    lie = mydata.shape[1]
    hangmax = np.amax(mydata, axis=0)  
    hangmin = np.amin(mydata, axis=0) 
    orig_center, LNXS_iteration, center_iteration = CSA_kmeans(mydata, n_dim1=k_num, n_dim2=lie, xmax=hangmax, xmin=hangmin, size_pop=size_pop, max_gen=max_gen, iters = iters)
    plt.plot(LNXS_iteration)
    final_label, final_center = kmeans(mydata, orig_center, iters)
    return final_label, final_center, orig_center, LNXS_iteration, center_iteration

def SSE(cluster_data, final_label, final_center):
    total_result = []
    for i in range(cluster_num):
        arg = np.where(final_label == i)
        cal_result  = np.sum((cluster_data[arg] - final_center[i])**2, axis=1)
        total_result.append(np.sum(cal_result))
    return np.sum(total_result)

def LKXS(cluster_data, final_label, final_center):
    total_d = []
    for i in range(cluster_data.shape[0]):
        my_leibie = np.delete(np.where(final_label==final_label[i])[0], np.where(final_label==final_label[i])[0]==i)
        d_in = np.mean(np.sqrt(np.sum((cluster_data[i] - cluster_data[my_leibie])**2, axis=1)))
        my_others = []
        for j in range(cluster_num):
            if j != final_label[i]:
                my_others.append(j) 
        other_result = []        
        for k in my_others:
            other_leibie = np.where(final_label==k)[0]
            d_other = np.mean(np.sqrt(np.sum((cluster_data[i] - cluster_data[other_leibie])**2, axis=1)))
            other_result.append(d_other)
        min_d_other = min(other_result)
        my_d = (min_d_other - d_in)/max(min_d_other,d_in)
        total_d.append(my_d)
    total_d = sum(total_d)/len(total_d)
    lkxs = total_d
    return lkxs
    

seq_length = 12   #This is timestep 12
lead_time = 1     
for lead_time in [1]:
    data = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\Results_for_kmeans.xlsx',sheet_name='Station1') #The information contains the 4 most important climate factors and the corresponding runoff data, n rows and 5 columns
    data = np.array(data)
    scaler = MinMaxScaler()
    scaler1 = scaler.fit(data)
    my_data = scaler1.transform(data) 
    scaler2 = scaler.fit(data[:,-1].reshape(-1,1))

    xs = []
    ys = []
    for i in range(len(my_data)-seq_length-lead_time+1):
        x = my_data[i:(i + seq_length),:]
        y = my_data[i+seq_length+lead_time-1,-1]
        xs.append(x)
        ys.append(y)
    data_x = np.array(ys) 
    data_x = data_x.reshape(-1,1)
    train_datax = data_x[0:int(data_x.shape[0]*0.8)]
    test_datax = data_x[int(data_x.shape[0]*0.8):]

    cluster_data = train_datax
    cluster_num = 5   #This is the cluster number, which you can modify from [3,4,5]

    final_label, final_center, orig_center, LNXS_iteration, center_iteration = kmeans_classification(mydata=cluster_data, k_num=cluster_num, size_pop=20, max_gen=100, iters=100)
    my_SSE = SSE(cluster_data, final_label, final_center)
    my_LKXS = LKXS(cluster_data, final_label, final_center)
    
    test_distance = np.sqrt(np.sum((test_datax - final_center[:, np.newaxis])**2, axis=2))
    test_label = np.argmin(test_distance, axis=0)    

    train_datax1 = scaler2.inverse_transform(train_datax)
    data_x1 = scaler2.inverse_transform(data_x)
    train_center =  scaler2.inverse_transform(final_center)
    for i in range(cluster_num):
        plt.plot(train_datax1[final_label == i],'r')
        plt.show()
    for j in range(cluster_num):
        a = len(np.where(final_label==j)[0])
        b = len(np.where(test_label==j)[0])    
        
    my_label = np.vstack((final_label.reshape(-1,1),test_label.reshape(-1,1)))
    
    my_LKXS = np.array(my_LKXS) 
    my_LKXS = my_LKXS.reshape(-1,1)

    my_label = pd.DataFrame(my_label, columns = ['label'])
    train_center = pd.DataFrame(train_center, columns =['center'])
    data_x1 = pd.DataFrame(data_x1, columns =['runoff'])
    my_LKXS = pd.DataFrame(my_LKXS, columns =['LKXS'])
    
    writer = pd.ExcelWriter(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\Station1\Period'+str(lead_time)+'\Total_'+str(cluster_num)+'\\clustering_label.xlsx')
    my_label.to_excel(writer, sheet_name='label', index=False)
    train_center.to_excel(writer, sheet_name='center', index=False)
    data_x1.to_excel(writer, sheet_name='runoff', index=False)
    my_LKXS.to_excel(writer, sheet_name='LKXS', index=False)
    writer.close()
    
    final_label = np.array(final_label)    
    for lead_time in [2,3,4,5,6]:
        my_final_label = final_label[lead_time-1:]
        my_final_label = pd.DataFrame(my_final_label,columns=['label'])
        my_final_label.to_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\Station1\Period'+str(lead_time)+'\Total_'+str(cluster_num)+'\\clustering_label.xlsx', index=False,sheet_name='lable')    