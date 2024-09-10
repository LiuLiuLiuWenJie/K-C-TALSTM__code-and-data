# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:39:54 2024

@author: Wenjie Liu
"""

import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch import nn
import math


def error(testY,predY):
    sum_testY_arg=0
    testY_arg=0
    for i in range(0,len(testY)):
        sum_testY_arg=sum_testY_arg+testY[i]
    testY_arg=sum_testY_arg/len(testY)
    sum_predY_arg=0
    predY_arg=0
    for i in range(0,len(predY)):
        sum_predY_arg=sum_predY_arg+predY[i]
    predY_arg=sum_predY_arg/len(predY)     
    sum_R1=0
    sum_R2=0
    sum_R3=0
    for i in range(0,len(testY)):
        sum_R1=sum_R1+(testY[i]-testY_arg)*(predY[i]-predY_arg)
        sum_R2=sum_R2+(testY[i]-testY_arg)**2
        sum_R3=sum_R3+(predY[i]-predY_arg)**2
    R=sum_R1/((sum_R2**0.5)*(sum_R3**0.5))
    sum_RMSE=0  
    for i in range(0,len(testY)):
        sum_RMSE=sum_RMSE+(testY[i]-predY[i])**2
    RMSE=(sum_RMSE/len(testY))**0.5
    sum_NSE1=0
    sum_NSE2=0
    for i in range(0,len(testY)):
        sum_NSE1=sum_NSE1+(testY[i]-predY[i])**2
        sum_NSE2=sum_NSE2+(testY[i]-testY_arg)**2
    NSE=1-sum_NSE1/sum_NSE2
    sum_MAE=0
    for i in range(0,len(testY)):
        sum_MAE=sum_MAE+abs((predY[i,]-testY[i]))
    MAE=sum_MAE/len(testY)
    return RMSE,MAE,R,NSE

def j_transition(x, h, U, W, b):
    j_uotput = torch.tanh(torch.einsum("abc,bcd->abd", h, U) +\
                          torch.einsum("abc,cbd->acd", x, W) + b)
    return j_uotput

def gate_transition(x,fc_gate):
    gate_otput = torch.sigmoid(fc_gate(x))
    return gate_otput

def exp_method(x):
    x = torch.exp(x)
    x = x / torch.sum(x, dim=1, keepdim=True)
    return x

class ILSTM_SV(nn.Module):
    def __init__(self, input_dim,timestep,output_dim, hidden_size, std=0.01):
        super().__init__()
        self.W_j = nn.Parameter(torch.randn(input_dim, 1, hidden_size) * std)
        self.U_j = nn.Parameter(torch.randn(input_dim, hidden_size, hidden_size) * std)
        self.b_j = nn.Parameter(torch.randn(input_dim, hidden_size) * std)
        self.W_i = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        self.W_f = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        self.W_o = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        self.FC_mulit_FV = nn.Linear(hidden_size, output_dim)
        self.FC_aten_predicor = nn.Linear(2 * hidden_size, 1)
        self.FC_predicor_output = nn.Linear(timestep, 1)
        self.FC_predicor_vect = nn.Linear(2 * hidden_size, 1)
        self.FC_temporal_output = nn.Linear(input_dim, 1)
        self.FC_temporal_vect = nn.Linear(input_dim+hidden_size, 1)
        self.FC_aten_temporal = nn.Linear(input_dim+hidden_size, 1)
        self.FC_final_output=nn.Linear(2 * output_dim, 1)
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        outputs = []
        h_t = torch.zeros(x.shape[0], self.input_dim, self.hidden_size)
        c_t = torch.zeros(x.shape[0], self.input_dim*self.hidden_size)
        for t in range(x.shape[1]):
            x_timstep=x[:,t,:].unsqueeze(1)
            gate_input= torch.cat([x[:, t, :], h_t.view(h_t.shape[0], -1)], dim=1) 
            j_t = j_transition(x_timstep,h_t,self.U_j,self.W_j,self.b_j) 
            i_t = gate_transition(gate_input,self.W_i)  
            f_t = gate_transition(gate_input,self.W_f)  
            o_t = gate_transition(gate_input,self.W_o)  
            c_t = c_t * f_t + i_t * j_t.reshape(j_t.shape[0], -1) 
            h_t = (o_t * torch.tanh(c_t)).view(h_t.shape[0], self.input_dim, self.hidden_size) 
            outputs += [h_t] 
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3) 
        outputs = self.dropout(outputs)
        mulit_FV_aten=self.FC_mulit_FV(outputs)  
        mulit_FV_aten = exp_method(mulit_FV_aten)  
        mulit_FV_aten_input = mulit_FV_aten*outputs  
        predicor_aten_input = mulit_FV_aten_input.permute(0, 2, 3, 1) 
        predicor_aten_output = self.FC_predicor_output(predicor_aten_input) 
        if len(predicor_aten_output.squeeze().shape) == 2:
            predicor_aten_output = predicor_aten_output.squeeze()
            predicor_aten_output = predicor_aten_output.unsqueeze(dim=0)
            predicor_comb_vect = torch.cat([predicor_aten_output, h_t], dim=2)  
        else:
            predicor_comb_vect = torch.cat([predicor_aten_output.squeeze(), h_t], dim=2)  
        predicor_comb_vect_FCoutput = self.FC_predicor_vect(predicor_comb_vect)  
        predicor_aten = self.FC_aten_predicor(predicor_comb_vect)  
        predicor_aten = exp_method(predicor_aten)  
        predicor_prediction = torch.sum(predicor_aten * predicor_comb_vect_FCoutput, dim=1)  
        temporal_aten_input = mulit_FV_aten_input.permute(0, 1, 3, 2)  
        temporal_aten_output = self.FC_temporal_output(temporal_aten_input)  
        if len(temporal_aten_output.squeeze().shape) == 2:
            temporal_aten_output = temporal_aten_output.squeeze()
            temporal_aten_output = temporal_aten_output.unsqueeze(dim=0)
            temporal_comb_vect = torch.cat([temporal_aten_output, x], dim=2) 
        else:
            temporal_comb_vect = torch.cat([temporal_aten_output.squeeze(), x], dim=2)  
        temporal_comb_vect_FCoutput = self.FC_temporal_vect(temporal_comb_vect) 
        temporal_aten = self.FC_aten_temporal(temporal_comb_vect) 
        temporal_aten = exp_method(temporal_aten) 
        temporal_prediction = torch.sum(temporal_aten * temporal_comb_vect_FCoutput, dim=1) 
        final_input = torch.cat([predicor_prediction, temporal_prediction], dim=1) 
        prediction=self.FC_final_output(final_input) 
        return prediction, mulit_FV_aten, predicor_aten, temporal_aten

def generatedatafluxnet(path, my_sheetname):
    data = pd.read_excel(path, sheet_name=my_sheetname)
    data = np.array(data)
    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()
    scaler = scaler.fit(data)
    scaler1 = scaler1.fit(data[:, -1].reshape(-1,1))
    return data,scaler1,scaler

def LSTMDataGenerator(data, lead_time, batch_size,seq_length):
    my_data = data 
    xs = []
    ys = []
    for i in range(len(my_data)-seq_length-lead_time+1):
        x = my_data[i:(i + seq_length),:]
        y = my_data[i+seq_length+lead_time-1,-1]
        xs.append(x)
        ys.append(y)
    train_xt = np.array(xs)
    train_yt = np.array(ys)
    train_xt = torch.from_numpy(train_xt).float()
    train_yt = torch.from_numpy(train_yt).float()
    train_zt = train_yt
    return train_xt, train_yt, train_zt

def train_lstm(h_s, l_r):
        model = ILSTM_SV(cali_datax.shape[2], cali_datax.shape[1], 1, 2**int(h_s))
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
        loss_func = torch.nn.MSELoss()
        for epoch in range(total_epoch):
            for step, (x, y) in enumerate(train_loader):
                output, mulit_FV_aten, predicor_aten,temporal_aten = model(x)
                y = y.view(-1, 1)
                train_loss = loss_func(output, y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        model.eval()
        valid_output, mulit_FV_aten, predicor_aten, temporal_aten = model(test_datax)
        valid_loss = loss_func(valid_output, test_datay.view(-1, 1)) 
        my_loss = valid_loss.item()
        return my_loss, model
def CSA(n_dim, size_pop, max_gen, xmax, xmin): 

    a = 0.1
    b = 0.15  
    M = 3
    pop = np.zeros((size_pop, n_dim))
    fitness = np.zeros((size_pop,1))
    u_fitness = np.zeros((size_pop,1))
    v_fitness = np.zeros((size_pop,1))
    for i in range(n_dim):
        for j in range(size_pop):
            pop[j][i] = np.random.uniform(0,1)*(xmax[i]-xmin[i])+xmin[i]
    model1={}
    model2={}
    model3={}
    for i in range(size_pop):
        fitness[i], model1[i] = train_lstm(pop[i][0], pop[i][1])
    px = np.argsort(fitness)  
    gbest = np.zeros((M,n_dim)) 
    fitness_gbest=np.zeros(M)
    for i in range(0, M):
        gbest[i] = pop[px[i]]  
        fitness_gbest[i]=fitness[px[i]]
    pbest=pop
    fitness_pbest = fitness 
    best_model = model1
    
    generation_y = np.zeros(max_gen)
    generation_x = np.zeros((max_gen,n_dim)) 
    t = 0
    U=np.zeros((size_pop,n_dim))
    v=np.zeros((size_pop,n_dim))
    c = 0.5 * np.sum(np.vstack((xmax,xmin)),axis = 0)
    while t < max_gen:
        for i in range(0, size_pop):
            for j in range(0, n_dim):
                ind = np.random.randint(0, M)
                A = math.log(1 / np.random.uniform(0, 1)) * (gbest[ind][j] - pop[i][j])
                B = a * np.random.uniform(0, 1) * (np.mean(gbest, 0)[j] - pop[i][j])
                C = b * np.random.uniform(0, 1) * (np.mean(pbest, 0)[j] - pop[i][j])
                U[i][j]=pop[i][j]+A+B+C
                U[i][j]=np.clip(U[i][j],xmin[j],xmax[j])
        for i in range(0, size_pop):
            for j in range(0, n_dim):
                if U[i][j] >= c[j]:
                    if abs(U[i][j] - c[j]) < np.random.uniform(0, 1) * abs(xmax[j] - xmin[j]):
                        v[i][j]=np.random.uniform(xmax[j] + xmin[j] - U[i][j], c[j])
                    else:
                        v[i][j] =np.random.uniform(xmin[j], xmax[j] + xmin[j] - U[i][j])
                else:
                    if abs(U[i][j] - c[j]) < np.random.uniform(0, 1) * abs(xmax[j] - xmin[j]):
                        v[i][j] = np.random.uniform(c[j], xmax[j] + xmin[j] - U[i][j])
                    else:
                        v[i][j] =np.random.uniform(xmax[j] + xmin[j] - U[i][j], xmax[j])
                v[i][j] = np.clip(v[i][j], xmin[j], xmax[j])
        for i in range(size_pop):
            u_fitness[i], model2[i] = train_lstm(U[i][0], U[i][1])
            v_fitness[i], model3[i] = train_lstm(v[i][0], v[i][1])
        
        for i in range(size_pop):
            if u_fitness[i]<=v_fitness[i]:
                pop[i]=U[i]
                fitness[i]=u_fitness[i]
                model1[i] = model2[i]
            else:
                pop[i]=v[i]
                fitness[i]=v_fitness[i]
                model1[i] = model3[i]
        for i in range(size_pop):
            if fitness_pbest[i] > fitness[i]:
                fitness_pbest[i] = fitness[i]
                pbest[i] = pop[i]
                best_model[i] = model1[i]
            for j in range(0, M):
                if fitness_gbest[j] > fitness_pbest[i]:
                    fitness_gbest[j] = fitness_pbest[i]
                    gbest[j] = pop[i]  
                    my_model = best_model[i]
                    break
        generation_y[t] = fitness_gbest[0]
        generation_x[t] = gbest[0]
        t = t + 1
    return generation_x[-1], my_model, generation_y[-1], generation_y

def create_predictions(model, data_test_x,scaler):
    preds, mulit_FV_aten, predicor_aten, temporal_aten  = model(data_test_x)
    preds = preds.cpu()
    preds = preds.detach().numpy()
    preds = scaler.inverse_transform(preds)
    return preds, mulit_FV_aten, predicor_aten, temporal_aten

    
for station_name in ['Station1']:  #This is station
    for my_leibie in [5]:     #This is the number of clusters [3,4,5]
        for lead_time in [1]:   #This is the forecast period [1,2,3,4,5,6]
            total_epoch = 100
            batch_size = 15
            seq_length = 12
        
            data = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\\Results_for_kmeans.xlsx',sheet_name=station_name)
            data = np.array(data)
            lable = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\Period'+str(lead_time)+'\Total_'+str(my_leibie)+'\\clustering_label.xlsx')
            lable = np.array(lable)
            k_num = int(max(lable))+1


            scaler = MinMaxScaler()
            scaler1 = MinMaxScaler()
            scaler = scaler.fit(data)
            scaler1 = scaler1.fit(data[:, -1].reshape(-1,1))
            my_data = scaler.transform(data)

            xs = []
            ys = []
            for i in range(len(my_data)-seq_length-lead_time+1):
                x = my_data[i:(i + seq_length),:]
                y = my_data[i+seq_length+lead_time-1,-1]
                xs.append(x)
                ys.append(y)
            xt = np.array(xs) 
            yt = np.array(ys)
            yt = yt.reshape(-1,1)
            xt = torch.from_numpy(xt).float()
            yt = torch.from_numpy(yt).float()

            #zt用于计算error
            zt = yt
            zt = zt.detach().numpy()
            zt = zt.reshape(-1,1)
            zt = scaler1.inverse_transform(zt)
            #储存zt用于计算error！！！！！！！！！！！！
            zt = pd.DataFrame(zt, columns = ['observed_runoff'])
            zt.to_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\observed_runoff.xlsx', index = False)


            index = np.arange(0, xt.shape[0])
            index = index.reshape(-1,1)
            if lable.shape[0] == xt.shape[0]:
                print('The lable is right')
            #划分训练集、验证集和测试集
            train_xt = xt[0 : int(0.7*xt.shape[0]), :, :]
            train_yt = yt[0 : int(0.7*xt.shape[0]), :]
            vali_xt = xt[int(0.7*xt.shape[0]) : int(0.8*xt.shape[0]), :]
            vali_yt = yt[int(0.7*xt.shape[0]) : int(0.8*xt.shape[0]), :]
            test_xt = xt[int(0.8*xt.shape[0]): , :]
            test_yt = yt[int(0.8*xt.shape[0]): , :]

            cali_xt = xt[0 : int(0.8*xt.shape[0]), :]
            cali_yt = yt[0 : int(0.8*xt.shape[0]), :]

            train_lable = lable[0 : int(0.7*xt.shape[0]), :]
            vali_lable = lable[int(0.7*xt.shape[0]) : int(0.8*xt.shape[0]), :]
            test_lable = lable[int(0.8*xt.shape[0]): , :]
            cali_lable = lable[0 : int(0.8*xt.shape[0]), :]

            train_index = index[0 : int(0.7*xt.shape[0]), :]
            vali_index = index[int(0.7*xt.shape[0]) : int(0.8*xt.shape[0]), :]
            test_index = index[int(0.8*xt.shape[0]): , :]
            cali_index = index[0 : int(0.8*xt.shape[0]), :]

                
            empty_predtest = [[] for i in range(k_num)]
            empty_predcali = [[] for i in range(k_num)]

            temporal_column =[]
            predicor_column =[]

            for i in range(my_leibie):
                train_arg = np.where(train_lable == i)[0]
                train_datax = train_xt[train_arg]
                train_datay = train_yt[train_arg]
                train_in = train_index[train_arg]
                    
                vali_arg = np.where(vali_lable == i)[0]
                vali_datax = vali_xt[vali_arg]
                vali_datay = vali_yt[vali_arg]
                vali_in = vali_index[vali_arg]
                    
                test_arg = np.where(test_lable == i)[0]
                test_datax = test_xt[test_arg]
                test_datay = test_yt[test_arg]
                test_in = test_index[test_arg]
                    
                cali_arg = np.where(cali_lable == i)[0]
                cali_datax = cali_xt[cali_arg]
                cali_datay = cali_yt[cali_arg]
                cali_in = cali_index[cali_arg]
                    
                    
                train_data = Data.TensorDataset(cali_datax, cali_datay)
                train_loader = Data.DataLoader(
                      dataset = train_data,
                      batch_size = batch_size,
                      shuffle = False,
                      num_workers = 0
                      )

                my_dim = 2
                my_size_pop = 15
                my_max_gen = 20
                my_xmax = np.array([9, 0.05])
                my_xmin = np.array([4, 0.001])
                best_canshu, my_model, best_vali_mse, iter_csa = CSA(n_dim=my_dim, size_pop=my_size_pop, max_gen=my_max_gen, xmax=my_xmax, xmin=my_xmin)
                    
                pred_test, mulit_FV_aten_test, predicor_import_test, temporal_import_test = create_predictions(my_model, test_datax, scaler1)
                pred_cali, mulit_FV_aten_cali, predicor_import_cali, temporal_import_cali = create_predictions(my_model, cali_datax, scaler1)
                    
                orig_test = test_datay.detach().numpy()
                orig_test = orig_test.reshape(-1,1)
                    
                orig_cali = cali_datay.detach().numpy()
                orig_cali = orig_cali.reshape(-1,1)
                    
                orig_test = scaler1.inverse_transform(orig_test)
                orig_cali = scaler1.inverse_transform(orig_cali)
                    
                RMSE1, MAE1, R1, NSE1 = error(orig_cali, pred_cali)
                RMSE2, MAE2, R2, NSE2 = error(orig_test, pred_test)
                    
                plt.plot(orig_test,'black')
                plt.plot(pred_test,'blue')
                plt.show()

                mulit_FV_aten_test = mulit_FV_aten_test.detach().numpy()
                predicor_import_test = predicor_import_test.detach().numpy()
                temporal_import_test = temporal_import_test.detach().numpy()  
                    
                mulit_FV_aten_test = mulit_FV_aten_test.reshape(mulit_FV_aten_test.shape[0], -1)
                predicor_import_test = predicor_import_test.reshape(predicor_import_test.shape[0], -1)
                temporal_import_test = temporal_import_test.reshape(temporal_import_test.shape[0], -1)
                    
                final_temporal_import_test = temporal_import_test.mean(axis=0)
                final_predicor_import_test = predicor_import_test.mean(axis=0)
                
                predicor_column = []
                for j in range(predicor_import_test.shape[1]):
                    predicor_column.append('Factor-'+str(j+1))
                temporal_column = []
                for j in range(12):
                    temporal_column.append('T-'+str(12-j))    
                    
                predicor_import_test = pd.DataFrame(predicor_import_test, columns=predicor_column)
                mulit_FV_aten_test = pd.DataFrame(mulit_FV_aten_test)
                temporal_import_test = pd.DataFrame(temporal_import_test, columns=temporal_column)
                    
                cali_in = cali_in.reshape(-1,1)
                test_in = test_in.reshape(-1,1)
                    
                now_predtest = np.hstack((test_in, pred_test))
                now_predcali = np.hstack((cali_in, pred_cali))
                value = [RMSE1, MAE1, R1, NSE1, RMSE2, MAE2, R2, NSE2]
                value = np.array(value)
                value = pd.DataFrame((RMSE1, MAE1, R1, NSE1, RMSE2, MAE2, R2, NSE2), columns = ['value']) 
                    
                now_predtest = pd.DataFrame(now_predtest, columns = ['Index','now_predtest'])
                now_predcali = pd.DataFrame(now_predcali, columns = ['Index','now_predcali'])                           
                writer = pd.ExcelWriter(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\K_'+str(i+1)+'\\forecasting_results.xlsx')

                now_predtest.to_excel(writer, sheet_name='now_predtest', index=False)
                now_predcali.to_excel(writer, sheet_name='now_predcali', index=False)
                value.to_excel(writer, sheet_name='value', index=False)
                predicor_import_test.to_excel(writer, sheet_name='predicor_import_test', index=False)
                temporal_import_test.to_excel(writer, sheet_name='temporal_import_test', index=False)
                mulit_FV_aten_test.to_excel(writer, sheet_name='mulit_FV_aten_test', index=False)
                writer.close()
  