# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:29:48 2024

@author: Wenjie Liu
"""


import numpy as np
import pandas as pd
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

for station_name in ['Station1']:      #This is station
    for my_leibie in [5]:         #This is number of clustering
        for lead_time in [1,2,3,4,5,6]:   #This is forecast period
            final_predcali = np.empty((0,2))    
            final_predtest = np.empty((0,2))  
            for ii in range(my_leibie):
                data = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\K_'+str(ii+1)+'\\forecasting_results.xlsx', sheet_name='value')
                data = np.array(data)

                predcali = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\K_'+str(ii+1)+'\\forecasting_results.xlsx', sheet_name='now_predcali')       
                predtest = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\K_'+str(ii+1)+'\\forecasting_results.xlsx', sheet_name='now_predtest')   
                
                predcali = np.array(predcali)
                predtest = np.array(predtest)
                final_predcali = np.vstack((final_predcali, predcali))
                final_predtest = np.vstack((final_predtest, predtest))
                
            orig_data = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\observed_runoff.xlsx')
            orig_data = np.array(orig_data)
            index_cali = np.argsort(final_predcali[:,0])
            index_test = np.argsort(final_predtest[:,0])
            my_final_predcali = final_predcali[index_cali]
            my_final_predtest = final_predtest[index_test]
            orig_cali = orig_data[0:int(0.8*orig_data.shape[0])  , :]
            orig_test = orig_data[int(0.8*orig_data.shape[0]):  , :]
            if ( orig_cali.shape[0] == my_final_predcali.shape[0] and orig_test.shape[0] == my_final_predtest.shape[0] ) :
                print('Shape is right!')
                    
            RMSE1, MAE1, R1, NSE1 = error(orig_cali, my_final_predcali[:,-1].reshape(-1,1))
            RMSE2, MAE2, R2, NSE2 = error(orig_test, my_final_predtest[:,-1].reshape(-1,1))

                
            result_cali = np.hstack((orig_cali,my_final_predcali[:,-1].reshape(-1,1)))
            result_test = np.hstack((orig_test,my_final_predtest[:,-1].reshape(-1,1)))
                
            result_cali = pd.DataFrame(result_cali, columns=['real','pred'])
            result_test = pd.DataFrame(result_test, columns=['real','pred'])
            result_value = pd.DataFrame([RMSE1, MAE1, R1, NSE1,RMSE2, MAE2, R2, NSE2], columns=['value'])
                
            writer = pd.ExcelWriter(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\\'+station_name+'\\Period'+str(lead_time)+'\\Total_'+str(my_leibie)+'\\statistical_index.xlsx')
            result_cali.to_excel(writer, sheet_name='train', index=False)
            result_test.to_excel(writer, sheet_name='test', index=False)
            result_value.to_excel(writer, sheet_name='value', index=False)
            writer.close()
            
        






































