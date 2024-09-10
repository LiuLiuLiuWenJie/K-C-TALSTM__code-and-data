# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:57:36 2024

@author: Wenjie Liu
"""


import minepy
import numpy as np
import pandas as pd
def MIC(x, y):
    mic = minepy.MINE(alpha=0.6, c=15)
    mic.compute_score(x, y)
    final_min = mic.mic()
    return final_min

#To read the climatic phenomenon index data, make sure that the date of the climatic phenomenon index corresponds to the runoff station
datax = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\Data.xlsx',sheet_name='109 climate phenomenon index')
datax = np.array(datax)

#To read runoff data, you can change the station
datay = pd.read_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\Data.xlsx',sheet_name='Runoff_Station1')
datay = np.array(datay)
datay = datay.reshape(-1,1)

total_mic = np.ones((12, datax.shape[1]))
for i in range(datax.shape[1]):
    for j in range(1,13):
        x = datax[0:datax.shape[0]-j , i]
        y = datay[j:, 0]
        final_mic = MIC(x, y)
        total_mic[j-1,i] = final_mic
        
#The MIC coefficient of the station was calculated to screen the top 4 most important climate phenomenon indices       
pd.DataFrame(total_mic).to_excel(r'c:\Users\Wenjie Liu\Desktop\K_C_TALSTM\MIC_Station1.xlsx',index=False)

