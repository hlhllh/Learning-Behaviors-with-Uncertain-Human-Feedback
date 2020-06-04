# -*- coding: utf-8 -*-
"""
ucb 
"""

import numpy as np
import scipy.integrate
import random
import pickle
import pylab
import matplotlib.pyplot as plt
import cv2
import time
import os
from util import show_image, gau, get_action, get_p #environment

os.makedirs(r'data',exist_ok=True) # path to store data

E = 0.1
STEP = 75 # number of steps
SIGMA = 3
N_ACT = 6 # number of actions
N_S = 4 # number of state
dog_size = 100
rat_size = 50
length = 500
#plt.ion()

#def main():
#    record3 = np.zeros(10)
#    num=0
record_all = np.zeros([8,75])
record_p = np.zeros([8,STEP])
norm_act = np.zeros(8)

for N_ACT in [6]: # np.linspace(2,9,8):
    N_S = int(N_S)
    PI = [np.mod(i,N_ACT) for i in range(N_S)] # policy of rats
    p_act = get_p(PI,N_S,N_ACT)
    prob=p_act
    jus = np.zeros(5)
    num = 0
    correct = 0
    record_tad = np.zeros(STEP)
    record_tad_p = np.zeros(STEP)
    for k in range(1):
        record_ad = np.zeros(STEP)
        record_ad_p = np.zeros(STEP)
        record_n = np.zeros([N_S,N_ACT])
        lamda = np.random.randint(N_ACT,size=N_S) #initialize policy of dog
        lamda_re = lamda.copy() + 1
        record=np.zeros(STEP)
        p = 1/N_ACT*np.ones([N_S,N_ACT]) # policy for each observation
        t = 0
        done = 0
        s = 0
        step_t = 0 
        while done != 1:
            if random.random() < 0:
                a = np.random.randint(0,N_ACT)
            else:
                a = lamda[s]
            dog_action = a
            rat_action = get_action(p_act[s],N_ACT)
            f,result = show_image(s,rat_action,dog_action,dog_size,rat_size,length, N_ACT, PI)
            step_t += 1
            if f == 3 or step_t >= 16: # limit the number of feedback
                s += 1
                step_t = 0
                if s >= N_S:
                    print('finish experiment')
                    break
                else:
                    continue
            record_n[s,a] += 1
            
            if f == 0:
                r = 1
            elif f == 1:
                r = -1
            else:
                r = 0   
#update of ucb----------------------------------------------
            p[s,a] = p[s,a]*(record_n[s,a]-1)/record_n[s,a] + r/record_n[s,a]
            p_t = p.copy()
            for i in range(N_S):
                for j in range(N_ACT):
                    if record_n[i,j] != 0:
                        p_t[i,j] +=  (2*np.log(sum(record_n[i,:]))/record_n[i,j])**0.5
                    else:
                        p_t[i,j] = 1000
                    
            for n in range(N_S): # in case that more than one action has the maximal value
                lamda_t = [i for i, j in enumerate(p_t[n,:]) if j == max(p_t[n,:])]
                if len(lamda_t) != 1:
                    lamda[n] = lamda_t[np.random.randint(0,len(lamda_t))]
                else:
                    lamda[n] = lamda_t[0]

#record-----------------------------------------------
            if t == 0:
                record_ad[t] = np.abs(a-PI[s])
                record_ad_p[t] = prob[s,a]
            else:
                record_ad[t] = np.abs(a-PI[s])+record_ad[t-1]
                record_ad_p[t] = prob[s,a] + record_ad_p[t-1]
            t = t+1
            record[t-1] = result
            jus[np.mod(t,5)]=(lamda == PI).all()
            done = 1 if t>=STEP or (jus == np.ones(5)).all() else 0 #input("terminated or not (0 or 1):") 
            if done == 1 and (lamda != PI).any(): # t<STEP
                num += 1
            if done == 1:
                for i in range(STEP):
                    if record_ad[i] == 0 and i > 0:
                        record_ad[i] = record_ad[i-1]   
                    if record_ad_p[i] == 0 and i>0:
                        record_ad_p[i] = record_ad_p[i-1]  
        record_tad = record_tad + record_ad
        record_tad_p += record_ad_p
        norm_act[N_ACT-2] += np.linalg.norm(lamda-PI,2)**2

        #---------------------------------------------------------
        # data processing   
    record_tad = record_tad/(k+1)
    norm_act[N_ACT-2] /= (k+1)
    record_p[N_ACT-2] = record_tad_p/(k+1)
    record_all[N_ACT-2,:] = record_tad         
    #        num += t    
    print('ucb error step', num)
    np.save(r'data\ucb_record_p',record_all)
    np.save(r'data\ucb_record_all',record_p)
    np.save(r'data\ucb_norm_act',norm_act)
    #record1=np.zeros(STEP)
    #record2=np.zeros(STEP)        
    #for i in range(STEP):
    #    '''if i > 4 and sum(record[i-5:i])/5 != 1:
    #        record3[k] = i'''
    #    record2[i] = sum(record[0:i])/(i+1)
    #    if i > 4:
    #        record1[i] = sum(record[i-5:i])/5            
    ##    pickle.dump(record1, open('F:/data_ana/tmpi_1p.txt', 'wb'))
    ##    pickle.dump(record2, open('F:/data_ana/tmpi_2p.txt', 'wb'))
    #pylab.plot(record1,color='blue')
    #pylab.plot(record2,color='green')
    '''record4 = sum(record3)/10
    #pickle.dump([TU_P,TU_M,record4], open('F:/data_ana/tmpb_2.txt', 'wb'))
    print(record4)'''
    
    
#if __name__ == '__main__':
#  main()      
