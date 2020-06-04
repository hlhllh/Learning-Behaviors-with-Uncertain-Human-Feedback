# -*- coding: utf-8 -*-
"""
ISABL
"""

import numpy as np
import scipy as sp
import scipy.integrate
import random
from collections import deque
import pickle
import pylab
import matplotlib.pyplot as plt
import cv2
import time
import os
from util import show_image, gau, get_action, get_p #environment

os.makedirs(r'data',exist_ok=True)

E = 0.1
STEP = 75
N_ACT = 6 # number of actions
N_S = 4 # number of states
dog_size = 100 # size of figures
rat_size = 50
length = 500 # side length of background

def randomPolicy():
    return np.array(np.random.randint(N_ACT, size=N_S)) # up right down left 0 1 2 3

def EMupdate(a,b,h_n):
    lamda_n = np.zeros([N_S,1])
    for i in range(N_S):
        temp = np.argmax(a*(h_n[i,:,0]-h_n[i,:,1])+b*h_n[i,:,2])
        try:
            lamda_n[i] = temp[0]
        except: 
            lamda_n[i] = temp         
    return lamda_n

def fun1(x,y,h,lamda):
    # x: mu_plus y: mu_minis
    a=1
    for i in h:
        if i[1] == lamda[i[0]]:
            if i[2] == 0:
                a = a*((1-E)*(1-x))
            elif i[2] == 1:
                a = a*(E*(1-y))
            else:
                a = a*((1-E)*x+E*y)
        else:
            if i[2] == 0:
                a = a*(E*(1-x))
            elif i[2] == 1:
                a = a*((1-E)*(1-y))
            else:
                a = a*(E*x+(1-E)*y)    
    return a

def fun2(x,y,h,lamda):
     # x: mu_plus y: mu_minis
    a = fun1(x,y,h,lamda)
    b = a*(np.log(((1-E)*x+E*y)/(E*x+(1-E)*y)))
    return b

#def main():
record_all = np.zeros([8,75])
record_p = np.zeros([8,STEP])
norm_act = np.zeros(8)

for N_ACT in [6]: # np.linspace(2,9,8):
    N_S = int(N_S)
    PI = [np.mod(i,N_ACT) for i in range(N_S)]
    p_act = get_p(PI,N_S,N_ACT)
    prob=p_act
    jus = np.zeros(5)
    num = 0
    correct = 0
    num_act = 0
    record_tad = np.zeros(STEP)
    record_tad_p = np.zeros(STEP)
    
    for k in range(1):
        record_ad = np.zeros(STEP)
        record_ad_p = np.zeros(STEP)
        record=np.zeros(STEP)
        lamda = randomPolicy()
        h = deque()
        h_n = np.zeros([N_S,N_ACT,3]) # counting the feedback number respect to o,a
        t = 0
        done = 0
        s = 0
        step_t = 0 
        while done != 1:
            a = np.random.randint(0,N_ACT)
            dog_action = a
            rat_action = get_action(p_act[s],N_ACT)
            f,result = show_image(s,rat_action,dog_action,dog_size,rat_size,length, N_ACT, PI)
            step_t += 1
            if f == 3 or step_t >= 16:
                s += 1
                step_t = 0
                if s >= N_S:
                    print('finish experiment')
                    break
                else:
                    continue
            h.append((s,a,f))
            if len(h) > STEP: # limitation of memory
                h.popleft()
            h_n[s,a,f] += 1
#update of isabl ------------------------------------------------
            lamda = randomPolicy()
            lamda_t = np.zeros([N_S,1])
            a_p = sp.integrate.dblquad(fun1 , 0, 1, lambda y: 0, lambda y: 1, args=(h,lamda))[0]
            a_p = a_p*np.log((1-E)/E) # calculate parameters
            b_p = sp.integrate.dblquad(fun2, 0, 1, lambda y: 0, lambda y: 1, args=(h,lamda))[0]
            while (lamda != lamda_t).any(): # EM algorithm
                lamda_t = lamda.copy()
                lamda = EMupdate(a_p,b_p,h_n)
            record[t-1] = result    
            
            if t == 0:
                record_ad[t] = np.abs(a-PI[s])
                record_ad_p[t] = prob[s,a]
            else:
                record_ad[t] = np.abs(a-PI[s])+record_ad[t-1]
                record_ad_p[t] = prob[s,a] + record_ad_p[t-1]
            t=t+1

            jus[np.mod(t,5)]=(lamda.T == PI).all()
            done = 1 if t>=STEP or (jus == np.ones(5)).all() else 0 # termination
            if done == 1 and (lamda.T != PI).any(): # and t<STEP:
                num += 1
            if done == 1 and (lamda.T == PI).all(): # and lamda != 2:    
                correct += 1
                num_act += (t+1)
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
    print('isabl error step',num)
    np.save(r'data\isabl_record_p',record_all)
    np.save(r'data\isabl_record_all',record_p)
    np.save(r'data\isabl_norm_act',norm_act)

#if __name__ == '__main__':
#  main()       
