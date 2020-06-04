# -*- coding: utf-8 -*-
"""
ABLUF algortihm
"""

import numpy as np
import scipy as sp
import scipy.integrate
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time
import os
from util import show_image, gau, get_action, get_p #environment

os.makedirs(r'data',exist_ok=True)

E = 0
STEP = 75
ALPHA = 0.4 #learning rate
N_ACT = 6 # number of actions
N_S = 4 # number of states
dog_size = 100 # size of figures
rat_size = 50
length = 500 # side length of background

def gau_sig(x,mu,sigma):
    g_div = ((x-mu)**2) * np.exp(-(x-mu)**2/(2*sigma**2)) #/sigma**3 ignore sigma**3 to increase the step size
    return g_div

def EMupdate(h,lamda,sigma,h_n):
    temp = np.zeros(N_ACT)
    a = np.zeros(N_S)
    for o in range(N_S):
        for i in range(N_ACT):
            temp[i] = sp.integrate.dblquad(fun1, 0, 1, lambda x: 0.01, lambda x: 1, args=(h,lamda,i,o,sigma,h_n), epsrel = 1e-2, epsabs = 1e-2)[0]
        try:
            tem = [i for i, j in enumerate(temp) if j == max(temp)] #  
            a[o] = tem[np.random.randint(len(tem))]
        except:
            a[o] = np.argmax(temp)
    return a

def fun1(x,y,h,lamda,a_n,o,sigma,h_n):
    # x: mu_plus y: mu_minis
    record_h = 1
    record_a = 0
    for i in h:
        if i[2] == 0:
            record_h = record_h*fun_po(i[1],lamda[i[0]],x,sigma)
        if i[2] == 1:
            record_h = record_h*fun_ne(i[1],lamda[i[0]],y,sigma)
        else:
            record_h = record_h*fun_0(i[1],lamda[i[0]],x,y,sigma)
        if i[0] == o:
            if i[2] == 0:
                record_a = record_a+fun_po_log(i[1],a_n,x,sigma)
            if i[2] == 1:
                record_a = record_a+fun_ne_log(i[1],a_n,y,sigma)
            else:
                record_a = record_a+fun_0_log(i[1],a_n,x,y,sigma)
    return record_h*record_a

def fun_po(a_o,a,x,sigma):        
    return gau(a_o,a,sigma)*(1-x)
    
def fun_ne(a_o,a,y,sigma):
    return (1-gau(a_o,a,sigma)*0.99)*(1-y)

def fun_0(a_o,a,x,y,sigma):
    return 1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma)

def fun_po_log(a_o,a,x,sigma):        
    return np.log(gau(a_o,a,sigma))#+np.log(1-x)
    
def fun_ne_log(a_o,a,y,sigma):
    return np.log(1-gau(a_o,a,sigma)*0.99)#+np.log(1-y)

def fun_0_log(a_o,a,x,y,sigma):
    return np.log(1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma))

def get_grad(h_n,lamda,weight=1):
    grad = np.zeros(2)
    for s_t in range(N_S): # compute the gradient
        a_ba = np.argmax(abs([i for i in range(N_ACT)] - lamda[s_t]))
        a_la = int(lamda[s_t])
        for a_t in range(N_ACT):            
            if a_t != lamda[s_t]:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_la,0] !=0:
                    re0 = (h_n[s_t,a_t,0]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_la,0]/sum(h_n[s_t,a_la,:]))                   
                    if re0<1:
#                        grad[0] += -weight[s_t,a_t] * (gau(a_t,a_la,sigma)-re0) * gau_sig(a_t,a_la,sigma)
                        grad[0] += -(gau(a_t,a_la,sigma)-re0) * gau_sig(a_t,a_la,sigma)
            if a_t != a_ba:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_ba,1] !=0:
                    re1 = (h_n[s_t,a_t,1]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_ba,1]/sum(h_n[s_t,a_ba,:]))                    
                    if re1<1:
#                        grad[1] += weight[s_t,a_t] * (1 - 0.99*gau(a_t,a_la,sigma) - re1) * gau_sig(a_t,a_la,sigma)
                        grad[1] += (1 - 0.99*gau(a_t,a_la,sigma) - re1) * gau_sig(a_t,a_la,sigma)
    return sum(grad)/(N_ACT*N_S)

#-main()-------------------------------------------------
record_all = np.zeros([8,STEP]) # record of data
record_p = np.zeros([8,STEP])
norm_act = np.zeros(8)
record_time = np.zeros([8,STEP])
# for N_ACT in [6]: # np.linspace(2,9,8):
N_ACT = int(N_ACT)
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
    sigma = 3 #np.ones(2)*3 # parameter .5 or 1
    record = np.zeros(STEP)
    lamda = np.random.randint(N_ACT,size=N_S)
    hist = deque() # historical records
    h_n = np.zeros([N_S,N_ACT,3]) # counting the feedback number respect to o,a
    t = 0
    done = 0
    s = 0 #np.random.randint(0,N_S)
    step_t = 0 
    while done != 1:
        start = time.perf_counter()
        a = int(lamda[s])
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

        hist.append((s,a,f))
        if len(hist) > STEP: # limitation of memory
            hist.popleft()
        h = hist.copy()
        h_n[s,a,f] += 1
        lamda = np.random.randint(N_ACT,size=N_S)
        lamda_t = [num+1 if num<N_ACT-1 else 0 for num in lamda]  
        n_l = 0
# update of abluf --------------------------------------------------
        while (lamda != lamda_t).any() and n_l < 3: # EM algorithm for \lambda and \mu
            lamda_t = lamda.copy()
            lamda = EMupdate(h,lamda,sigma,h_n)
            n_l += 1
        if t>0:
            grad = get_grad(h_n,lamda)
            sigma += ALPHA * grad # update of \sigma
        end = time.perf_counter() 
        print(t,':',end-start)
        record_time[N_ACT-2,t] += end-start
        record[t] = result
        if t == 0:
            record_ad[t] = np.abs(a-PI[s])
            record_ad_p[t] = prob[s,a]
        else:
            record_ad[t] = np.abs(a-PI[s])+record_ad[t-1]
            record_ad_p[t] = prob[s,a] + record_ad_p[t-1]
        t=t+1
#            if np.mod(t,20)==0:
#                print(lamda)
        jus[np.mod(t,5)]=(lamda == PI).all()
        done = 1 if t>=STEP or (jus == np.ones(5)).all() else 0 #
        # done = 1 if (jus == np.ones(5)).all() else 0 #
        if done == 1 and (lamda != PI).any(): # and t<STEP:
            num += 1
        if done == 1 and (lamda == PI).all(): # and t<STEP: 
            correct += 1
            num_act += t
            if record_ad[i] == 0 and i > 0:
                record_ad[i] = record_ad[i-1]  
            if record_ad_p[i] == 0 and i>0:
                record_ad_p[i] = record_ad_p[i-1]  
record_tad = record_tad + record_ad
record_tad_p += record_ad_p
norm_act[N_ACT-2] += np.linalg.norm(lamda-PI,2)**2

#---------------------------------------------------------
h_array = np.array(hist)
record_tad = record_tad/(k+1) 
record_p[N_ACT-2] = record_tad_p/(k+1)
norm_act[N_ACT-2] /= (k+1)
record_time = record_time/(k+1)
record_all[N_ACT-2,:] = record_tad  
np.save(r'data\con_record_p',record_all)
np.save(r'data\con_record_all',record_p)
np.save(r'data\con_norm_act',norm_act)


# if __name__ == '__main__':
#     main()
