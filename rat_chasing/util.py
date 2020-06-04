# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate
import random
import pickle
import pylab
import matplotlib.pyplot as plt
import cv2
import time

def get_p(PI,N_S,N_ACT):
    p_act = np.zeros([N_S,N_ACT])
    for j in range(N_S):
        p_tmp = np.zeros(N_ACT)
        for i in range(N_ACT):
            p_tmp[i] = gau(i,PI[j],1)
        p_act[j] = p_tmp/sum(p_tmp)
    return p_act

def get_action(p_act_s,N_ACT):
    tmp = 0
    ran = np.random.random()
    for i in range(N_ACT):
        tmp += p_act_s[i]
        if ran <= tmp:
            return i
            break

def gau(x,mu,sigma):
    g = np.exp(-(x-mu)**2/(2*sigma**2))  #1/(sigma*np.sqrt(2*np.pi))*
    return g

def bound(current_action,size,limit,action):
    action_point = np.round(np.linspace(0,limit,action)[current_action])
    down = int(action_point-size/2)
    up = int(action_point+size/2)
    if down < 0:
        down = 0
        up = size
    elif up>limit:
        down = limit-size
        up = limit
    return down,up

def show_image(state,rat_action,dog_action,dog_size,rat_size,length,N_ACT,PI):
    action = N_ACT
    img_background = cv2.imread('grass.jpg') #plt.imread
    img_dog = cv2.imread('dog_w.png')
    img_rat = cv2.imread('rat.png')
    img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
    img_background = cv2.cvtColor(img_background, cv2.COLOR_BGR2RGB)
    img_rat = cv2.cvtColor(img_rat, cv2.COLOR_BGR2RGB)
    img_background = cv2.resize(img_background, (length, length))
    img_dog = cv2.resize(img_dog, (dog_size, dog_size)) #, fx=0.5, fy=0.5)
    img_rat = cv2.resize(img_rat, (rat_size, rat_size))
    
    dog_half = dog_size/2
    down = int(length/2-dog_half)
    up = int(length/2+dog_half)
    
    initial_state = img_background.copy()
    initial_state[down:up,down:up] = img_dog
    img_state = initial_state.copy()
    action_state = img_background.copy()
    # rat_action = np.random.randint(0,action)
    # dog_action = np.random.randint(0,action)
    rat_down, rat_up =  bound(rat_action,rat_size,length,action)
    dog_down, dog_up = bound(dog_action,dog_size,length,action)
    
    if state == 0: #down
        img_state[int(length-rat_size):length, int(length/2-rat_size/2):int(length/2+rat_size/2)] = img_rat      
        action_state[int(length-dog_size):length, dog_down:dog_up] = img_dog
        action_state[int(length-rat_size):length, rat_down:rat_up] = img_rat
    elif state == 1: #up
        img_state[0:rat_size, int(length/2-rat_size/2):int(length/2+rat_size/2)] = img_rat
        action_state[0:dog_size, dog_down:dog_up] = img_dog
        action_state[0:rat_size, rat_down:rat_up] = img_rat
    elif state == 2: #left
        img_state[int(length/2-rat_size/2):int(length/2+rat_size/2), 0:rat_size] = img_rat
        action_state[dog_down:dog_up, 0:dog_size] = img_dog
        action_state[rat_down:rat_up, 0:rat_size] = img_rat
    else: #right
        img_state[int(length/2-rat_size/2):int(length/2+rat_size/2), int(length-rat_size):length] = img_rat
        action_state[dog_down:dog_up, int(length-dog_size):length] = img_dog
        action_state[rat_down:rat_up, int(length-rat_size):length] = img_rat

    plt.figure()
    plt.xticks(np.linspace(0,length,action),np.arange(action))
    plt.yticks(np.linspace(0,length,action),np.arange(action))
    plt.title('State')
    plt.imshow(img_state)
#    plt.draw()
    plt.show(block=False)
    time.sleep(2)
#    plt.clf()
    plt.close('all')

    plt.figure()
    plt.xticks(np.linspace(0,length,action),np.arange(action))
    plt.yticks(np.linspace(0,length,action),np.arange(action))
    plt.title('Action')
    plt.imshow(action_state)
#    plt.draw()
    plt.show(block=False)
    user_input = input('input(good:0, bad:1, no_feedback: 2, next state: 3):')
    while user_input not in ['0','1','2','3']:
        print('please input feedback')
        user_input = input('input(good:0, bad:1, no_feedback: 2, next state: 3):')
    user_input = int(user_input)
#    plt.clf()
    plt.close('all')
    result = 1 if dog_action == PI[state] else 0
    return user_input,result