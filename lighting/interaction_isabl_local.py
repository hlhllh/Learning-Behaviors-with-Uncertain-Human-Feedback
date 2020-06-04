# -*- coding: utf-8 -*-
"""
isabl
"""

import remi.gui as gui
from remi import start, App
import os
import numpy as np
import scipy.integrate as integrate
import random
from collections import deque
from yeelight import Bulb

N_ACT=4
N_S=3
E = 0.1

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

def current_s(state):
    if state == 0:
        return 'reading'
    elif state == 1:
        return 'watching'
    else:
        return 'chatting'


class MyApp(App):
    record_n = np.zeros([N_S,N_ACT])
    record_a = np.zeros([N_S,100])-1
#    bulb = Bulb("192.168.1.100")
#    bulb.turn_on()
#    bulb.set_color_temp(4000)
    h = deque()
    
    def main(self):
        #Build UI
        main_container = gui.VBox(style={'margin':'0px auto'})
        self.lbl_result = gui.Label("")
        self.button_0 = gui.Button("Positive", width=200, height=30, margin='10px')
        self.button_0.set_on_click_listener(self.on_button_0)
        self.button_1 = gui.Button("Negative", width=200, height=30, margin='10px')
        self.button_1.set_on_click_listener(self.on_button_1)
        self.button_2 = gui.Button("No Feedback", width=200, height=30, margin='10px')
        self.button_2.set_on_click_listener(self.on_button_2)
        self.button_3 = gui.Button("Done", width=200, height=30, margin='10px')
        self.button_3.set_on_click_listener(self.on_button_3)
        self.lbl = gui.Label("Compelete!")
        main_container.append([self.lbl_result, self.button_0, self.button_1, self.button_2, self.button_3,self.lbl] )

        self.P_a = [0.5,0.8] #the probability of getting reward for a_0 and a_1 respectively
        self.h_n = np.zeros([N_S,N_ACT,3])
        self.p = np.ones([N_S,N_ACT]) * 1/N_ACT # initial probability that choosing different as       
#        self.record_n = np.zeros(2)
        self.a = int(random.randint(0,N_ACT-1)) #choose an arm randomly
        self.take_action(self.a)
        self.state = 0
        self.record_n[self.state,self.a] = 1
        self.lamda = np.random.randint(N_ACT,size=N_S)
#        print(self.record_n)
        self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
        self.iterations = 0
        self.record_a[self.state,self.iterations] = self.a
        return main_container

    def on_button_0(self, emitter):
        self.algorithm(0) #positive
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_1(self, emitter):
        self.algorithm(1) #negative
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    
    def on_button_2(self, emitter):
        self.algorithm(2) #no feedback
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_3(self, emitter):
        self.algorithm(4)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def take_action(self,action):
        # controll the light
#        self.bulb.set_brightness(int(action*33) if action != 0 else 1)
        return
    
    def algorithm(self, reward):
        if reward == 4: #next state
            self.state += 1
            if self.state+1 > 3:
                self.close()
    #            self.state = 0
#                self.bulb.turn_off()
            else:
                self.iterations = 0
                self.a = 2 #int(random.randint(0,N_ACT-1)) #choose an arm randomly
                self.take_action(self.a)
                self.record_a[self.state,self.iterations] = self.a
                self.record_n[self.state,self.a] += 1
                self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
        else: # update
            r = reward
            
            self.h.append((self.state,self.a,r))
            self.h_n[self.state,self.a,r] += 1
            self.lamda = randomPolicy()
            lamda_t = np.zeros([N_S,1])
            a_p = integrate.dblquad(fun1 , 0, 1, lambda y: 0, lambda y: 1, args=(self.h,self.lamda))[0]
            a_p = a_p*np.log((1-E)/E) # calculate parameters
            b_p = integrate.dblquad(fun2, 0, 1, lambda y: 0, lambda y: 1, args=(self.h,self.lamda))[0]
            while (self.lamda != lamda_t).any(): # EM algorithm
                lamda_t = self.lamda.copy()
                self.lamda = EMupdate(a_p,b_p,self.h_n)
            
            self.a = int(self.lamda[self.state])
            self.take_action(self.a)
            self.record_n[self.state,self.a] += 1
            self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
            self.iterations = self.iterations + 1
            self.record_a[self.state,self.iterations] = self.a


if __name__ == "__main__":
    # starts the webserver
    app=MyApp
    start(app, address='127.0.0.1', port=8082, start_browser=True, username=None, password=None)
    record_isabl_h = np.array(app.h)
    record_isabl_n = app.record_n
    record_isabl_a = app.record_a
    record_isabl_ad = record_isabl_a.copy()
    for i in range(N_S):
        for j in range(100):
            if record_isabl_ad[i,j] == -1:
                record_isabl_ad[i,j] = record_isabl_ad[i,j-1]
    for i in range(100):
        if i == 0:
            record_isabl_ad[:,i] = abs(record_isabl_ad[:,i] - record_isabl_ad[:,99])
        else:
            record_isabl_ad[:,i] = record_isabl_ad[:,i-1]+ abs(record_isabl_ad[:,i] - record_isabl_ad[:,99])
    record_sum = sum(record_isabl_ad[:,99])
            
    
    