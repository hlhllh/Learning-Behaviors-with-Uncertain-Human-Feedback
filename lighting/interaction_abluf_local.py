# -*- coding: utf-8 -*-
"""
ABLUF
"""

import remi.gui as gui
from remi import start, App
import os
import numpy as np
import scipy.integrate as integrate
import scipy as sp
import random
from collections import deque
from yeelight import Bulb

N_ACT=4 #number of action 
N_S=3 # number of state
E = 0.1
ALPHA = 0.8

def randomPolicy():
    return np.array(np.random.randint(N_ACT, size=N_S)) # up right down left 0 1 2 3

def gau(x,mu,sigma):
    g = np.exp(-(x-mu)**2/(2*sigma**2))  #1/(sigma*np.sqrt(2*np.pi))*
    return g

def gau_sig(x,mu,sigma):
    g_div = ((x-mu)**2) * np.exp(-(x-mu)**2/(2*sigma**2)) #/sigma**3 ignore sigma**3 to increase the step size
    return g_div    
            
def EMupdate(h,lamda,sigma,h_n,state):
    temp = np.zeros(N_ACT)
#    a = np.zeros(N_S)
    a = lamda.copy()
#    if len(h_t)>10:
#        h = np.array(h_t)[-10:]
#    else:
#        h = h_t
#    for o in range(N_S):
    for o in [state]:
        for i in range(N_ACT):
            temp[i] = sp.integrate.dblquad(fun1, 0, 1, lambda x: 1-x, lambda x: 1, args=(h,lamda,i,o,sigma,h_n))[0]
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
    return gau(a_o,a,sigma[0])*(1-x)
    
def fun_ne(a_o,a,y,sigma):
    return (1-gau(a_o,a,sigma[1])*0.99)*(1-y)

def fun_0(a_o,a,x,y,sigma):
    return 1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma)

def fun_po_log(a_o,a,x,sigma):        
    return np.log(gau(a_o,a,sigma[0]))#+np.log(1-x)
    
def fun_ne_log(a_o,a,y,sigma):
    return np.log(1-gau(a_o,a,sigma[1])*0.99)#+np.log(1-y)

def fun_0_log(a_o,a,x,y,sigma):
    return np.log(1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma))

def get_grad(h_n,lamda,sigma,weight=1):
    grad = np.zeros(2)
    for s_t in range(N_S): # compute the gradient
        a_ba = np.argmax(abs([i for i in range(N_ACT)] - lamda[s_t]))
        a_la = int(lamda[s_t])
        for a_t in range(N_ACT):            
            if a_t != lamda[s_t]:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_la,0] !=0:
                    re0 = (h_n[s_t,a_t,0]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_la,0]/sum(h_n[s_t,a_la,:]))                   
                    if re0<1:
#                        grad[0] += -weight[s_t,a_t] * (gau(a_t,a_la,sigma[0])-re0) * gau_sig(a_t,a_la,sigma[0])
                        grad[0] += -(gau(a_t,a_la,sigma[0])-re0) * gau_sig(a_t,a_la,sigma[0])
            if a_t != a_ba:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_ba,1] !=0:
                    re1 = (h_n[s_t,a_t,1]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_ba,1]/sum(h_n[s_t,a_ba,:]))                    
                    if re1<1:
#                        grad[1] += weight[s_t,a_t] * (1 - 0.99*gau(a_t,a_la,sigma[1]) - re1) * gau_sig(a_t,a_la,sigma[1])
                        grad[1] += (1 - 0.99*gau(a_t,a_la,sigma[1]) - re1) * gau_sig(a_t,a_la,sigma[1])
    return grad/(N_ACT*N_S)

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
    hist = deque()
#    bulb = Bulb("192.168.1.100")
#    bulb.turn_on()
#    bulb.set_color_temp(4000)
    
    
    def main(self):
        # build UI
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
        
        self.sigma = np.ones(2)*3
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
        # ABLUF
        if reward == 4: # next state
            self.state += 1
            if self.state+1 > 3:
                self.close()
    #            self.state = 0
#                self.bulb.turn_off()
            else:
                self.iterations = 0
#                self.lbl.set_text("Wait!")
                self.a = 2 #int(random.randint(0,N_ACT-1)) #choose an arm randomly
                self.take_action(self.a)
                self.record_a[self.state,self.iterations] = self.a
                self.record_n[self.state,self.a] += 1
                self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
        else: # update
            r = reward
            self.hist.append((self.state,self.a,r))
            h = self.hist.copy()
            self.h_n[self.state,self.a,r] += 1
            lamda_t = [num+1 if num<N_ACT-1 else 0 for num in self.lamda]  
            n_l = 0
            while (self.lamda != lamda_t).any() and n_l < 3: # EM algorithm
                lamda_t = self.lamda.copy()
                self.lamda = EMupdate(h,self.lamda,self.sigma,self.h_n,self.state)
                n_l += 1
            if self.iterations>0:
                grad = get_grad(self.h_n,self.lamda,self.sigma)
                self.sigma += ALPHA * grad #* sigma
            self.a = int(self.lamda[self.state])
            self.take_action(self.a)
            self.record_n[self.state,self.a] += 1
            self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
            self.iterations = self.iterations + 1
            self.record_a[self.state,self.iterations] = self.a
            


if __name__ == "__main__":
    # starts the webserver
    app=MyApp
    start(app, address='127.0.0.1', port=8083, start_browser=True, username=None, password=None)
    record_con_n = app.record_n
    record_con_a = app.record_a
    record_con_h = np.array(app.hist)
    record_con_ad = record_con_a.copy()
    for i in range(N_S):
        for j in range(100):
            if record_con_ad[i,j] == -1:
                record_con_ad[i,j] = record_con_ad[i,j-1]
    for i in range(100):
        if i == 0:
            record_con_ad[:,i] = abs(record_con_ad[:,i] - record_con_ad[:,99])
        else:
            record_con_ad[:,i] = record_con_ad[:,i-1]+ abs(record_con_ad[:,i] - record_con_ad[:,99])
    record_sum = sum(record_con_ad[:,99])
            
    
    