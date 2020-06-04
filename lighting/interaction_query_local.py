# -*- coding: utf-8 -*-
"""
query
"""

import remi.gui as gui
from remi import start, App
import os
from collections import deque
import numpy as np
import random
from yeelight import Bulb
N_ACT=4
N_S=3

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
    h = deque()
#    bulb = Bulb("192.168.1.100")
#    bulb.turn_on()
#    bulb.set_color_temp(4000)
    
    def main(self):
        #build UI for query
        main_container = gui.VBox(style={'margin':'0px auto'})
        main_container_1 = gui.HBox(style={'margin':'0px auto'})
        sub_container_1 = gui.VBox(style={'margin':'0px auto'})
        sub_container_2 = gui.VBox(style={'margin':'0px auto'})
        self.lbl_result = gui.Label("")
        self.button_0 = gui.Button("0", width=200, height=30, margin='10px')
        self.button_0.set_on_click_listener(self.on_button_0)
        self.button_1 = gui.Button("1", width=200, height=30, margin='10px')
        self.button_1.set_on_click_listener(self.on_button_1)
        self.button_2 = gui.Button("2", width=200, height=30, margin='10px')
        self.button_2.set_on_click_listener(self.on_button_2)
        self.button_3 = gui.Button("3", width=200, height=30, margin='10px')
        self.button_3.set_on_click_listener(self.on_button_3)
        self.button_4 = gui.Button("4", width=200, height=30, margin='10px')
        self.button_4.set_on_click_listener(self.on_button_4)
        self.button_5 = gui.Button("5", width=200, height=30, margin='10px')
        self.button_5.set_on_click_listener(self.on_button_5)
        self.button_6 = gui.Button("6", width=200, height=30, margin='10px')
        self.button_6.set_on_click_listener(self.on_button_6)
        self.button_7 = gui.Button("7", width=200, height=30, margin='10px')
        self.button_7.set_on_click_listener(self.on_button_7)
        self.button_8 = gui.Button("8", width=200, height=30, margin='10px')
        self.button_8.set_on_click_listener(self.on_button_8)
        self.button_9 = gui.Button("9", width=200, height=30, margin='10px')
        self.button_9.set_on_click_listener(self.on_button_9)
        self.button_10 = gui.Button("Done", width=200, height=30, margin='10px')
        self.button_10.set_on_click_listener(self.on_button_10)
        
        self.lbl = gui.Label("Compelete!")
        sub_container_1.append([self.button_0, self.button_1, self.button_2, 
                                self.button_3, self.button_4])
        sub_container_2.append([self.button_5,self.button_6, self.button_7, self.button_8, 
                                self.button_9])
        main_container_1.append([sub_container_1, sub_container_2] ) 
        main_container.append([self.lbl_result, main_container_1, self.button_10, self.lbl] ) 

        self.P_a = [0.5,0.8] #the probability of getting reward for a_0 and a_1 respectively
        self.p = np.ones([N_S,N_ACT]) * 1/N_ACT # initial probability that choosing different as       
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
        self.algorithm(0)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_1(self, emitter):
        self.algorithm(1)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    
    def on_button_2(self, emitter):
        self.algorithm(2)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_3(self, emitter):
        self.algorithm(3)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_4(self, emitter):
        self.algorithm(4)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    
    def on_button_5(self, emitter):
        self.algorithm(5)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    
    def on_button_6(self, emitter):
        self.algorithm(6)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_7(self, emitter):
        self.algorithm(7)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    def on_button_8(self, emitter):
        self.algorithm(8)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
        
    def on_button_9(self, emitter):
        self.algorithm(9)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))    
        
    def on_button_10(self, emitter):
        self.algorithm(10)
        self.lbl.set_text("Compelete! %s" %str(self.iterations))
    
    def take_action(self,action):
        # controll the bulb
#        self.bulb.set_brightness(int(action*33) if action != 0 else 1)
        return
    
    def algorithm(self, reward):
        if reward == 10:
            self.state += 1
            if self.state+1 > 3:
                self.close()
    #            self.state = 0
#                self.bulb.turn_off()
            else:
                self.iterations = 0
                self.a = int(random.randint(0,N_ACT-1)) #choose an arm randomly
                self.take_action(self.a)
                self.record_a[self.state,self.iterations] = self.a
                self.record_n[self.state,self.a] += 1
                self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
        else: # change light level according to users' choices
            r = reward
            self.h.append((self.state,self.a,r))
            
            self.a = reward
            self.take_action(self.a)
            self.record_n[self.state,self.a] += 1
            self.lbl_result.set_text( "The light is %s, the state is %s" %(str(self.a),str(current_s(self.state))))
            self.iterations = self.iterations + 1
            self.record_a[self.state,self.iterations] = self.a
#        if self.state+1 > 3:
#            self.close()
#            self.bulb.turn_off()


if __name__ == "__main__":
    # starts the webserver
    app=MyApp
    start(app, address='127.0.0.1', port=8081, start_browser=True, username=None, password=None)
    record_ucb_h = np.array(app.h)
    record_ucb_n = app.record_n
    record_ucb_a = app.record_a
    record_ucb_ad = record_ucb_a.copy()
    for i in range(N_S):
        for j in range(100):
            if record_ucb_ad[i,j] == -1:
                record_ucb_ad[i,j] = record_ucb_ad[i,j-1]
    for i in range(100):
        if i == 0:
            record_ucb_ad[:,i] = abs(record_ucb_ad[:,i] - record_ucb_ad[:,99])
        else:
            record_ucb_ad[:,i] = record_ucb_ad[:,i-1]+ abs(record_ucb_ad[:,i] - record_ucb_ad[:,99])
    record_sum = sum(record_ucb_ad[:,99])
            
    
    


