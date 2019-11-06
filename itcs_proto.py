#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import copy
from tkinter import *
from math import*
from time import*
import random

is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

class Memory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY 
        self.memory = []
        self.index = 0  
        #self.save = 0
    
    def clear(self):
        self.memory = []
        self.index = 0
        
    def save(self,succ, ep_num,trial):
        np.savetxt(str(succ)+"_ep_"+str(ep_num)+"_"+str(trial)+".csv", np.array(self.memory),fmt = "%f", delimiter=",")
    
    def load(self,path):
        self.memory = np.loadtxt("path", delimiter=",").tolist()

    def stock_before(self, state, action):
        if len(self.memory) < self.capacity:
            self.memory.append([])  # 메모리가 가득차지 않은 경우
        else:
            #np.savetxt("log_"+str(self.save)+".csv", np.array(self.memory),fmt = "%f", delimiter=",")
            #self.save += 1
            self.memory = random.sample(self.memory, 5000)

        self.memory[self.index].append(state)
        self.memory[self.index].append(action)
    
    def stock_after(self,state_next,reward):
        self.memory[self.index].append(state_next)
        self.memory[self.index].append(reward)
        
        self.index = (self.index + 1) % self.capacity
        #print(self.memory)
        #print("")
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        """
        input of the layer
        [
             number of cars in this section, lane change stats(digitized),
             x coordinates of cars in this section(digitized), y coordinates of cars in this section(digitized),
             Vx of cars in this section(digitized), Vy coordinates of cars in this section(digitized),
             ax coordinates of cars in this section(digitized), ay coordinates of cars in this section(digitized),
             [map of the unit seciton],
             section 0 ~ 7
        ]
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_mid)
        self.fc5 = nn.Linear(n_mid, n_out)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.atan(self.fc5(x))
        output = torch.tanh(x)
        return (output + 1) / 2

BATCH_SIZE = 32

class Brain:
    def __init__(self,n_in,n_mid,n_out,gamma):
        self.memory = Memory(10000)
        self.model = Net(n_in, n_mid, n_out).double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.gamma = gamma
        if is_cuda:
            self.model = self.model.cuda()
        
    def get_param(self,path):
        self.model.load_state_dict(torch.load(path))
        if is_cuda:
            self.model = self.model.cuda()
        self.model.eval()

    def save_param(self,path):
        if is_cuda:
            torch.save(self.model.cpu().state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        
    def modify_weight(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mini_batch = self.memory.sample(BATCH_SIZE)
        #print(mini_batch)
        state_batch = torch.DoubleTensor(np.array(mini_batch)[:,0].tolist())
        action_batch = torch.DoubleTensor(np.array(mini_batch)[:,1].tolist())
        next_state_batch = torch.DoubleTensor(np.array(mini_batch)[:,2].tolist())
        reward_batch = torch.DoubleTensor(np.array(mini_batch)[:,3].tolist())
        
        if is_cuda:
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            reward_batch = reward_batch.cuda()
        
        self.model.eval()
        state_action_batch = self.model(state_batch)
        next_state_values = self.model(next_state_batch)
        expected_action = reward_batch + self.gamma * next_state_values
        """print("before")
        print(self.model.state_dict())"""
        self.model.train()
        """print("state_action_batch")
        print(state_action_batch)
        print("expected_action")
        print(expected_action)"""
        loss = F.smooth_l1_loss(state_action_batch, expected_action)
        
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()  
        """print("after")
        print(self.model.state_dict())"""
        
    def decide_action(self,state,epsilon):
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state)
            if is_cuda:
                action = action.cpu()
        else:
            action = torch.DoubleTensor([random.uniform(0,1), random.uniform(0,1)])

        return action
        
class Agent:
    def __init__(self):
        self.brain = Brain(19,128,2,0.99)

    def update_q_function(self):
        self.brain.modify_weight()

    def get_action(self,state,epsilon):
        action = self.brain.decide_action(state,epsilon)
        return action
        
    def save(self,succ,ep_num,trial):
        self.brain.memory.save(succ,ep_num,trial)
        
    def memory_clear(self):
        self.brain.memory.clear()

    def memorize_before(self,state,action):
        self.brain.memory.stock_before(state, action)
    
    def memorize_after(self,state_next,reward):
        self.brain.memory.stock_after(state_next,reward)

#timer
def update_time(): 
    global root, elapsedtime, start, stop, timer
    elapsedtime = time() - start
    set_time(elapsedtime)
    timer = root.after(50, update_time)

def set_time(elap):
    global run, timestr
    minutes = int(elap/60)
    seconds = int(elap - minutes*60.0)
    hseconds = int((elap - minutes*60.0 - seconds)*100)
    timestr.set('%02d:%02d:%02d' % (minutes, seconds, hseconds))

def start_time():
    global run, start, elapsedtime
    if not run:
        start = time() - elapsedtime
        update_time()
        run = True

def f_start_time():
    start_time_button.configure(text = "pause", command = lambda:f_stop_time())
    start_time_button.place(x = 1200, y = 100) 
    start_time()

def stop_time():
    global root, elapsedtime, timer, start, run
    set_time(elapsedtime)
    if run:
        root.after_cancel(timer)
        elapsedtime = time() - start
        set_time(elapsedtime)
        run = False

def f_stop_time():
    start_time_button.configure(text = "start", command = lambda:f_start_time())
    start_time_button.place(x = 1200, y = 100)
    stop_time()

def reset_time():
    global start, elapsedtime
    start = time()
    elapsedtime = 0.0
    set_time(elapsedtime)

def get_time(elap):
    minutes = int(elap/60)
    seconds = int(elap - minutes*60.0)
    hseconds = int((elap - minutes*60.0 - seconds)*100)
    
    return minutes, seconds, hseconds

def plc_timer():
    global timer_text
    timer_text = Label(textvariable = timestr, font = ("Arial","40"))
    set_time(elapsedtime)
    timer_text.place(x=1200, y=30)
    start_time_button.place(x=1200,y=100) 
    stop_time_button.place(x=1200,y=200) 
    reset_time_button.place(x=1200,y=150)

class Environment:

    def __init__(self,map_section):
        self.map_vector = map_section
        self.map = np.loadtxt("map.csv", dtype=int, delimiter=",").tolist()
        self.scalar_data = [[0,0,300,280,300],[0,299,317,100,318]]
        self.vector_data = [[0,0,0,0],[0,0,0,0]]
        self.scalar_limit = [4,500,500,500,500]
        self.vector_limit = [70,70,25,25]
        self.noc_data = 2
        self.noc = 2
        self.adj = [-1,-1,-1,-1,-1,-1,-1,-1]
        self.car_scalar = [[0,0,300,280,300],[0,299,317,100,318]]
        self.car_vector = [[0,0,0,0],[0,0,0,0]]
        #self.step = step
        self.dt = 0.25
        self.agent = Agent()
        self.time = 0
        self.timelimit = 480
        self.state = self.get_state()
        self.running = False
        self.complete_trial = 0
    def mk_map(self):
        self.map  = np.full((500,500),5)
        for i in self.map_vector[0]:
            self.map[i[0]:i[2], i[1]] = 0
        for i in self.map_vector[1]:
            self.map[i[0], i[1]:i[3]] = 1
        for i in self.map_vector[2]:
            self.map[i[0], i[1]:i[3]] = 2
        for i in self.map_vector[3]:
            self.map[i[0]:i[2], i[1]] = 3
        for i in self.map_vector[4]:
            if i[1] == i[3]:
                self.map[i[0]:i[2], i[1]] = 4
            else:
                self.map[i[0], i[1]:i[3]] = 4
        for i in self.map_vector[6]:
            self.map[i[0]:i[2], i[1]:i[3]] = 4
        for i in self.map_vector[5]:
            self.map[i[0]:i[2], i[1]:i[3]] = 5
        np.savetxt("map.csv", self.map, fmt='%i', delimiter=",")
    
    def draw(self):
        global can
        can.delete("all")
        plc_timer()
        can.create_rectangle(0,0,1000,1000, fill='#646464')
        for i in self.map_vector[0]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='white')
        for i in self.map_vector[1]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='white')
        for i in self.map_vector[2]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='white')
        for i in self.map_vector[3]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='white')
        for i in self.map_vector[4]:
            if i[1] == i[3]:
                can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='yellow')
            else:
                can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='yellow')
        for i in self.map_vector[6]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2, fill='yellow')
        for i in self.map_vector[5]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2, fill='#646464')

    def draw_map(self):
        global can
        can.create_rectangle(0,0,1000,1000, fill='#646464')
        for i in self.map_vector[0]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='white')
        for i in self.map_vector[1]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='white')
        for i in self.map_vector[2]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='white')
        for i in self.map_vector[3]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='white')
        for i in self.map_vector[4]:
            if i[1] == i[3]:
                can.create_rectangle(i[0]*2, i[1]*2, i[2]*2+1, i[3]*2, fill='yellow')
            else:
                can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2+1, fill='yellow')
        for i in self.map_vector[6]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2, fill='yellow')
        for i in self.map_vector[5]:
            can.create_rectangle(i[0]*2, i[1]*2, i[2]*2, i[3]*2, fill='#646464')
        
    def digitize_scalar(self,car,car_limits):
        car_temp = np.array(car).astype(int)
        result = np.ones(len(car_limits))
        for i in range(len(car_limits)):
            temp = car_temp[:,i]
            max_limit = car_limits[i]
            result[i] = sum([(x) * ((max_limit)**y) for y, x in enumerate(temp)])
        return result
    def digitize_vector(self,car,car_limits):
        car_temp = np.array(car).astype(int)
        result = np.ones(len(car_limits))
        for i in range(len(car_limits)):
            temp = car_temp[:,i]
            max_limit = car_limits[i]
            result[i] = sum([(x+car_limits[i]) * ((car_limits[i]*2)**y) for y, x in enumerate(temp)])
        return result
    def decompose(self,digitized,limit,num):
        temp = np.ones(num)
        for i in range(num):
            temp[i] = digitized % limit
            digitized //= limit
        return temp
    def decompose_scalar(self,digi,limit,noc):
        res = []
        for i in range(len(digi)):
            temp = self.decompose(digi[i],limit[i],noc)
            res.append(temp)
        return np.array(res).transpose()
    def decompose_vector(self,digi,limit,noc):
        res = []
        digi.astype(int)
        for i in range(len(digi)):
            temp = self.decompose(digi[i],limit[i]*2,noc) - limit[0]
            res.append(temp)
        return np.array(res).transpose()

    def get_state(self):
        ds = self.digitize_scalar(self.car_scalar,self.scalar_limit)
        dv = self.digitize_vector(self.car_vector,self.vector_limit)
        concas = np.concatenate((np.array([self.noc]), ds,dv,np.array([self.time]),np.array(self.adj)), axis=None)
        return torch.DoubleTensor(concas.tolist())
    
    def state_next(self,action):
        #print(np.array(action)*((self.vector_limit[2]*2)**(self.noc)))
        action = np.array(action)*((self.vector_limit[2]*2)**(self.noc))
        res = self.decompose_vector(action,self.vector_limit[2:],self.noc)
        #print(res)
        for i in range(self.noc):
            self.car_vector[i][2] += res[i][0]
            self.car_vector[i][3] += res[i][1]
            v_temp = [self.car_vector[i][0], self.car_vector[i][1]]
            self.car_vector[i][0] += self.car_vector[i][2] * self.dt
            self.car_vector[i][1] += self.car_vector[i][3] * self.dt
            self.car_scalar[i][1] += (v_temp[0] + self.car_vector[i][0]) /2 *self.dt
            self.car_scalar[i][2] += (v_temp[1] + self.car_vector[i][1]) /2 *self.dt
    
    def check(self,time):
        if self.noc == 0:
            return True, [1]
        if time == self.timelimit - 1:
            return False, [-1]
        loc = np.array(self.car_scalar)[:,3:].tolist()
        for i in range(len(loc)-1):
            for j in range(i+1,len(loc)):
                if (loc[i][0] == loc[j][0]) and (loc[i][1] == loc[j][1]):
                    #print("collision")
                    return True,[-1]
        #print(self.noc)
        for i in range(self.noc):
            if int(self.car_scalar[i][1]) < 0 or int(self.car_scalar[i][1]) > 500 or int(self.car_scalar[i][2]) < 0 or int(self.car_scalar[i][2]) > 500:
                self.reset_state()
                return False, [-1]
            temp = self.map[int(self.car_scalar[i][1])][int(self.car_scalar[i][2])]
            
            if temp == 4:
                self.car_scalar[i][0] += self.dt
            else:
                self.car_scalar[i][0] = 0
            if temp == 5:
                #print("not on road")
                return True,[-1]
            if self.car_vector[i][0] > self.vector_limit[0] or self.car_vector[i][1] > self.vector_limit[1]:
                #print("over velocity")
                return False,[-1]
            if temp == 0:
                if self.car_vector[i][1] > 0:
                    #print("reverse")
                    return False,[-1]
            elif temp == 1:
                if self.car_vector[i][0] < 0:
                    #print("reverse")
                    return False,[-1]
            elif temp == 2:
                if self.car_vector[i][0] > 0:
                    #print("reverse")
                    return False,[-1]
            elif temp == 3:
                if self.car_vector[i][1] < 0:
                    #print("reverse")
                    return False,[-1]
            
        lane_stat = 0
        des_list = []
        for i in range(self.noc):
            if (loc[i][0] == self.car_scalar[i][3]) and (loc[i][1] == self.car_scalar[i][4]) and self.car_vector[i][0] ==0 and self.car_vector[i][1] == 0:
                des_list.append(i)
                
            if self.car_scalar[i][0] > self.scalar_limit[0]:
                lane_stat = 1
        
        for i in des_list:
            self.car_scalar = np.delete(self.car_scalar, i, axis=0)
            self.car_vector = np.delete(self.car_vector, i, axis=0)
            self.noc -= 1
        
        if lane_stat == 1:
            return False, [-1]
        if self.noc == 0:
            #print("success")
            return True, [1]
        for i in range(self.noc):
            if loc[i][0] == self.scalar_data[i][1] and loc[i][1] == self.scalar_data[i][2]:
                return False, [-1]
        return False, [0]
    
    def reset_state(self):
        global can
        self.car_scalar = copy.deepcopy(self.scalar_data)
        self.car_vector = copy.deepcopy(self.vector_data)
        can.delete("all")
        self.time = 0
        self.state = self.get_state()
        self.complete_trial = 0
        self.noc = copy.deepcopy(self.noc_data)
        self.draw_map()
        self.draw_state()


    def draw_state(self):
        global can
        for car in self.car_scalar:
            can.create_oval(int(car[1])*2, int(car[2])*2, int(car[1])*2+10, int(car[2])*2+10, fill="blue")
        time_text = Label(textvariable = "time    "+str(self.dt * self.time), font = ("Arial","40"))
        time_text.place(x=1050, y=30)
        time_text = Label(textvariable = "success "+str(self.complete_trial), font = ("Arial","40"))
        time_text.place(x=1050, y=80)

    def f_run(self):
        run_button.configure(text = "stop", command = lambda:self.f_stop())
        run_button.place(x = 1050, y = 150)
        self.running = True
        self.run()

    def run(self):
        if self.running == True:
            global can
            can.delete("all")
            self.draw_map()
            if is_cuda:
                self.state = self.state.cuda()
            action = self.agent.get_action(self.state,(0.8 * (1 / (self.complete_trial//2 + 1))))
            self.agent.memorize_before(self.state.cpu().tolist(),action.cpu().tolist())
            self.state_next(action)
            done, reward = self.check(time)
            self.state = self.get_state()
            self.agent.memorize_after(self.state.tolist(),reward)
            
            if done:
                if(reward[0] == 1):
                    #print("ep ",ep_num," tiral ",trial," success in time ",time)
                    #self.agent.save("f_",ep_num,trial)
                    self.complete_trial += 1
                """else:
                    #print("ep ",ep_num," tiral ",trial," failed in time ",time)
                    #self.agent.save("s_",ep_num,trial)
                    self.complete_trial = 0"""
                #self.agent.memory_clear()
                self.reset_state()
            self.draw_state()
            self.agent.update_q_function()
            self.time += 1

            self.process = root.after(50, self.run)
    
    def f_stop(self):
        run_button.configure(text = "run", command = lambda:self.f_run())
        run_button.place(x = 1050, y = 150)
        self.stop()

    def stop(self):
        global root
        self.running = False
        root.after_cancel(self.process)


def quit():
    env.agent.brain.save_param("params/param")
    root.destroy()

#############################################
map = [np.loadtxt("map0.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map1.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map2.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map3.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map4.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map5.csv", dtype=int, delimiter=",").tolist(),
       np.loadtxt("map6.csv", dtype=int, delimiter=",").tolist(),
      ]
adj = [-1,-1,-1,-1,-1,-1,-1,-1]
car_limit = [[4,500,500,500,500],[70,70,25,25]]
ep = [[[[0,0,300,280,300]],[[0,0,0,0]]],[[[0,299,317,100,318]],[[0,0,0,0]]],[[[0,0,300,280,300],[0,299,317,100,318]],[[0,0,0,0],[0,0,0,0]]]]
step = [[500,200],[240,200],[240,200]]

            
##############################################
env = Environment(map)
env.agent.brain.get_param("params/param")
##############################################

#initialize canvas
root=Tk()
root.title("Intelligent_Transport_Control_System")
root.geometry('1200x1000')
can=Canvas(root, bg="white", width = 1500,  height = 1000)
can.place(x=0, y=0)

#buttons
run_button = Button(text = "run", font = ("Arial", "16"), command = lambda:env.f_run())
quit_button = Button(text = "quit", font = ("Arial", "16"), command = lambda:quit())
run_button.place(x = 1050, y = 150)
quit_button.place(x = 1400, y = 900)

###############################################
#time variables
"""start = 0.0
elapsedtime = 0.0
timer=0.0
run = False
timestr = StringVar()
start_time_button = Button(text='Start', command = lambda:f_start_time())
stop_time_button = Button(text='Stop', command = lambda:stop_time())
reset_time_button = Button(text='Reset', command = lambda:reset_time())"""




###############################################

env.draw_map()
env.draw_state()
"""for i in [10,100,150]:
    can.create_oval(i,i,i+2,i+2, fill="blue")"""

root.mainloop()