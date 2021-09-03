#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import get_model_dir, write_mat
import __main__ as main
import os
import csv
from std_msgs.msg import Float64MultiArray
torch.manual_seed(1)
import rospy
from math import sin,cos
import time
import random

# NN Model
class MyModel(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.dev = dev
        self.linear_1 = nn.Linear(2, 10)
        self.linear_2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.tanh(x)
        x = self.linear_2(x)
        return x

# VREP ROBOT ENV
class ENV:
    def __init__(self):
        # /MPC_solutions topis provides joint positions, and MPC solutions
        rospy.Subscriber("/MPC_solutions", Float64MultiArray, self.callback)
        self.NN_solutions = Float64MultiArray()
        # /pr/command sends velocities to Planar Robot
        self.pub = rospy.Publisher('/pr/command', Float64MultiArray, queue_size=1)
        self.init_poses = [0, 0]
        self.start()
        
    def callback(self, data):
        # data has joint positions and MPC solutions
        self.observation = data.data[0:4]

    # Feasibility check
    def config_space(self, theta_1, theta_2):
        l1 = 0.5
        l2 = 0.4
        R_S = 0.2
        R_quad = (l2/8+R_S)*(l2/8+R_S)-0.001
        s1 = sin(theta_1)
        c1 = cos(theta_1)
        s12 = sin(theta_1+theta_2)
        c12 = cos(theta_1+theta_2)
        O11 = -0.6
        O12 = 0.7
        O21 = 0.6
        O22 = 0.7
        x1 = l1*s1+0.1
        x2 = l1*s1+l2*s12+0.1
        t1 = (l1*c1+1*l2*c12/8-O11)*(l1*c1+1*l2*c12/8-O11)+(l1*s1+1*l2*s12/8-O12)*(l1*s1+1*l2*s12/8-O12)-R_quad
        t2 = (l1*c1+3*l2*c12/8-O11)*(l1*c1+3*l2*c12/8-O11)+(l1*s1+3*l2*s12/8-O12)*(l1*s1+3*l2*s12/8-O12)-R_quad
        t3 = (l1*c1+5*l2*c12/8-O11)*(l1*c1+5*l2*c12/8-O11)+(l1*s1+5*l2*s12/8-O12)*(l1*s1+5*l2*s12/8-O12)-R_quad
        t4 = (l1*c1+7*l2*c12/8-O11)*(l1*c1+7*l2*c12/8-O11)+(l1*s1+7*l2*s12/8-O12)*(l1*s1+7*l2*s12/8-O12)-R_quad
        t5 = (l1*c1+1*l2*c12/8-O21)*(l1*c1+1*l2*c12/8-O21)+(l1*s1+1*l2*s12/8-O22)*(l1*s1+1*l2*s12/8-O22)-R_quad
        t6 = (l1*c1+3*l2*c12/8-O21)*(l1*c1+3*l2*c12/8-O21)+(l1*s1+3*l2*s12/8-O22)*(l1*s1+3*l2*s12/8-O22)-R_quad
        t7 = (l1*c1+5*l2*c12/8-O21)*(l1*c1+5*l2*c12/8-O21)+(l1*s1+5*l2*s12/8-O22)*(l1*s1+5*l2*s12/8-O22)-R_quad
        t8 = (l1*c1+7*l2*c12/8-O21)*(l1*c1+7*l2*c12/8-O21)+(l1*s1+7*l2*s12/8-O22)*(l1*s1+7*l2*s12/8-O22)-R_quad
        answer = 0
        # If the given joint positions satisfy all above, it returns 1
        if (x1>0 and x2>0 and t1>0 and t2>0 and t3>0 and t4>0 and t5>0 and t6>0 and t7>0 and t8>0):
            answer = 1
        return answer

    def start(self):
        self.act_min = -np.array([1, 1])
        self.act_max = np.array([1, 1])

    def step(self, u):
        # The last value for VREP
        u = [u[0],u[1],1]
        self.NN_solutions.data = u
        self.pub.publish(self.NN_solutions)
        obs = self.observe()
        return obs
    
    def observe(self):
        return self.observation

    def reset(self, theta_1=100, theta_2=100, t = 5):   
        self.init_variables(theta_1, theta_2)
        
        # WE need some time for putting the robot on its initial position
        time.sleep(t)
        obs = self.observe()
        return obs 
    
    def init_variables(self, theta_1, theta_2):
        answer = 0
        if theta_1==100 and theta_2==100:
            while (answer<1):
                self.init_poses[0] = random.uniform(0.0, 0.1)
                self.init_poses[1] = random.uniform(0.0, 0.1)
                answer = self.config_space(self.init_poses[0],self.init_poses[1])
        else:
            self.init_poses[0] = theta_1
            self.init_poses[1] = theta_2

        print("Init pose: ", self.init_poses[0],self.init_poses[1]) 
        self.NN_solutions.data = [self.init_poses[0],self.init_poses[1],0]
        self.pub.publish(self.NN_solutions)

def save_log(run_name, actions, real_jp, real_jv, loss, i):
    write_mat('Network_log/' + run_name,
                    {'actions':actions,
                    'real_jp': real_jp,
                    'real_jv': real_jv,
                    'loss': loss},
                    str(i))    

def train(dev, model, x_train, y_train, optimizer, log_interval, loss_function):
    runLoss = 0
    record_loss = 0
    model.train()
    for b in range(0, len(x_train), n_batch):
        seq_data = np.array(x_train[b:b+n_batch])
        seq_label = np.array(y_train[b:b+n_batch])
        seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
        seq_label = torch.tensor([i for i in seq_label], dtype=torch.float32).to(dev)
        optimizer.zero_grad()
        y_pred = model(seq_data)
        single_loss = loss_function(y_pred, seq_label)
        runLoss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        if b % log_interval == 0:
            record_loss = single_loss.item()
            print ('Train epoch [{}/{}] loss: {:.6f}'.format(b, len(x_train), record_loss))

    return runLoss

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    env = ENV()
    model_dir = get_model_dir()
    run_name = model_dir.split('/')[1]
    episodes = 10000
    n_batch = 100
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(dev).to(dev)
    model.cuda()
    model.load_state_dict(torch.load('weights/model_dagger_20210819_154210_1901.pth'))
    # model = MyModel(dev).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    nn_jv = []
    real_jp = []
    real_jv = []
    nSteps = 400
    log_interval = 200
    loss = 0
    
    observation = env.reset()
    seq_label = observation[0:2]
    seq_data = observation[2:4]

    # Exploring with NN:
    for i in range(episodes):
        print("Episode", i+1, " is started")
        loss = 0
        observation = env.reset()
        seq_label = observation[0:2]
        seq_data = observation[2:4]
        back_iter = 1
        nSteps = 400
        temp_jp = [[0 for c in range(2)] for r in range(nSteps)]
        temp_jv = [[0 for c in range(2)] for r in range(nSteps)]
        temp_nn_jv = [[0 for c in range(2)] for r in range(nSteps)]
        t = time.time()
        for b in range(nSteps):
            
            seq_data_torch = torch.tensor(seq_data, dtype=torch.float32).to(dev)

            # Prediction
            y_pred = model(seq_data_torch)
            pred = y_pred.cpu()
            pred = pred.detach().numpy()

            # Send action
            observation = env.step(pred)
            temp_nn_jv[b] = pred
            temp_jp[b] = seq_data
            temp_jv[b] = seq_label

            # New data collection
            seq_data = observation[2:4]
            seq_label = observation[0:2]

            time.sleep(0.05)

        for z in range(nSteps):
            real_jp.append(temp_jp[z])
            real_jv.append(temp_jv[z])
            nn_jv.append(temp_nn_jv[z])
    
        print("---Training is started---")
        loss = train(dev, model, real_jp, real_jv, optimizer, log_interval, loss_function)
        elapsed = time.time() - t
        
        print("Episode ", i+1, " runloss = ", loss, ' time = ', elapsed)
        save_log(run_name, temp_nn_jv, temp_jp, temp_jv, loss, i+1)

        if i % 100 == 0:
            if not os.path.exists('weights'):
                os.makedirs('weights')
            torch.save(model.state_dict(), 'weights/model_{}_{}.pth'.format(run_name, i+1))

