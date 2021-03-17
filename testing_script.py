# https://github.com/keras-team/keras-io/blob/master/examples/rl/deep_q_network_breakout.py

import sys

import os
import sys
import glob
import numpy as np
from math import sin, cos
import math
import random
import copy
import random
from math import cos, sin
import keras
import time

# EXPLOITATION VS EXPLORATATION
# POLICY NETWORK
# vALUE NETWORK
# REWARD CALCULATION
# DIFFERNT TARGET
##replay memory
# from keras import layers
# from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten, Dense
# from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.optimizers import Adam
import tensorflow as tf

optimizer = keras.optimizers.Adam(learning_rate=0.0025)
eps_decay = 0.1
NUM_ACTIONS = 6
MOVE_INTERVAL = 0.2
ROTATE_INTERVAL = math.radians(10)
MAX_LENGTH = 10
BATCH_SIZE = 32
UPDATE_INTERVAL = 100
MEMORY_SIZE = 1000
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.3 # Minimum epsilon greedy parameter
epsilon_max = 1.0
max_steps_per_episode =200

loss_function = keras.losses.Huber()


def create_q_model():
    model = Sequential()
    inputs = Input(shape=(MAX_LENGTH, MAX_LENGTH, 1))
    model.add(inputs)
    model.add(
        Conv2D(filters=32,
               kernel_size=(1, 1),
               activation="relu",
               kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(
        Conv2D(filters=64,
               kernel_size=(1, 1),
               activation="relu",
               kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(NUM_ACTIONS, activation="softmax"))
    return model


def action_up(_cords):
    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.y_cord = float(_xy.y_cord) + MOVE_INTERVAL
    return temps


def action_down(_cords):
    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.y_cord = float(_xy.y_cord) - MOVE_INTERVAL
    return temps


def action_left(_cords):
    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cord = float(_xy.x_cord) - MOVE_INTERVAL
    return temps


def action_right(_cords):
    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cord = float(_xy.x_cord) + MOVE_INTERVAL
    return temps



# https://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
def action_rotate_xy(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cord = _xy.x_cord * c + _xy.y_cord * s
        _xy.y_cord = - 1 * _xy.x_cord * s + _xy.y_cord * c
    return temps


def get_rmsd(_false_dist, _real_dist, _dim):
    real_distance_map = np.sum((_false_dist - _real_dist) ** 2) / (_dim)

    return float(real_distance_map ** 0.5)


def action_rotate_yx(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cord = _xy.x_cord * c - _xy.y_cord * s
        _xy.y_cord = _xy.x_cord * s + _xy.y_cord * c
    return temps
# new_Q - old_Q
def get_rewards(_false_dist, _real_dist, _dim):
    tots = 1 / (1 + get_rmsd(_false_dist, _real_dist, _dim))
    if tots > 0.75:
        return 100
    else:
        return 10*tots

def get_rewards_2(_agent_1, _dim):
    global best_rmsd
    rmsd = ( (( _agent_1[0].x_cord-6.5 )**2)+ (( _agent_1[1].x_cord-6.5 )**2)+ (( _agent_1[0].y_cord-4)**2)+ (( _agent_1[1].x_cord-5.5)**2))**0.5

    # print( "tots "+str(tots)+"\n")
    # print("rmsd "+str( get_rmsd(_false_dist, _real_dist, _dim))+"\n")
    best_rmsd =rmsd
    if rmsd < 1.0:
        return 100

    return rmsd

def action_rotate_yx_1(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    #EXPERIMENTAL
    temps = copy.deepcopy(_cords)
    tot_x= 0
    tot_y=0
    for val in _cords:
        tot_x+=float(val.x_cord)
        tot_y += float(val.y_cord)
    tot_x =tot_x/len(_cords)
    tot_y = tot_y / len(_cords)
    for _xy in temps:
        #(cos(angle) * (p.x - cx) - sin(angle) * (p.y - cy) + cx,
        _xy.x_cord =( _xy.x_cord - tot_x) * c - (_xy.y_cord-tot_y) * s+tot_x
        _xy.y_cord =( _xy.x_cord - tot_x) * s + (_xy.y_cord-tot_y) * c+tot_y
    return temps



def action_rotate_xy_1(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    #EXPERIMENTAL
    temps = copy.deepcopy(_cords)
    tot_x= 0
    tot_y=0
    for val in _cords:
        tot_x+=float(val.x_cord)
        tot_y += float(val.y_cord)
    tot_x =tot_x/len(_cords)
    tot_y = tot_y / len(_cords)
    for _xy in temps:
        #(cos(angle) * (p.x - cx) - sin(angle) * (p.y - cy) + cx,


        _xy.x_cord =( _xy.x_cord - tot_x) * c + (_xy.y_cord-tot_y)  * s +tot_x
        _xy.y_cord = - 1 *( _xy.x_cord - tot_x) * s + (_xy.y_cord-tot_y)* c +tot_y

    return temps

def distance_calculation(_cord_1, _cord_2):
    return (((_cord_1.x_cord - _cord_2.x_cord) ** 2) + ((_cord_1.y_cord - _cord_2.y_cord) ** 2)) ** 0.5


# STATE
def get_state(_cord_1, _cord_2):
    len_1 = len(_cord_1)
    len_2 = len(_cord_2)
    # first in column
    cords = np.zeros((MAX_LENGTH, MAX_LENGTH))
    for value_1 in range(0, len_1):
        # second in row
        for value_2 in range(0, len_2):
            cords[value_1][value_2] = distance_calculation(_cord_1[value_1], _cord_2[value_2])
    return cords


class cord_2d:
    x_cord = 0
    y_cord = 0
    pass


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

plt.style.use('fivethirtyeight')


def read_cord_files_xy(_files):
    cord_x_1 = []
    cord_y_1 = []
    cord_data_1 = open(_files, 'r').read()
    lines = cord_data_1.split('\n')
    for line in lines:
        if len(line) > 1:
            x, y = line.split(",")
            cord_x_1.append(float(x))
            cord_y_1.append(float(y))
    return cord_x_1, cord_y_1


def read_cord_files(_files):
    cords = []
    cord_data_1 = open(_files, 'r').read()
    lines = cord_data_1.split('\n')
    for line in lines:
        if len(line) > 1:
            x, y = line.split(",")
            temp = cord_2d()
            temp.x_cord = float(x)
            temp.y_cord = float(y)
            cords.append(temp)
    return cords


def render_3(i):
    cord_x_1, cord_y_1 = read_cord_files_xy("new_ag_1.txt")
    cord_x_2, cord_y_2 = read_cord_files_xy("new_ag_2.txt")
    plt.cla()

    plt.plot(cord_x_1, cord_y_1, marker='o', color='g', linewidth=2)
    plt.plot(cord_x_2, cord_y_2, marker='x', color='b', linewidth=2)

    plt.savefig('/home/rajroy/rl_logs/img/'+str(time.time()) + "_.png")



def get_new_cords(_cp_agent_1, _action_number):
    temp_agent = copy.deepcopy(_cp_agent_1)
    print_val=0
    if _action_number == 0:
        temp_agent = action_up(_cp_agent_1)
        if print_val ==1:
            print("action_up")
    elif _action_number == 1:
        temp_agent = action_down(_cp_agent_1)
        if print_val == 1:
            print("action_down")
    elif _action_number == 2:
        temp_agent = action_left(_cp_agent_1)
        if print_val == 1:
            print("action_left")
    elif _action_number == 3:
        temp_agent = action_right(_cp_agent_1)
        if print_val == 1:
            print("action_right")
    elif _action_number == 4:
        temp_agent = action_rotate_xy_1(_cp_agent_1)
        if print_val == 1:
            print("action_rotate_xy")
    elif _action_number == 5:
        temp_agent = action_rotate_yx_1(_cp_agent_1)
        if print_val == 1:
            print("action_rotate_yx")
    return temp_agent


def take_action(_state, _action_number, _agent_1, _agent_2, _concerned_size,_qa):
    new_cords = get_new_cords(_agent_1, _action_number)
    next_state = get_state(new_cords, _agent_2)
    # get_current_visualiztion(_new_ag_1=new_cords, _new_ag_2=_agent_2)
    # reward = get_rewards2(_false_dist=_state,_real_dist= _qa,_dim= _concerned_size)
    prev =    get_rewards_2( _agent_1,_dim= _concerned_size)
    cur_reward = get_rewards_2( new_cords,_dim= _concerned_size)
    done_flag = False
    reward = prev - cur_reward
    print(reward)
    done_flag = False
    if reward > 0:
        reward = reward + 0.2
    else:
        reward = reward - 0.2
    if reward > 20:
        print(reward)
    if reward == 100:
        done_flag = True

    # get_current_visualiztion(_new_ag_1=new_cords, _new_ag_2=_agent_2)
    return next_state, reward, done_flag,new_cords,agent_2



def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        # if _cont[len(_cont) - 1].strip() != "END":
        #     f.write("END")
    return

def save_cords_2_file(_cords,_file):
    out_string =""
    for val in _cords:
        out_string+=str(val.x_cord)+","+str(val.y_cord)+"\n"
        write2File(_file,out_string)

model = create_q_model()
model_target = create_q_model()
l = 10


# AGENT

agent_1 = read_cord_files("cord_1.txt")
agent_2 = read_cord_files("cord_2.txt")
real_agent_1 = read_cord_files("cord_1_real.txt")
real_agent_2 = read_cord_files("cord_2_real.txt")
_dim_1 = len(agent_1)
_dim_2 = len(agent_2)
dim = _dim_1 * _dim_2
real_distance_map = get_state(real_agent_1, real_agent_2)

agent_distance_map = get_state(agent_1, agent_2)
r = get_rmsd(real_distance_map, agent_distance_map, dim)
print(r)
real_agent_1 = read_cord_files("cord_1_real.txt")
real_agent_2 = read_cord_files("cord_2_real.txt")

action_history = []
state_history = []
state_next_history = []
done_history = []
rewards_history = []


episode_reward_history = []
episode_count = 0


def get_current_visualiztion(_new_ag_1,_new_ag_2):
    save_cords_2_file(_cords=_new_ag_1, _file="new_ag_1.txt")
    save_cords_2_file(_cords=_new_ag_2, _file="new_ag_2.txt")

    ani = animation.FuncAnimation(plt.gcf(), render_3, interval=1000)
    plt.show()


while True:  # Run until solved

    # agent_1 = read_cord_files("cord_1.txt")
    # agent_2 = read_cord_files("cord_2.txt")
    # real_distance_map = get_state(real_agent_1, real_agent_2)
    frame_count = 0
    agent_1 = read_cord_files("cord_1.txt")
    agent_2 = read_cord_files("cord_2.txt")
    agent_distance_map = get_state(agent_1, agent_2)
    episode_reward = 0
    save_cords_2_file(_cords=agent_1, _file="new_ag_1.txt")
    save_cords_2_file(_cords=agent_2, _file="new_ag_2.txt")
    for timestep in range(1, max_steps_per_episode):


        frame_count += 1
        cur_state = get_state(agent_1, agent_2)

        print('take your action:')
        e_action = input()
        action = int(e_action)

        next_state, rewards, done ,agent_1,agent_2= take_action(_state=cur_state, _action_number=action, _agent_1=agent_1,
                                                _agent_2=agent_2, _concerned_size=dim,_qa =real_distance_map)
        print(rewards)
        state = next_state
        save_cords_2_file(_cords=agent_1, _file="new_ag_1.txt")
        save_cords_2_file(_cords=agent_2, _file="new_ag_2.txt")