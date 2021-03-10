import copy
import random
from math import cos, sin

import numpy as np
import time

# EXPLOITATION VS EXPLORATATION
# POLICY NETWORK
# vALUE NETWORK
# REWARD CALCULATION
# DIFFERNT TARGET
##replay memory
from tensorflow import keras
from tensorflow.keras import layers

NUM_ACTIONS = 6
MOVE_INTERVAL = 0.1
ROTATE_INTERVAL = 1
MAX_LENGTH = 10
BATCH_SIZE = 10
EPISODE = 10
MEMORY_SIZE = 10
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(MAX_LENGTH, MAX_LENGTH, 1))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Flatten()(layer2)

    layer4 = layers.Dense(32, activation="relu")(layer3)
    action = layers.Dense(NUM_ACTIONS, activation="linear")(layer4)

    return keras.Model(inputs=inputs, outputs=action)


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
        _xy.x_cords = float(_xy.x_cords) - MOVE_INTERVAL
    return temps


def action_right(_cords):
    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cords = float(_xy.x_cords) + MOVE_INTERVAL
    return temps

#https://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
def action_rotate_xy(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cords = _xy.x_cords * c + _xy.y_cords * s
        _xy.y_cords = - 1 * _xy.x_cords * s + _xy.y_cords * c
    return temps


def action_yx(_cords):
    s = sin(ROTATE_INTERVAL)  # angle is in radians
    c = cos(ROTATE_INTERVAL)  # angle is in radians

    temps = copy.deepcopy(_cords)
    for _xy in temps:
        _xy.x_cords = _xy.x_cords * c - _xy.y_cords * s
        _xy.y_cords = _xy.x_cords * s + _xy.y_cords * c
    return temps


def distance_calculation(_cord_1, _cord_2):
    return (((_cord_1.x - _cord_2.x) ** 2) + ((_cord_2.y - _cord_2.y) ** 2)) ** 0.5


# STATE
def get_state(_cord_1, _cord_2):
    len_1 = len(_cord_1)
    len_2 = len(_cord_2)
    # first in column
    cords = np.zero((MAX_LENGTH, MAX_LENGTH))
    for value_1 in range(0, len_1):
        # second in row
        for value_2 in range(0, len_2):
            cords[value_1][value_2] = distance_calculation(_cord_1[value_1], _cord_2[value_2])
    return cords


def state_generation():
    return None


# new_Q - old_Q
def reward_calculation():
    return None


class cord_2d:
    x_cord = 0
    y_cord = 0
    pass


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

plt.style.use('fivethirtyeight')


# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)

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
            temp.x = float(x)
            temp.y = float(y)
            cords.append(temp)
    return cords


def render_3(i):
    cord_x_1, cord_y_1 = read_cord_files_xy("cord_1.txt")
    cord_x_2, cord_y_2 = read_cord_files_xy("cord_2.txt")
    plt.cla()

    plt.plot(cord_x_1, cord_y_1, marker='o', color='g', linewidth=2)
    plt.plot(cord_x_2, cord_y_2, marker='x', color='b', linewidth=2)


def get_random_cords(_len):
    cord_list = []
    cord_data = open('cord_1.txt', 'r').read()
    lines = cord_data.split('\n')
    for val in range(0, l):
        temp_cord = cord_2d()
        temp_cord.x_cord = random.randrange(1, 100)
        temp_cord.y_cord = random.randrange(1, 100)
        cord_list.append(temp_cord)

    return cord_list


l = 10
# ani = animation.FuncAnimation(plt.gcf(), render_3, interval=100)
# plt.show()
episodes = 100
# AGENT
agent_1 = read_cord_files("cord_1.txt")
agent_2 = read_cord_files("cord_2.txt")
