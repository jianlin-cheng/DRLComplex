# https://github.com/keras-team/keras-io/blob/master/examples/rl/deep_q_network_breakout.py
#https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
import sys
#https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
#https://gsurma.medium.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
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

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
eps_decay = 0.001
NUM_ACTIONS = 5
MOVE_INTERVAL = 0.1
ROTATE_INTERVAL = math.radians(15)
MAX_LENGTH = 10
BATCH_SIZE = 32
UPDATE_INTERVAL = 100
MEMORY_SIZE = 10000
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1 # Minimum epsilon greedy parameter
epsilon_max = 1.0
max_steps_per_episode =1000

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

    #EXPERIMENTAL
    temps = copy.deepcopy(_cords)
    # tot_x= 0
    # tot_y=0
    # for val in _cords:
    #     tot_x+=float(val.x_cord)
    #     tot_y += float(val.y_cord)
    # tot_x =tot_x/len(_cords)
    # tot_y = tot_y / len(_cords)
    for _xy in temps:
        _xy.x_cord = _xy.x_cord * c - _xy.y_cord * s
        _xy.y_cord = _xy.x_cord * s + _xy.y_cord * c
    return temps

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
    tot_x = tot_x / float(len(_cords))
    tot_y = tot_y / float(len(_cords))

    for _xy in temps:
        #(cos(angle) * (p.x - cx) - sin(angle) * (p.y - cy) + cx,
        _xy.x_cord =( _xy.x_cord - tot_x) * c - (_xy.y_cord-tot_y) * s +tot_x
        _xy.y_cord =( _xy.x_cord - tot_x) * s + (_xy.y_cord-tot_y) * c +tot_y
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
    tot_x =tot_x/float(len(_cords))
    tot_y = tot_y / float(len(_cords))
    for _xy in temps:
        #(cos(angle) * (p.x - cx) - sin(angle) * (p.y - cy) + cx,


        _xy.x_cord =( _xy.x_cord - tot_x) * c + (_xy.y_cord-tot_y)  * s+tot_x
        _xy.y_cord = - 1 *( _xy.x_cord - tot_x) * s + (_xy.y_cord-tot_y)* c+tot_y

    return temps


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



plt.style.use('fivethirtyeight')




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



def get_new_cords(agent_1, _action_number):
    temp_agent = copy.deepcopy(agent_1)
    print_val=0
    if _action_number == 0:
        temp_agent = action_up(agent_1)
        if print_val ==1:
            print("action_up")
    elif _action_number == 1:
        temp_agent = action_down(agent_1)
        if print_val == 1:
            print("action_down")
    elif _action_number == 2:
        temp_agent = action_left(agent_1)
        if print_val == 1:
            print("action_left")
    elif _action_number == 3:
        temp_agent = action_right(agent_1)
        if print_val == 1:
            print("action_right")
    elif _action_number == 5:
        temp_agent = action_rotate_xy_1(agent_1)
        if print_val == 1:
            print("action_rotate_xy")
    elif _action_number == 4:
        temp_agent = action_rotate_yx_1(agent_1)
        if print_val == 1:
            print("action_rotate_yx")
    return temp_agent


def take_action(_state, _action_number, _agent_1, _agent_2, _concerned_size,_qa):
    new_cords = get_new_cords(_agent_1, _action_number)
    next_state = get_state(new_cords, _agent_2)
    # get_current_visualiztion(_new_ag_1=new_cords, _new_ag_2=_agent_2)
    # prev_reward  = get_rewards(_false_dist=_state,_real_dist= _qa,_dim= _concerned_size)
    # reward = get_rewards(_false_dist=_state,_real_dist= _qa,_dim= _concerned_size)
    save_cords_2_file(_cords=agent_1, _file="new_ag_1.txt")
    save_cords_2_file(_cords=agent_2, _file="new_ag_2.txt")
    prev =    get_rewards_2( _agent_1,_dim= _concerned_size)
    cur_reward = get_rewards_2( new_cords,_dim= _concerned_size)
    done_flag = False
    reward = prev - cur_reward
    if reward > 0:
        reward =reward +0.2
    else:
        reward =reward - 0.2
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
big_counter =0

best_rmsd = 0
while True:  # Run until solved
    if best_rmsd  > 0 :
        print()
    # agent_1 = read_cord_files("cord_1.txt")
    # agent_2 = read_cord_files("cord_2.txt")
    # real_distance_map = get_state(real_agent_1, real_agent_2)
    frame_count = 0
    agent_1 = read_cord_files("cord_1.txt")
    agent_2 = read_cord_files("cord_2.txt")
    agent_distance_map = get_state(agent_1, agent_2)
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
#  reset

        frame_count += 1
        cur_state = get_state(agent_1, agent_2)

        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(NUM_ACTIONS)

        else:
            state_tensor = tf.convert_to_tensor(cur_state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= eps_decay/10
        epsilon = max(epsilon, epsilon_min)
        next_state, rewards, done ,agent_1,agent_2= take_action(_state=cur_state, _action_number=action, _agent_1=agent_1,
                                                _agent_2=agent_2, _concerned_size=dim,_qa =real_distance_map)
        episode_reward+=rewards
        # replay memory stuffss
        action_history.append(action)
        state_history.append(cur_state)
        state_next_history.append(next_state)
        done_history.append(done)
        rewards_history.append(rewards)
        state = next_state
        if len(done_history) > BATCH_SIZE:
            big_counter = big_counter + 1
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=BATCH_SIZE)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(np.expand_dims(state_next_sample, axis=3))
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, NUM_ACTIONS)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # if big_counter ==10:
            #     print("debug")

        if timestep % (UPDATE_INTERVAL) == 0:

            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            # print(template.format(running_reward, episode_count, timestep))
            print(template.format(np.mean(episode_reward_history), episode_count, timestep))
            save_cords_2_file(_cords=agent_1, _file="new_ag_1.txt")
            save_cords_2_file(_cords=agent_2, _file="new_ag_2.txt")
            # get_current_visualiztion(_new_ag_1=agent_1, _new_ag_2=agent_2)

            # Limit the state and reward history
        if len(rewards_history) > MEMORY_SIZE:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

            # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > MEMORY_SIZE:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        # print("avg value "+str(    running_reward)+"\n" )
        episode_count += 1
        # print( "max value "+str(max(rewards_history))+"\n" )
        if max(rewards_history)>70:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            get_current_visualiztion(_new_ag_1=agent_1,_new_ag_2=agent_2)

            break