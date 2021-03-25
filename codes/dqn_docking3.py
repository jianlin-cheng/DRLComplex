
import sys
from rosetta import *
from pyrosetta import *
from rosetta.protocols.rigid import *
from rosetta.core.scoring import *
from pyrosetta import PyMOLMover
from rosetta.protocols.rigid import *
import pyrosetta.rosetta.protocols.rigid as rigid_moves
import pyrosetta.rosetta.protocols.rigid as rigid_moves
from pyrosetta import PyMOLMover

init()
pmm = PyMOLMover()
pmm.keep_history(True)

import os
import sys
import glob
import numpy as np
from math import sin, cos
import math
import random

import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()
L = tf.keras.layers 





class env():

    def __init__(self, true_pdb_file, pdb_file):
        self.true_pose = pyrosetta.pose_from_pdb(true_pdb_file)
        self.pose = pyrosetta.pose_from_pdb(pdb_file)
        self.original_pose = Pose()
        self.original_pose.assign(self.pose)
        self.prev_pose = Pose()
        self.prev_pose.assign(self.pose)


        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        self.n_actions = 12

        self.rotation_x_forward = self.get_rotation_matrix('x', 1)
        self.rotation_y_forward = self.get_rotation_matrix('y', 1)
        self.rotation_z_forward = self.get_rotation_matrix('z', 1)

        self.rotation_x_backward = self.get_rotation_matrix('x', -1)
        self.rotation_y_backward = self.get_rotation_matrix('y', -1)
        self.rotation_z_backward = self.get_rotation_matrix('z', -1)

    def find_dist(self, res_i, res_j):
        atm_i = 'CA' if self.pose.residue(res_i).name()[0:3] == 'GLY' else 'CB'
        atm_j = 'CA' if self.pose.residue(res_j).name()[0:3] == 'GLY'  else 'CB'
        xyz_i = self.pose.residue(res_i).xyz(atm_i)
        xyz_j = self.pose.residue(res_j).xyz(atm_j)
        
        dist = (xyz_i - xyz_j).norm()
    
        return dist
        
        
        
    def get_distance_map(self):

        start_A = self.pose.conformation().chain_begin(1)
        end_A = self.pose.conformation().chain_end(1)
        start_B = self.pose.conformation().chain_begin(2)
        end_B = self.pose.conformation().chain_end(2)
        distance_map = np.empty(shape=(end_A-start_A+1,end_B-start_B+1))
        #print(start_A, end_A)
        #print(start_B, end_B)


        for i in range(start_A, end_A+1):
          for j in range(start_B, end_B+1):
            distance_map[i-1, j-1-end_A] = self.find_dist(i, j)

        distance_map = distance_map[:, :, tf.newaxis]
        distance_map = distance_map / distance_map.max()
        return distance_map
        
    def get_rotation_matrix(self,axis_name, degree_magnitude):
        degree_magnitude = math.radians(degree_magnitude)
        if axis_name == 'x':
          rotation_matrix = np.array([[1, 0, 0],[0, cos(degree_magnitude), -sin(degree_magnitude)],[0, sin(degree_magnitude), cos(degree_magnitude)]])
        elif axis_name == 'y':
          rotation_matrix = np.array([[cos(degree_magnitude), 0, sin(degree_magnitude)],[0, 1, 0],[-sin(degree_magnitude), 0, cos(degree_magnitude)]])
        elif axis_name == 'z':
          rotation_matrix = np.array([[cos(degree_magnitude), -sin(degree_magnitude), 0],[sin(degree_magnitude), cos(degree_magnitude), 0],[0, 0, 1]])

        return rotation_matrix

    def rotatePose(self, R):
        start_A = self.pose.conformation().chain_begin(1)
        end_A = self.pose.conformation().chain_end(1)
        for r in range(start_A, end_A+1):
          for a in range(1, len(self.pose.residue(r).atoms())+1):
            v = np.array([self.pose.residue(r).atom(a).xyz()[0], self.pose.residue(r).atom(a).xyz()[1], self.pose.residue(r).atom(a).xyz()[2]])
            newv = R.dot(v)
            self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newv[0], newv[1], newv[2]))
            
    
    
    def translatePose(self, t):
        start_A = self.pose.conformation().chain_begin(1)
        end_A = self.pose.conformation().chain_end(1)
        for r in range(start_A, end_A+1):
          for a in range(1, len(self.pose.residue(r).atoms())+1):
            newx = self.pose.residue(r).atom(a).xyz()[0] + t[0]
            newy = self.pose.residue(r).atom(a).xyz()[1] + t[1]
            newz = self.pose.residue(r).atom(a).xyz()[2] + t[2]
            self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newx, newy, newz))
    
    
    def step(self, action):
        
        if action == 0:
          self.rotatePose(self.rotation_x_forward)
        elif action == 1:
          self.rotatePose(self.rotation_y_forward)
        elif action == 2:
          self.rotatePose(self.rotation_z_forward)
        elif action == 3:
          self.translatePose([1, 0, 0])
        elif action == 4:
          self.translatePose([0, 1, 0])
        elif action == 5:
          self.translatePose([0, 0, 1])
        elif action == 6:
          self.rotatePose(self.rotation_x_backward)
        elif action == 7:
          self.rotatePose(self.rotation_y_backward)
        elif action == 8:
          self.rotatePose(self.rotation_z_backward)
        elif action == 9:
          self.translatePose([-1, 0, 0])
        elif action == 10:
          self.translatePose([0, -1, 0])
        else:
          self.translatePose([0, 0, -1])
          
        curr_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        pmm.apply(self.pose)
        
        if curr_ca_rmsd <= 0.4:
          done = True
          reward  = 1000
        elif curr_ca_rmsd >= 40:
          done = True
          reward = -40
        else:
          done = False
          diff = self.prev_ca_rmsd - curr_ca_rmsd
          reward = diff if diff < 0 else 1 / (1 + curr_ca_rmsd)
          
        #print(reward)
        self.prev_ca_rmsd = curr_ca_rmsd
          
        return self.get_distance_map(),reward, done
        
    def reset(self):
        self.pose = self.original_pose.clone()
        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        return self.get_distance_map()    

    def get_current_state(self):
        return self.get_distance_map()
    
    def restore_pose(self):
        self.pose = self. prev_pose.clone()
        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        return self.get_distance_map()


    def set_prev_pose(self):
        self.prev_pose = self.pose.clone()
    
 


class ReplayMemory():
    """Stores past transition experience for training
    to decorrelate samples
    """

    def __init__(self, memory_size=10000):
        self.memory_size = memory_size
        self.distance_maps = None  # shape to be determined by first distance_map
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.done = np.empty(self.memory_size, dtype=np.bool)
        self.current_idx = 0
        self.num_data = 0


    def store_distance_map(self, distance_map):
        if self.distance_maps is None:
            self.distance_maps = np.empty((self.memory_size,) + distance_map.shape, dtype=np.float32)
        self.distance_maps[self.current_idx] = distance_map


    def store_transition(self, action, reward, done):
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.done[self.current_idx] = done
        self.current_idx = (self.current_idx + 1) % self.memory_size
        if self.num_data < self.memory_size:
            self.num_data += 1

    def sample(self, batch_size):
        if self.num_data < self.memory_size:
            idxes = random.sample(range(self.num_data - 1), batch_size)
        else:
            idxes = random.sample(range(self.memory_size - 1), batch_size)
        obs_sample = np.stack([self.distance_maps[idx] for idx in idxes])
        action_sample = self.actions[idxes]
        reward_sample = self.rewards[idxes]
        next_obs_sample = np.stack([self.distance_maps[idx + 1] for idx in idxes])
        done_sample = self.done[idxes]

        return obs_sample, action_sample, reward_sample, next_obs_sample, done_sample



class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        
        with tf.variable_scope(name, reuse=reuse):
            
            
            self.network = tf.keras.models.Sequential()
            self.network.add(L.Conv2D(16, (3, 3), strides=[2,2,], activation='relu', input_shape=state_shape))
            self.network.add(L.Conv2D(32, (3, 3), strides=[2,2,], activation='relu'))
            self.network.add(L.Conv2D(64, (3, 3), strides=[2,2,], activation='relu'))
            self.network.add(L.Flatten())
            self.network.add(L.Dense(256, activation='relu'))
            self.network.add(L.Dense(n_actions, activation='linear'))
            
            self.state_t = tf.placeholder('float32', [None,] + list(state_shape))
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t)
            
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        
        qvalues = qvalues = self.network(state_t)
        return qvalues
    
    def get_qvalues(self, state_t):
        
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})
    
    def sample_actions(self, qvalues):
        
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)




env = env('/home/esdft/Desktop/3HE4A_3HE4B.pdb', '/home/esdft/Desktop/3HE4_AB.pdb')
dist_map = env.get_distance_map()
state_dim = dist_map.shape
n_actions = env.n_actions

'''print(dist_map)
print(state_dim)
print(type(state_dim))
print(n_actions)


print(env.step(8))

print(env.pose.residue(1).atom('CB').xyz())
print(env.pose.residue(2).atom('CB').xyz())
print(env.pose.residue(3).atom('CB').xyz())
print(env.pose.residue(4).atom('CB').xyz())
print(env.pose.residue(5).atom('CB').xyz())
print(env.pose.residue(6).atom('CB').xyz())
print(env.pose.residue(7).atom('CB').xyz())
print(env.pose.residue(8).atom('CB').xyz())



print(env.step(2))

print(env.pose.residue(1).atom('CB').xyz())
print(env.pose.residue(2).atom('CB').xyz())
print(env.pose.residue(3).atom('CB').xyz())
print(env.pose.residue(4).atom('CB').xyz())
print(env.pose.residue(5).atom('CB').xyz())
print(env.pose.residue(6).atom('CB').xyz())
print(env.pose.residue(7).atom('CB').xyz())
print(env.pose.residue(8).atom('CB').xyz())

print(env.prev_ca_rmsd)

print(env.get_current_state())'''

agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0.5)
sess.run(tf.global_variables_initializer())

replay = ReplayMemory()


def evaluate(env, agent, n_games=3, greedy=False, t_max=10000):
    
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done = env.step(action)
            #print(action, r, done)
            reward += r
            if done: break
                
        rewards.append(reward)
    mean_reward = np.mean(rewards)
    with open(os.getcwd()+'/rewards.txt', 'a') as f:
      f.write(str(mean_reward))
      f.write('\t')
      f.write(str(env.prev_ca_rmsd))
      f.write('\n')
    print(mean_reward)
    print(env.prev_ca_rmsd)
    return mean_reward


 
#evaluate(env, agent, n_games=3)
#print(env.prev_ca_rmsd)

def play_and_record(agent, env, exp_replay, initial_state, n_steps=1):

    s = initial_state
    total_reward = 0.0
    
    for t in range(n_steps):
        qs = agent.get_qvalues([s])
        a = agent.sample_actions(qs)[0]
        
        exp_replay.store_distance_map(s)
        
        next_s, r, done = env.step(a)
        
        exp_replay.store_transition(a, r, done)
        
        
        total_reward +=r
        
        if done: s=env.reset()
        else: s=next_s
    
    return total_reward
 
 
#initial_state=env.reset()
#play_and_record(agent, env, replay, initial_state=env.reset(), n_steps=100)

#print(replay.sample(32))
 
 
 
 
 
 
 
 
target_network = DQNAgent("target_network", state_dim, n_actions)

def load_weigths_into_target_network(agent, target_network):

    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)
    
    
    
    

obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
actions_ph = tf.placeholder(tf.int32, shape=[None])
rewards_ph = tf.placeholder(tf.float32, shape=[None])
next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
is_done_ph = tf.placeholder(tf.float32, shape=[None])

is_not_done = 1 - is_done_ph
gamma = 0.99


current_qvalues = agent.get_symbolic_qvalues(obs_ph)
current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)


next_qvalues_target = target_network.get_symbolic_qvalues(next_obs_ph)
next_state_values_target = is_not_done * tf.reduce_max(next_qvalues_target, axis=1)
reference_qvalues = rewards_ph + gamma * next_state_values_target
td_loss = (current_action_qvalues - reference_qvalues) ** 2
td_loss = tf.reduce_mean(td_loss)
train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=agent.weights)



sess.run(tf.global_variables_initializer())



from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pandas import DataFrame
moving_average = lambda x, span, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(span=span, **kw).mean().values

mean_rw_history = []
td_loss_history = []


exp_replay = ReplayMemory(10**5)
play_and_record(agent, env, exp_replay,env.get_current_state(), n_steps=10000)

def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    return {
        obs_ph:obs_batch, actions_ph:act_batch, rewards_ph:reward_batch, 
        next_obs_ph:next_obs_batch, is_done_ph:is_done_batch
    }
    
    
    




for i in trange(10**5):
    
    # play
    s = env.restore_pose()
    play_and_record(agent, env, exp_replay, s, 10)
    env.set_prev_pose()
    
    # train
    _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, batch_size=64))
    td_loss_history.append(loss_t)
    
    # adjust agent parameters
    if i % 500 == 0:
        load_weigths_into_target_network(agent, target_network)
        agent.epsilon = max(agent.epsilon * 0.99, 0.01)
        mean_rw_history.append(evaluate(env, agent, n_games=3))
    
    if i % 100 == 0:
        clear_output(True)
        print("buffer size = %i, epsilon = %.5f" % (exp_replay.memory_size, agent.epsilon))
        
        #plt.subplot(1,2,1)
        #plt.title("mean reward per game")
        #plt.plot(mean_rw_history)
        #plt.grid()

        assert not np.isnan(loss_t)
        #plt.figure(figsize=[12, 4])
        #plt.subplot(1,2,2)
        #plt.title("TD loss history (moving average)")
        #plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
        #plt.grid()
        #plt.show()

