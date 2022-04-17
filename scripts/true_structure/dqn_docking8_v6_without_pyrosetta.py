
import sys
from sklearn.metrics import pairwise_distances
import time
from multiprocessing import Process


from utils import *


import os
import sys
import glob
import numpy as np
from math import sin, cos
import math
import random
import scipy

import tensorflow as tf
import sklearn

tf.reset_default_graph()
sess = tf.InteractiveSession()
L = tf.keras.layers



targets_file = '/home/esdft/DeepRLP/data/targets.txt'
initial_pdbs = '/home/esdft/DeepRLP/data/'
true_pdbs = '/home/esdft/DeepRLP/data/'



class env():

    def __init__(self, true_pdb_file, pdb_file, target_name):


        chain1, chain2 = get_chains(target_name, pdb_file)

        self.CA_chain1 = get_ca_from_chain(chain1)
        self.CA_chain2 = get_ca_from_chain(chain2)

        self.atoms_chain1 = get_atoms_from_chain(chain1)
        self.atoms_chain2 = get_atoms_from_chain(chain2)

        self.original_CA_chain2 = get_ca_from_chain(chain2)
        self.original_atoms_chain2 = get_atoms_from_chain(chain2)


        self.true_ca_atoms = get_CA_atoms(target_name, true_pdb_file)


        #self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        self.prev_ca_rmsd = compute_RMSD(self.true_ca_atoms, np.concatenate((self.CA_chain1, self.CA_chain2), axis=0))
        self.n_actions = 12








        self.rotation_x_forward = self.get_rotation_matrix('x', 1)
        self.rotation_y_forward = self.get_rotation_matrix('y', 1)
        self.rotation_z_forward = self.get_rotation_matrix('z', 1)

        self.rotation_x_backward = self.get_rotation_matrix('x', -1)
        self.rotation_y_backward = self.get_rotation_matrix('y', -1)
        self.rotation_z_backward = self.get_rotation_matrix('z', -1)
        
        
        
    def get_distance_map(self):
        
        #start_time = time.time()

        distance_map = pairwise_distances(self.CA_chain1, self.CA_chain2, metric='euclidean', n_jobs=1)

        distance_map.resize(60, 60)

        #end_time = time.time() - start_time
        #print(end_time)

        #cmp_mat = (distance_map-distance_map1)>0.00000001
        #print(np.any(cmp_mat==True))

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

    def rotate(self, R):
        self.atoms_chain2 = np.dot(R, self.atoms_chain2.transpose()).transpose()
        self.CA_chain2 = np.dot(R, self.CA_chain2.transpose()).transpose()
        
    
    
    def step(self, action):
        
        if action == 0:
          self.rotate(self.rotation_x_forward)
        elif action == 1:
          self.rotate(self.rotation_y_forward)
        elif action == 2:
          self.rotate(self.rotation_z_forward)
        elif action == 3:
          self.atoms_chain2 = self.atoms_chain2 + np.array([1, 0, 0])
          self.CA_chain2 = self.CA_chain2 + np.array([1, 0, 0])
        elif action == 4:
          self.atoms_chain2 = self.atoms_chain2 + np.array([0, 1, 0])
          self.CA_chain2 = self.CA_chain2 + np.array([0, 1, 0])
        elif action == 5:
          self.atoms_chain2 = self.atoms_chain2 + np.array([0, 0, 1])
          self.CA_chain2 = self.CA_chain2 + np.array([0, 0, 1])
        elif action == 6:
          self.rotate(self.rotation_x_backward)
        elif action == 7:
          self.rotate(self.rotation_y_backward)
        elif action == 8:
          self.rotate(self.rotation_z_backward)
        elif action == 9:
          self.atoms_chain2 = self.atoms_chain2 + np.array([-1, 0, 0])
          self.CA_chain2 = self.CA_chain2 + np.array([-1, 0, 0])
        elif action == 10:
          self.atoms_chain2 = self.atoms_chain2 + np.array([0, -1, 0])
          self.CA_chain2 = self.CA_chain2 + np.array([0, -1, 0])
        else:
          self.atoms_chain2 = self.atoms_chain2 + np.array([0, 0, -1])
          self.CA_chain2 = self.CA_chain2 + np.array([0, 0, -1])
          
        #curr_ca_rmsd = CA_rmsd(self.true_pose, self.pose)


        curr_ca_rmsd = compute_RMSD(self.true_ca_atoms, np.concatenate((self.CA_chain1, self.CA_chain2), axis=0))
        
        
        if curr_ca_rmsd <= 1:
          done = True
          reward  = 100
        elif curr_ca_rmsd >= 60:
          done = True
          reward = -60
        else:
          done = False
          diff = self.prev_ca_rmsd - curr_ca_rmsd
          reward = diff
          #diff = math.log10(self.prev_energy) - math.log10(self.curr_energy)
          
        #print(reward)
        self.prev_ca_rmsd = curr_ca_rmsd

        #self.prev_energy = self.curr_energy
          
        return self.get_distance_map(),reward, done
        
    def reset(self):
        self.CA_chain2 = self.original_CA_chain2.copy()
        self.atoms_chain2 = self.original_atoms_chain2.copy()
        self.prev_ca_rmsd = compute_RMSD(self.true_ca_atoms, np.concatenate((self.CA_chain1, self.CA_chain2), axis=0))
        return self.get_distance_map()    

    def get_current_state(self):
        return self.get_distance_map()
    
    '''def restore_pose(self):
        self.pose = self. prev_pose.clone()
        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        return self.get_distance_map()


    def set_prev_pose(self):
        self.prev_pose = self.pose.clone()'''
    
 


class ReplayMemory():
    """Stores past transition experience for training
    to decorrelate samples
    """

    def __init__(self, memory_size=10000):
        self.memory_size = memory_size
        self.distance_maps = None  # shape to be determined by first distance_map
        self.next_distance_maps = None
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.done = np.empty(self.memory_size, dtype=np.bool)
        self.current_idx = 0
        self.num_data = 0


    def store_distance_map(self, distance_map):
        if self.distance_maps is None:
            self.distance_maps = np.empty((self.memory_size,) + distance_map.shape, dtype=np.float32)
        self.distance_maps[self.current_idx] = distance_map

    def store_next_distance_map(self, next_distance_map):
        if self.next_distance_maps is None:
            self.next_distance_maps = np.empty((self.memory_size,) + next_distance_map.shape, dtype=np.float32)
        self.next_distance_maps[self.current_idx] = next_distance_map


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
        next_obs_sample = np.stack([self.next_distance_maps[idx] for idx in idxes])
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









with open(targets_file, 'r') as f1:
    targets = [line.rstrip() for line in f1]



num_targets = len(targets)

Dict = {}

for j in range(num_targets):
    Dict[j] = env(f'{true_pdbs}/{targets[j]}.pdb', f'{initial_pdbs}/{targets[j]}.start.pdb', targets[j])




env1 = Dict[0]
dist_map = env1.get_distance_map()
state_dim = dist_map.shape
n_actions = env1.n_actions


'''all_ca_atoms = np.concatenate((env1.CA_chain1, env1.CA_chain2), axis=0)

true_ca_atoms = env1.true_ca_atoms



print(compute_RMSD(all_ca_atoms, true_ca_atoms))

#print(all_atoms)
all_atoms = np.concatenate((env1.atoms_chain1, env1.atoms_chain2), axis=0)
write_pdb(f'{initial_pdbs}/{targets[0]}.start.pdb', all_atoms, 'test_modified.pdb')


start = time.time()
env1.step(0)
env1.step(3)
end = time.time() - start

print('time')
print(end)

print(env1.prev_ca_rmsd)
#env1.reset()
print(env1.prev_ca_rmsd)

all_atoms = np.concatenate((env1.atoms_chain1, env1.atoms_chain2), axis=0)
#print(all_atoms)
write_pdb(f'{initial_pdbs}/{targets[0]}.start.pdb', all_atoms, 'test4_modified.pdb')'''



agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=1)
sess.run(tf.global_variables_initializer())




replay = ReplayMemory()


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    
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
      #f.write('\t')
      #f.write(str(env.prev_energy))
      f.write('\n')
    print(mean_reward)
    print(env.prev_ca_rmsd)
    #print(env.prev_energy)
    return mean_reward


 
#evaluate(env, agent, n_games=20)
#print(env.prev_ca_rmsd)

def play_and_record(agent, env, exp_replay, n_steps=1):

    s = env.get_current_state()
    total_reward = 0.0
    
    for t in range(n_steps):
        qs = agent.get_qvalues([s])
        a = agent.sample_actions(qs)[0]
        
        exp_replay.store_distance_map(s)
        
        next_s, r, done = env.step(a)
        
        exp_replay.store_next_distance_map(next_s)
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
saver = tf.train.Saver()


#new_saver = tf.train.Saver()
#new_saver = tf.train.import_meta_graph('my_model_3LO2.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))



from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pandas import DataFrame
moving_average = lambda x, span, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(span=span, **kw).mean().values

mean_rw_history = []
td_loss_history = []


exp_replay1 = ReplayMemory(100000)

for _ in range(10000):
    for t in range(num_targets):
        play_and_record(agent, Dict[t], exp_replay1, n_steps=1)



def sample_batch():
    
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay1.sample(128)
    
    return {
        obs_ph:obs_batch, actions_ph:act_batch, rewards_ph:reward_batch, 
        next_obs_ph:next_obs_batch, is_done_ph:is_done_batch
    }
    
    
def train():
    _, loss_t = sess.run([train_step, td_loss], sample_batch()) 



#new_saver = tf.train.import_meta_graph('my_model.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))

for i in trange(10**6):
    
    # play
    #s = env.restore_pose()
    #for _ in range(1):
    #################play_and_record(agent, Dict[i%num_targets], exp_replay1, 10)
    
    for t in range(num_targets):
        play_and_record(agent, Dict[t], exp_replay1, 10)
    #env.set_prev_pose()
    
    # train
    #_, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, batch_size=64))
    #td_loss_history.append(loss_t)
    train()
    saver.save(sess, 'my_model')
    
    
    # adjust agent parameters
    if i % 500 == 0:
        load_weigths_into_target_network(agent, target_network)
        agent.epsilon = max(agent.epsilon * 0.99, 0.01)
        #mean_rw_history.append(evaluate(env, agent, n_games=1))
        
    
    #if i % 100 == 0:
        #clear_output(True)
        #print("buffer size = %i, epsilon = %.5f" % (exp_replay.memory_size, agent.epsilon))
        
        #plt.subplot(1,2,1)
        #plt.title("mean reward per game")
        #plt.plot(mean_rw_history)
        #plt.grid()

        #assert not np.isnan(loss_t)
        #plt.figure(figsize=[12, 4])
        #plt.subplot(1,2,2)
        #plt.title("TD loss history (moving average)")
        #plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
        #plt.grid()
        #plt.show()'''



