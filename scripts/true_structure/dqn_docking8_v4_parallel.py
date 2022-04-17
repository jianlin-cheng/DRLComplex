
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
from sklearn.metrics import pairwise_distances
import time
from multiprocessing import Process


from docking_gd_parallel import *

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
import sklearn

tf.reset_default_graph()
sess = tf.InteractiveSession()
L = tf.keras.layers




class env():

    def __init__(self, true_pdb_file, pdb_file):
        self.true_pose = pyrosetta.pose_from_pdb(true_pdb_file)
        self.pose = pyrosetta.pose_from_pdb(pdb_file)

        #self.atom_restraints = add_cons_to_pose(self.pose, res_file)
        #self.scorefxn = ScoreFunction()
        ##self.scorefxn.add_weights_from_file(weight_file)
        #self.scorefxn.set_weight(atom_pair_constraint, 1)


        #self.atom_restraints.apply(self.pose)
        #self.prev_energy = self.scorefxn(self.pose)
        #self.pose.remove_constraints()



        self.original_pose = Pose()
        self.original_pose.assign(self.pose)
        self.prev_pose = Pose()
        self.prev_pose.assign(self.pose)


        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        self.n_actions = 12


        self.all_residues_ids_chain1 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(self.pose, 'A')
        self.all_residues_ids_chain2 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(self.pose, 'B')



        #Rotation
        self.spinmover_x = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_x.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(1, 0, 0))
        self.spinmover_x.angle_magnitude(1)

        self.spinmover_x_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_x_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(1, 0, 0))  
        self.spinmover_x_backward.angle_magnitude(-1)

        self.spinmover_y = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_y.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 1, 0))
        self.spinmover_y.angle_magnitude(1)

        self.spinmover_y_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_y_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 1, 0))  
        self.spinmover_y_backward.angle_magnitude(-1)

        self.spinmover_z = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_z.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, 1))
        self.spinmover_z.angle_magnitude(1)

        self.spinmover_z_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_z_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, 1))  
        self.spinmover_z_backward.angle_magnitude(-1)



        #Translation
        self.transmover_x = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_x.step_size(1)
        self.transmover_x.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(1, 0, 0))

        self.transmover_x_backward = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_x_backward.step_size(-1)
        self.transmover_x_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(1, 0, 0))

        self.transmover_y = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_y.step_size(1)
        self.transmover_y.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 1, 0))

        self.transmover_y_backward = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_y_backward.step_size(-1)
        self.transmover_y_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 1, 0))

        self.transmover_z = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_z.step_size(1)
        self.transmover_z.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, 1))

        self.transmover_z_backward = rigid_moves.RigidBodyTransMover(self.pose, 1)
        self.transmover_z_backward.step_size(-1)
        self.transmover_z_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, 1))



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

        '''start_A = self.pose.conformation().chain_begin(1)
        end_A = self.pose.conformation().chain_end(1)
        start_B = self.pose.conformation().chain_begin(2)
        end_B = self.pose.conformation().chain_end(2)
        distance_map1 = np.empty(shape=(end_A-start_A+1,end_B-start_B+1))
        #print(start_A, end_A)
        #print(start_B, end_B)


        for i in range(start_A, end_A+1):
          for j in range(start_B, end_B+1):
            distance_map1[i-1, j-1-end_A] = self.find_dist(i, j)'''
        
        #start_time = time.time()

        xyz_CA_chain1 = np.array([self.pose.residue(r).xyz('CA') for r in self.all_residues_ids_chain1])
        xyz_CA_chain2 = np.array([self.pose.residue(r).xyz('CA') for r in self.all_residues_ids_chain2])
        distance_map = pairwise_distances(xyz_CA_chain1, xyz_CA_chain2, metric='euclidean', n_jobs=1)

        distance_map.resize(150, 150)

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

    def rotatePose(self, R):
        start_A = self.pose.conformation().chain_begin(2)
        end_A = self.pose.conformation().chain_end(2)
        for r in range(start_A, end_A+1):
          for a in range(1, len(self.pose.residue(r).atoms())+1):
            v = np.array([self.pose.residue(r).atom(a).xyz()[0], self.pose.residue(r).atom(a).xyz()[1], self.pose.residue(r).atom(a).xyz()[2]])
            newv = R.dot(v)
            self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newv[0], newv[1], newv[2]))
            
    
    
    def translatePose(self, t):
        start_A = self.pose.conformation().chain_begin(2)
        end_A = self.pose.conformation().chain_end(2)
        for r in range(start_A, end_A+1):
          for a in range(1, len(self.pose.residue(r).atoms())+1):
            newx = self.pose.residue(r).atom(a).xyz()[0] + t[0]
            newy = self.pose.residue(r).atom(a).xyz()[1] + t[1]
            newz = self.pose.residue(r).atom(a).xyz()[2] + t[2]
            self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newx, newy, newz))
    
    
    def step(self, action):
        
        if action == 0:
          #self.rotatePose(self.rotation_x_forward)
          self.spinmover_x.apply(self.pose)
        elif action == 1:
          #self.rotatePose(self.rotation_y_forward)
          self.spinmover_y.apply(self.pose)
        elif action == 2:
          #self.rotatePose(self.rotation_z_forward)
          self.spinmover_z.apply(self.pose)
        elif action == 3:
          #self.translatePose([1, 0, 0])
          self.transmover_x.apply(self.pose)
        elif action == 4:
          #self.translatePose([0, 1, 0])
          self.transmover_y.apply(self.pose)
        elif action == 5:
          #self.translatePose([0, 0, 1])
          self.transmover_z.apply(self.pose)
        elif action == 6:
          #self.rotatePose(self.rotation_x_backward)
          self.spinmover_x_backward.apply(self.pose)
        elif action == 7:
          #self.rotatePose(self.rotation_y_backward)
          self.spinmover_y_backward.apply(self.pose)
        elif action == 8:
          #self.rotatePose(self.rotation_z_backward)
          self.spinmover_z_backward.apply(self.pose)
        elif action == 9:
          #self.translatePose([-1, 0, 0])
          self.transmover_x_backward.apply(self.pose)
        elif action == 10:
          #self.translatePose([0, -1, 0])
          self.transmover_y_backward.apply(self.pose)
        else:
          #self.translatePose([0, 0, -1])
          self.transmover_z_backward.apply(self.pose)
          
        curr_ca_rmsd = CA_rmsd(self.true_pose, self.pose)

        #self.atom_restraints.apply(self.pose)
        #self.curr_energy = self.scorefxn(self.pose)
        #self.pose.remove_constraints()

        


        #pmm.apply(self.pose)
        
        
        if curr_ca_rmsd <= 1:
          done = True
          reward  = 100
        elif curr_ca_rmsd >= 40:
          done = True
          reward = -40
        else:
          done = False
          diff = self.prev_ca_rmsd - curr_ca_rmsd
          reward = diff
          #diff = math.log10(self.prev_energy) - math.log10(self.curr_energy)
          '''if diff < 0:
             reward = diff
          else:
             reward = 1 / (1 + curr_ca_rmsd)'''
          
        #print(reward)
        self.prev_ca_rmsd = curr_ca_rmsd

        #self.prev_energy = self.curr_energy
          
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




#env = env('/home/esdft/DeepRLP/data/3HE4A_3HE4B.pdb', '/home/esdft/DeepRLP/data/3HE4A_3HE4B_GD.pdb', '/home/esdft/DeepRLP/data/3HE4A_3HE4B.rr', '/home/esdft/DeepRLP/data/talaris2013.wts')
#env = env('/home/esdft/DeepRLP/data/1A2D.pdb', '/home/esdft/DeepRLP/data/1A2D_GD.pdb')
env2 = env('/home/esdft/DeepRLP/data/3HE4A_3HE4B.pdb', '/home/esdft/DeepRLP/data/3HE4_AB.pdb')
env1 = env('/home/esdft/DeepRLP/data/5C39.pdb', '/home/esdft/DeepRLP/data/5C39_AB.pdb')
env3 = env('/home/esdft/DeepRLP/data/1PD3.pdb', '/home/esdft/DeepRLP/data/1PD3_AB.pdb')
env4 = env('/home/esdft/DeepRLP/data/1Z9Z.pdb', '/home/esdft/DeepRLP/data/1Z9Z_AB.pdb')
env5 = env('/home/esdft/DeepRLP/data/5LLJ.pdb', '/home/esdft/DeepRLP/data/5LLJ_AB.pdb')
#env6 = env('/home/esdft/DeepRLP/data/4E83.pdb', '/home/esdft/DeepRLP/data/4E83_AB.pdb')
#env7 = env('/home/esdft/DeepRLP/data/3LO2.pdb', '/home/esdft/DeepRLP/data/3LO2_AB.pdb')


env6 = env('/home/esdft/DeepRLP/data/3CI9A_3CI9B.pdb', '/home/esdft/DeepRLP/data/3CI9_AB.pdb')
env7 = env('/home/esdft/DeepRLP/data/2QL2A_2QL2B.pdb', '/home/esdft/DeepRLP/data/2QL2_AB.pdb')


dist_map = env1.get_distance_map()
state_dim = dist_map.shape
n_actions = env1.n_actions



#env.translatePose(np.array([0, 0, -5]))
#env.pose.dump_pdb('/home/esdft/Desktop/t1.pdb')


'''env.spinmover_x_backward.apply(env.pose)
env.pose.dump_pdb('/home/esdft/Desktop/r2.pdb')

env.spinmover_x.apply(env.pose)
env.pose.dump_pdb('/home/esdft/Desktop/r3.pdb')

elham = env.get_rotation_matrix('x', -1)
env.rotatePose(elham)
env.pose.dump_pdb('/home/esdft/Desktop/r1.pdb')'''

#start_time4 = time.time()

'''env.spinmover_x.apply(env.pose)
env.spinmover_x.apply(env.pose)
env.spinmover_x.apply(env.pose)
env.spinmover_x.apply(env.pose)
env.spinmover_x.apply(env.pose)'''

#env.spinmover_z_backward.apply(env.pose)
#env.spinmover_x_backward.apply(env.pose)
#env.spinmover_x_backward.apply(env.pose)
#env.spinmover_x_backward.apply(env.pose)
#env.spinmover_x_backward.apply(env.pose)

'''elham = env.get_rotation_matrix('z', 1)
elham = env.get_rotation_matrix('z', 1)

env.pose.dump_pdb('/home/esdft/Desktop/r1.pdb')

#env.spinmover_z.apply(env.pose)
#env.spinmover_x.apply(env.pose)
#env.spinmover_x.apply(env.pose)
#env.spinmover_x.apply(env.pose)
#env.spinmover_x.apply(env.pose)

end_time4 = time.time() - start_time4
print(end_time4)

elham = env.get_rotation_matrix('z', -1)
elham = env.get_rotation_matrix('z', -1)

env.pose.dump_pdb('/home/esdft/Desktop/r2.pdb')


elham = env.get_rotation_matrix('z', -2)
env.rotatePose(elham)
env.pose.dump_pdb('/home/esdft/Desktop/r3.pdb')'''


'''print(state_dim)
print(n_actions)

env.atom_restraints.apply(env.pose)
print(env.scorefxn(env.pose))
env.pose.remove_constraints()
#env.pose.remove_constraints()
#print(env.scorefxn.show(env.original_pose))

print(env.prev_energy)

for itr in range(1):
    env.step(3)
    

print(env.curr_energy)

env.atom_restraints.apply(env.pose)
#add_cons_to_pose(env.pose, '/home/esdft/DeepRLP/data/3HE4A_3HE4B.rr')
print(env.scorefxn(env.pose))
env.pose.remove_constraints()

#scorefxn = ScoreFunction()
#scorefxn.add_weights_from_file('/home/esdft/DeepRLP/data/talaris2013.wts')
#scorefxn.set_weight(atom_pair_constraint, 1)





for itr in range(1):
    env.step(9)

print(env.curr_energy)

env.atom_restraints.apply(env.pose)
print(env.scorefxn(env.pose))
env.pose.remove_constraints()'''
#env.pose.dump_pdb('/home/esdft/DeepRLP/data/3HE4_rotated1.pdb')

'''print(dist_map)
print(state_dim)
print(type(state_dim))
print(n_actions)




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


exp_replay1 = ReplayMemory(50000)
#exp_replay2 = ReplayMemory(15000)
#exp_replay3 = ReplayMemory(15000)
#exp_replay4 = ReplayMemory(15000)
#exp_replay5 = ReplayMemory(15000)
#exp_replay6 = ReplayMemory(15000)
#exp_replay7 = ReplayMemory(15000)
#play_and_record(agent, env, exp_replay,env.get_current_state(), n_steps=10000)

#for _ in range(10000):
play_and_record(agent, env1, exp_replay1, n_steps=5000)
play_and_record(agent, env2, exp_replay1, n_steps=5000)
play_and_record(agent, env3, exp_replay1, n_steps=5000)
play_and_record(agent, env4, exp_replay1, n_steps=5000)
play_and_record(agent, env5, exp_replay1, n_steps=5000)
play_and_record(agent, env6, exp_replay1, n_steps=5000)
play_and_record(agent, env7, exp_replay1, n_steps=5000)


def sample_batch():
    '''obs_batch1, act_batch1, reward_batch1, next_obs_batch1, is_done_batch1 = exp_replay1.sample(40)
    obs_batch2, act_batch2, reward_batch2, next_obs_batch2, is_done_batch2 = exp_replay2.sample(36)
    obs_batch3, act_batch3, reward_batch3, next_obs_batch3, is_done_batch3 = exp_replay3.sample(36)
    obs_batch4, act_batch4, reward_batch4, next_obs_batch4, is_done_batch4 = exp_replay4.sample(36)
    obs_batch5, act_batch5, reward_batch5, next_obs_batch5, is_done_batch5 = exp_replay5.sample(36)
    obs_batch6, act_batch6, reward_batch6, next_obs_batch6, is_done_batch6 = exp_replay6.sample(36)
    obs_batch7, act_batch7, reward_batch7, next_obs_batch7, is_done_batch7 = exp_replay7.sample(36)
    
    obs_batch = np.concatenate([obs_batch1, obs_batch2, obs_batch3, obs_batch4, obs_batch5, obs_batch6, obs_batch7])
    act_batch = np.concatenate([act_batch1, act_batch2, act_batch3, act_batch4, act_batch5, act_batch6, act_batch7])
    reward_batch = np.concatenate([reward_batch1, reward_batch2, reward_batch3, reward_batch4, reward_batch5, reward_batch6, reward_batch7])
    next_obs_batch = np.concatenate([next_obs_batch1, next_obs_batch2, next_obs_batch3, next_obs_batch4, next_obs_batch5, next_obs_batch6, next_obs_batch7])
    is_done_batch = np.concatenate([is_done_batch1, is_done_batch2, is_done_batch3, is_done_batch4, is_done_batch5, is_done_batch6, is_done_batch7])

    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = sklearn.utils.shuffle(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch)'''
    
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay1.sample(128)
    
    return {
        obs_ph:obs_batch, actions_ph:act_batch, rewards_ph:reward_batch, 
        next_obs_ph:next_obs_batch, is_done_ph:is_done_batch
    }
    
    
def train():
    _, loss_t = sess.run([train_step, td_loss], sample_batch()) 


for i in trange(10**6):
    
    # play
    #s = env.restore_pose()
    #for _ in range(1):
    play_and_record(agent, env1, exp_replay1, 10)
    play_and_record(agent, env2, exp_replay1, 10)
    play_and_record(agent, env3, exp_replay1, 10)
    play_and_record(agent, env4, exp_replay1, 10)
    play_and_record(agent, env5, exp_replay1, 10)
    play_and_record(agent, env6, exp_replay1, 10)
    play_and_record(agent, env7, exp_replay1, 10)
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
        #plt.show()



