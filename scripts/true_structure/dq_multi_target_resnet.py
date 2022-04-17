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

from resnet import ResnetBuilder
#from docking_gd_parallel import *
#
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
UPDATE_INTERVAL = 500
TRANSLATE_INTERVAL = 1
ROTATE_INTERVAL = 1.0
BATCH_SAMPLE_SIZE = 64
NUMBER_OF_ACTIONS = 12
# EPOCH Episode
# TARGET LOOP 1-10
# MINI BATCH LOOP 2
# REAL_PDB_DIR = "/home/rajroy/Documents/rl_dataset/selected_true/"
REAL_PDB_DIR = "/home/esdft/DeepRLP/data/"
CHANGED_PDB_DIR = "/home/esdft/DeepRLP/data/"
# CHANGED_PDB_DIR = "/home/rajroy/Documents/rl_dataset/changed_pdb/"
EPOCH = 50000
NUMBER_TARGET = 0
MINI_BATCH = 2
INNER_LOOP = UPDATE_INTERVAL+1


class env():

    def __init__(self, _true_pdb_file, _initial_pdb_file):
        self.true_pose = pyrosetta.pose_from_pdb(_true_pdb_file)
        self.pose = pyrosetta.pose_from_pdb(_initial_pdb_file)

        self.original_pose = Pose()
        self.original_pose.assign(self.pose)
        self.prev_pose = Pose()
        self.prev_pose.assign(self.pose)

        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        self.n_actions = NUMBER_OF_ACTIONS

        self.all_residues_ids_chain1 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(self.pose, 'A')
        self.all_residues_ids_chain2 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(self.pose, 'B')

        # Rotation
        self.spinmover_x = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_x.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(ROTATE_INTERVAL, 0, 0))
        self.spinmover_x.angle_magnitude(ROTATE_INTERVAL)

        self.spinmover_x_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_x_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(ROTATE_INTERVAL, 0, 0))
        self.spinmover_x_backward.angle_magnitude(-ROTATE_INTERVAL)

        self.spinmover_y = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_y.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, ROTATE_INTERVAL, 0))
        self.spinmover_y.angle_magnitude(ROTATE_INTERVAL)

        self.spinmover_y_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_y_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, ROTATE_INTERVAL, 0))
        self.spinmover_y_backward.angle_magnitude(-ROTATE_INTERVAL)

        self.spinmover_z = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_z.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, ROTATE_INTERVAL))
        self.spinmover_z.angle_magnitude(ROTATE_INTERVAL)

        self.spinmover_z_backward = rigid_moves.RigidBodyDeterministicSpinMover()
        self.spinmover_z_backward.spin_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, ROTATE_INTERVAL))
        self.spinmover_z_backward.angle_magnitude(-ROTATE_INTERVAL)

        # Translation
        self.transmover_x = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_x.step_size(TRANSLATE_INTERVAL)
        self.transmover_x.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(TRANSLATE_INTERVAL, 0, 0))

        self.transmover_x_backward = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_x_backward.step_size(-TRANSLATE_INTERVAL)
        self.transmover_x_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(TRANSLATE_INTERVAL, 0, 0))

        self.transmover_y = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_y.step_size(TRANSLATE_INTERVAL)
        self.transmover_y.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, TRANSLATE_INTERVAL, 0))

        self.transmover_y_backward = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_y_backward.step_size(-TRANSLATE_INTERVAL)
        self.transmover_y_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, TRANSLATE_INTERVAL, 0))

        self.transmover_z = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_z.step_size(TRANSLATE_INTERVAL)
        self.transmover_z.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, TRANSLATE_INTERVAL))

        self.transmover_z_backward = rigid_moves.RigidBodyTransMover(self.pose, TRANSLATE_INTERVAL)
        self.transmover_z_backward.step_size(-TRANSLATE_INTERVAL)
        self.transmover_z_backward.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(0, 0, TRANSLATE_INTERVAL))

        self.rotation_x_forward = self.get_rotation_matrix('x', TRANSLATE_INTERVAL)
        self.rotation_y_forward = self.get_rotation_matrix('y', TRANSLATE_INTERVAL)
        self.rotation_z_forward = self.get_rotation_matrix('z', TRANSLATE_INTERVAL)

        self.rotation_x_backward = self.get_rotation_matrix('x', -TRANSLATE_INTERVAL)
        self.rotation_y_backward = self.get_rotation_matrix('y', -TRANSLATE_INTERVAL)
        self.rotation_z_backward = self.get_rotation_matrix('z', -TRANSLATE_INTERVAL)

    def find_dist(self, res_i, res_j):
        atm_i = 'CA' if self.pose.residue(res_i).name()[0:3] == 'GLY' else 'CB'
        atm_j = 'CA' if self.pose.residue(res_j).name()[0:3] == 'GLY' else 'CB'
        xyz_i = self.pose.residue(res_i).xyz(atm_i)
        xyz_j = self.pose.residue(res_j).xyz(atm_j)
        dist = (xyz_i - xyz_j).norm()
        return dist

    def get_distance_map(self):
        xyz_CA_chain1 = np.array([self.pose.residue(r).xyz('CA') for r in self.all_residues_ids_chain1])
        xyz_CA_chain2 = np.array([self.pose.residue(r).xyz('CA') for r in self.all_residues_ids_chain2])
        distance_map = pairwise_distances(xyz_CA_chain1, xyz_CA_chain2, metric='euclidean', n_jobs=1)
        distance_map = distance_map[:, :, tf.newaxis]
        # NOT SURE
        distance_map = distance_map / distance_map.max()

        np_s = np.zeros(state_dim).squeeze()
        temp_state = np.array(distance_map).squeeze()
        l, b = np.array(temp_state).shape
        for counter in range(0, l):
            np_s[counter, 0:b] = temp_state[counter]

        return np.expand_dims(np_s, axis=2)

    def get_rotation_matrix(self, _axis_name, _degree_magnitude):
        degree_magnitude = math.radians(_degree_magnitude)
        if _axis_name == 'x':
            rotation_matrix = np.array([[1, 0, 0], [0, cos(degree_magnitude), -sin(degree_magnitude)],
                                        [0, sin(degree_magnitude), cos(degree_magnitude)]])
        elif _axis_name == 'y':
            rotation_matrix = np.array([[cos(degree_magnitude), 0, sin(degree_magnitude)], [0, 1, 0],
                                        [-sin(degree_magnitude), 0, cos(degree_magnitude)]])
        elif _axis_name == 'z':
            rotation_matrix = np.array(
                [[cos(degree_magnitude), -sin(degree_magnitude), 0], [sin(degree_magnitude), cos(degree_magnitude), 0],
                 [0, 0, 1]])

        return rotation_matrix

    def rotatePose(self, R):
        start_A = self.pose.conformation().chain_begin(2)
        end_A = self.pose.conformation().chain_end(2)
        for r in range(start_A, end_A + 1):
            for a in range(1, len(self.pose.residue(r).atoms()) + 1):
                v = np.array([self.pose.residue(r).atom(a).xyz()[0], self.pose.residue(r).atom(a).xyz()[1],
                              self.pose.residue(r).atom(a).xyz()[2]])
                newv = R.dot(v)
                self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newv[0], newv[1], newv[2]))

    def translatePose(self, t):
        start_A = self.pose.conformation().chain_begin(2)
        end_A = self.pose.conformation().chain_end(2)
        for r in range(start_A, end_A + 1):
            for a in range(1, len(self.pose.residue(r).atoms()) + 1):
                newx = self.pose.residue(r).atom(a).xyz()[0] + t[0]
                newy = self.pose.residue(r).atom(a).xyz()[1] + t[1]
                newz = self.pose.residue(r).atom(a).xyz()[2] + t[2]
                self.pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newx, newy, newz))

    def step(self, action):

        if action == 0:
            # self.rotatePose(self.rotation_x_forward)
            self.spinmover_x.apply(self.pose)
        elif action == 1:
            # self.rotatePose(self.rotation_y_forward)
            self.spinmover_y.apply(self.pose)
        elif action == 2:
            # self.rotatePose(self.rotation_z_forward)
            self.spinmover_z.apply(self.pose)
        elif action == 3:
            # self.translatePose([1, 0, 0])
            self.transmover_x.apply(self.pose)
        elif action == 4:
            # self.translatePose([0, 1, 0])
            self.transmover_y.apply(self.pose)
        elif action == 5:
            # self.translatePose([0, 0, 1])
            self.transmover_z.apply(self.pose)
        elif action == 6:
            # self.rotatePose(self.rotation_x_backward)
            self.spinmover_x_backward.apply(self.pose)
        elif action == 7:
            # self.rotatePose(self.rotation_y_backward)
            self.spinmover_y_backward.apply(self.pose)
        elif action == 8:
            # self.rotatePose(self.rotation_z_backward)
            self.spinmover_z_backward.apply(self.pose)
        elif action == 9:
            # self.translatePose([-1, 0, 0])
            self.transmover_x_backward.apply(self.pose)
        elif action == 10:
            # self.translatePose([0, -1, 0])
            self.transmover_y_backward.apply(self.pose)
        else:
            # self.translatePose([0, 0, -1])
            self.transmover_z_backward.apply(self.pose)

        curr_ca_rmsd = CA_rmsd(self.true_pose, self.pose)

        if curr_ca_rmsd <= 1:
            done = True
            reward = 100
        elif curr_ca_rmsd >= 40:
            done = True
            reward = -40
        else:
            done = False
            diff = self.prev_ca_rmsd - curr_ca_rmsd
            reward = diff

        # print(reward)
        self.prev_ca_rmsd = curr_ca_rmsd

        # self.prev_energy = self.curr_energy

        return self.get_distance_map(), reward, done

    def reset(self):
        self.pose = self.original_pose.clone()
        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        return self.get_distance_map()

    def get_current_state(self):
        return self.get_distance_map()

    def restore_pose(self):
        self.pose = self.prev_pose.clone()
        self.prev_ca_rmsd = CA_rmsd(self.true_pose, self.pose)
        return self.get_distance_map()

    def set_prev_pose(self):
        self.prev_pose = self.pose.clone()


class ReplayMemory():
    def __init__(self, memory_size=10000):
        self.memory_size = memory_size
        # a little engineerting so that all shape can  work
        self.distance_maps = None  # shape to be determined by first distance_map
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.done = np.empty(self.memory_size, dtype=np.bool)
        self.next_state = None
        self.current_idx = 0
        self.num_data = 0
        self.protein_ids = np.empty(self.memory_size, dtype="object")

    def store_distance_map(self, _cur_state):
        if self.distance_maps is None:
            self.distance_maps = np.empty((self.memory_size,) + _cur_state.shape, dtype=np.float32)
        self.distance_maps[self.current_idx] = _cur_state

    def store_next_state(self, _next_state):
        if self.next_state is None:
            self.next_state = np.empty((self.memory_size,) + _next_state.shape, dtype=np.float32)
        self.next_state[self.current_idx] = _next_state



    def store_transition(self, action, reward, done, _iid):
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.done[self.current_idx] = done
        self.done[self.current_idx] = done
        self.protein_ids[self.current_idx] = _iid
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
        next_obs_sample = np.stack([self.next_state[idx] for idx in idxes])
        done_sample = self.done[idxes]

        return obs_sample, action_sample, reward_sample, next_obs_sample, done_sample


class DQNAgent:
    def __init__(self, _name, _state_shape, _n_actions, _epsilon=0, reuse=False):
        with tf.variable_scope(_name, reuse=reuse):
            # self.network = tf.keras.models.Sequential()
            # self.network.add(L.Conv2D(16, (3, 3), strides=[2, 2, ], activation='elu', input_shape=_state_shape))
            # self.network.add(L.Conv2D(32, (3, 3), strides=[2, 2, ], activation='elu'))
            # self.network.add(L.Conv2D(64, (3, 3), strides=[2, 2, ], activation='elu'))
            # self.network.add(L.Conv2D(128, (3, 3), strides=[2, 2, ], activation='elu'))
            # # self.network.add(L.Conv2D(32, (3, 3), strides=[2, 2, ], activation='elu'))
            # # self.network.add(L.Conv2D(128, (3, 3), strides=[2, 2, ], activation='elu'))
            # # self.network.add(L.Conv2D(64, (3, 3), strides=[2, 2, ], activation='elu'))
            # # self.network.add(L.Conv2D(32, (3, 3), strides=[2, 2, ], activation='elu'))
            # # self.network.add(L.Conv2D(16, (3, 3), strides=[2, 2, ], activation='elu'))
            # self.network.add(L.Flatten())
            # self.network.add(L.Dense(256, activation='elu'))
            # self.network.add(L.Dense(128, activation='elu'))
            # self.network.add(L.Dense(64, activation='sigmoid'))
            # self.network.add(L.Dense(_n_actions, activation='softmax'))
            self.network = resnet.ResnetBuilder.build_resnet_18((list(_state_shape)),_n_actions)
            self.state_t = tf.placeholder('float32', [None, ] + list(_state_shape))
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=_name)
        self.epsilon = _epsilon

    def get_symbolic_qvalues(self, _state_t):
        qvalues = self.network(_state_t)
        return qvalues

    def get_qvalues(self, _state_t):
        sess = tf.get_default_session()
        # np_s = np.zeros(state_dim).squeeze()
        #
        # temp_state = np.array(_state_t).squeeze()
        # l, b = np.array(temp_state).squeeze().shape
        # for counter in range(0, l):
        #     np_s[counter, 0:b] = temp_state[counter]
        return sess.run(self.qvalues_t, {self.state_t: _state_t})

    def sample_actions(self, _qvalues):
        epsilon = self.epsilon
        batch_size, _n_actions = _qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = _qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


# env = env('/home/esdft/DeepRLP/data/3HE4A_3HE4B.pdb', '/home/esdft/DeepRLP/data/3HE4A_3HE4B_GD.pdb', '/home/esdft/DeepRLP/data/3HE4A_3HE4B.rr', '/home/esdft/DeepRLP/data/talaris2013.wts')
# true_pdb = "/home/rajroy/Downloads/3HE4A_3HE4B.pdb"
# initial_pdb = "/home/rajroy/Downloads/3HE4A_3HE4B_start.pdb"
# env = env(true_pdb,initial_pdb)
# dist_map = env.get_distance_map()
# state_dim = dist_map.shape
# state_dim = (291, 291, 1)
state_dim = (61, 61, 1)

n_actions = NUMBER_OF_ACTIONS
# true_pdb = "/home/rajroy/Downloads/3HE4A_3HE4B.pdb"
# initial_pdb = "/home/rajroy/Downloads/3HE4A_3HE4B_start.pdb"
# enviroment = env(true_pdb, initial_pdb)
# dist_map = enviroment.get_distance_map()
# state_dim = dist_map.shape
# n_actions = enviroment.n_actions

agent = DQNAgent("dqn_agent", state_dim, n_actions, _epsilon=0.8)
sess.run(tf.global_variables_initializer())

replay = ReplayMemory()


def evaluate(_env, _agent, _n_games=1, _greedy=False, _t_max=10000, _target_id="none"):
    rewards = []
    for _ in range(_n_games):
        s = _env.reset()
        reward = 0
        for _ in range(_t_max):
            qvalues = _agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if _greedy else _agent.sample_actions(qvalues)[0]
            s, r, done = _env.step(action)

            # print(action, r, done)
            reward += r
            if done: break

        rewards.append(reward)
    mean_reward = np.mean(rewards)
    with open(os.getcwd() + "/rewards_" + str(_target_id) + ".txt", 'a') as f:
        f.write(str(mean_reward))
        f.write('\t')
        f.write(str(_env.prev_ca_rmsd))
        # f.write('\t')
        # f.write(str(env.prev_energy))
        f.write('\n')
    print(mean_reward)
    print(_env.prev_ca_rmsd)
    # print(env.prev_energy)
    return mean_reward


# evaluate(env, agent, n_games=20)
# print(env.prev_ca_rmsd)

def play_and_record(_agent, _env, _exp_replay, _initial_state, _n_steps=1, _iid="none"):
    # Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    # Whenever game ends, add record with done=True and reset the game.
    # It is guaranteed that env has done=False when passed to this function.
    # PLEASE DO NOT RESET ENV UNLESS IT IS "DONE":returns: return sum of rewards over time//
    # current_state = _initial_state
    current_state = _env.get_current_state()
    total_reward = 0.0

    for t in range(_n_steps):
        qs = agent.get_qvalues([current_state])
        a = agent.sample_actions(qs)[0]
        exp_replay.store_distance_map(current_state)

        next_s, r, done = _env.step(a)
        exp_replay.store_transition(a, r, done, _iid)
        total_reward += r
        if done:
            current_state = _env.reset()
            exp_replay.store_next_state(_env.reset())
        else:
            current_state = next_s
            exp_replay.store_next_state(next_s)

    return total_reward


target_network = DQNAgent("target_network", state_dim, n_actions)


def load_weigths_into_target_network(_agent, _target_network):
    assigns = []
    for w_agent, w_target in zip(_agent.weights, _target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)


obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
actions_ph = tf.placeholder(tf.int32, shape=[None])
rewards_ph = tf.placeholder(tf.float32, shape=[None])
next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
is_done_ph = tf.placeholder(tf.float32, shape=[None])

is_not_done = 1 - is_done_ph
gamma = 0.99
# Take q-values for actions agent just took
current_qvalues = agent.get_symbolic_qvalues(obs_ph)
current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)
# Compute Q-learning TD error:
# L=1N∑i[Qθ(s,a)−Qreference(s,a)]**2
# With Q-reference defined as
# Qreference(s,a)=r(s,a)+γ⋅maxa′Qtarget(s′,a′)
next_qvalues_target = target_network.get_symbolic_qvalues(next_obs_ph)
# compute state values by taking max over next_qvalues_target for all actions
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

# moving_average = lambda x, span, **kw: DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values

mean_rw_history = []
td_loss_history = []

exp_replay = ReplayMemory(10 ** 3)


# play_and_record(agent, enviroment, exp_replay, enviroment.get_current_state(), _n_steps=10000)
#

def sample_batch(_exp_replay, _batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = _exp_replay.sample(_batch_size)
    return {
        obs_ph: obs_batch, actions_ph: act_batch, rewards_ph: reward_batch,
        next_obs_ph: next_obs_batch, is_done_ph: is_done_batch
    }


TARGET_LIST = ["1PD3A_1PD3B"]
# EPOCH Episode
# TARGET LOOP 1-10
# MINI BATCH LOOP 2
# INNER_LOOP

OMEGA_EPISLON = 1
for _ep in range(EPOCH):
    random.shuffle(TARGET_LIST)
    for _target_name in TARGET_LIST:
        # reinitialize
        enviroment = env(REAL_PDB_DIR + _target_name + ".pdb", CHANGED_PDB_DIR + _target_name + ".start.pdb")
        dist_map = enviroment.get_distance_map()
        # state_dim = dist_map.shape
        n_actions = enviroment.n_actions
        # s = enviroment.restore_pose()
        OMEGA_EPISLON = (OMEGA_EPISLON - (_ep / EPOCH) )* 0.9
        # agent.epsilon = max(1 - (_ep / EPOCH) * 0.9, 0.01)
        agent.epsilon = max(OMEGA_EPISLON, 0.01)
        for i in trange(INNER_LOOP):
            # play
            # s = enviroment.restore_pose()
            # #how many steps ??
            # MAY HELP

            play_and_record(_agent=agent, _env=enviroment, _exp_replay=exp_replay,
                            _initial_state=enviroment.get_distance_map(), _n_steps=10, _iid=_target_name)
            # enviroment.set_prev_pose()

            # train
            if i > BATCH_SAMPLE_SIZE:
                _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, _batch_size=BATCH_SAMPLE_SIZE))
                td_loss_history.append(loss_t)

            # adjust agent parameters
            if i % UPDATE_INTERVAL == 0 and i != 0:
                # saver = tf.train.Saver()
                # checkpoint_path = "/home/rajroy/wieghts/dqn_cp.ckpt"
                # saver.save(sess, checkpoint_path)
                load_weigths_into_target_network(agent, target_network)
                # FIX THIS
                # agent.epsilon = max(agent.epsilon * 0.99, 0.01)

                mean_rw_history.append(evaluate(enviroment, agent, _n_games=1, _target_id=_target_name))

            if i % 100 == 0 and i != 0 and i > BATCH_SAMPLE_SIZE:
                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" % (exp_replay.memory_size, agent.epsilon))
                assert not np.isnan(loss_t)
