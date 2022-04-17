
import sys
from rosetta import *
from pyrosetta import *
from rosetta.protocols.rigid import *
from rosetta.core.scoring import *
from pyrosetta import PyMOLMover
from rosetta.protocols.rigid import *
import pyrosetta.rosetta.protocols.rigid as rigid_moves
import pyrosetta.rosetta.protocols.rigid as rigid_moves

init()


import os
import sys
import glob
import numpy as np
from math import sin, cos
import math
import random
from sklearn.metrics import pairwise_distances
        
        
        
def get_distance_map(given_pose):

    all_residues_ids_chain1 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(given_pose, 'A')
    all_residues_ids_chain2 = pyrosetta.rosetta.core.pose.get_resnums_for_chain(given_pose, 'B')
    
    xyz_CA_chain1 = np.array([given_pose.residue(r).xyz('CA') if given_pose.residue(r).name()[0:3] == 'GLY' else given_pose.residue(r).xyz('CB') for r in all_residues_ids_chain1])
    xyz_CA_chain2 = np.array([given_pose.residue(r).xyz('CA') if given_pose.residue(r).name()[0:3] == 'GLY' else given_pose.residue(r).xyz('CB') for r in all_residues_ids_chain2])
    
    
    distance_map = pairwise_distances(xyz_CA_chain1, xyz_CA_chain2, metric='euclidean', n_jobs=1)
    
    



    return distance_map
        
 

def read_res(given_pose, res_file):
    
    start_A = given_pose.conformation().chain_begin(1)
    end_A = given_pose.conformation().chain_end(1)
    start_B = given_pose.conformation().chain_begin(2)
    end_B = given_pose.conformation().chain_end(2)
    distance_map = np.zeros(shape=(end_A-start_A+1,end_B-start_B+1))
    
    filename = res_file
    with open(filename) as f:
        content = f.readlines()

    lines = [x.rstrip() for x in content]
    
    

    for i in range(3, len(lines)):
        data = lines[i].split()
        
        res_x = int(data[0])
        res_y = int(data[1])
        lb = float(data[2])
        up = float(data[3])
        dist = float(data[4])
        
        if dist >= 0.5:
            distance_map[res_x-1, res_y-1] = 1
            
        else:
            distance_map[res_x-1, res_y-1] = 0
            
    
    
    return distance_map


def fnat(predicted_pose, true_dist):

    dist_map_predicted = get_distance_map(predicted_pose)
    dist_map_predicted[dist_map_predicted <= 8] = 1
    dist_map_predicted[dist_map_predicted > 8] = -1




    diff = dist_map_predicted - true_dist

    solutions = np.argwhere(diff == 0)


    return solutions.shape[0] / np.argwhere(dist_map_predicted == 1).shape[0] 






