
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
from contact_cmp import *        



predicted_pose = pyrosetta.pose_from_pdb('/home/esdft/DeepRLP/data/1B70A_1B70B_GD.pdb')
restraint_file = '/home/esdft/DeepRLP/data/1B70A_1B70B.rr'
true_contacts = read_res(predicted_pose, restraint_file)


print(fnat(predicted_pose, true_contacts))





