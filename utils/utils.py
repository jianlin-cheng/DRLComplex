import copy
import random
import subprocess
import numpy as np
import math
import keras
import time

# from keras import layers
# from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten, Dense
# from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.optimizers import Adam


class experience_replay:
    state = []
    action = 0
    reward = 0.0
    next_state = []
    done = False
    pass


class pdb_lines:
    atom = ''
    serial = ''
    atom_name = ''
    alt_loc = ''
    res_name = ''
    chain = ''
    res_num = ''
    icode = ''
    x = ''
    y = ''
    z = ''
    occupancy = ''
    temp_fact = ''
    element = ''
    charge = ''

    pass


TMALIGN_PATH = "/home/rajroy/Downloads/tools/TMalign"


def get_tm_align_score(_true, _current):
    contents = subprocess.check_output([TMALIGN_PATH, _true, _current])
    tmscore = ""
    for item in contents.decode("utf-8").split("\n"):
        if "TM-score=" in item:
            tmscore = item.strip().split(",")[2].strip().split("=")[1]

    # print(tmscore)
    return tmscore


def read_pdb(pdb):
    contents = []
    with open(pdb, "r") as f:
        for line in f:
            # if (line.startswith("ATOM")):
            #    pass
            contents.append(line)
    return contents


def split_line_to_tuple(line):
    a_pdb_line = pdb_lines()

    a_pdb_line.atom = line[0:6].strip()
    a_pdb_line.serial = line[6:12].strip()
    a_pdb_line.atom_name = line[12:16].strip()
    a_pdb_line.alt_loc = line[16].strip()
    a_pdb_line.res_name = line[17:20].strip()
    a_pdb_line.chain = line[20:22].strip()
    a_pdb_line.res_num = line[22:26].strip()
    a_pdb_line.icode = line[26:30].strip()
    a_pdb_line.x = line[30:38].strip()
    a_pdb_line.y = line[38:46].strip()
    a_pdb_line.z = line[46:54].strip()
    a_pdb_line.occupancy = line[54:60].strip()
    # a_pdb_line.temp_fact = line[60:76].strip()
    a_pdb_line.temp_fact = line[60:66].strip()
    a_pdb_line.element = line[76:78].strip()
    a_pdb_line.charge = line[78:80].strip()

    return a_pdb_line


# https://en.wikipedia.org/wiki/Rotation_matrix

def rotation_x(_degree, _input):
    x_matrix = [[1, 0, 0], [0, math.cos(_degree), -math.sin(_degree)], [0, math.sin(_degree), math.cos(_degree)]]

    for val in _input:
        if val:
            x = float(val.x)
            y = float(val.y)
            z = float(val.z)
            values = [[x], [y], [z]]
            new_values = np.dot(x_matrix, values)
            val.x = format(float(new_values[0]), '.3f')
            val.y = format(float(new_values[1]), '.3f')
            val.z = format(float(new_values[2]), '.3f')

    return _input


def rotation_y(_degree, _input):
    y_matrix = [[math.cos(_degree), 0, math.sin(_degree)], [0, 1, 0], [-math.sin(_degree), 0, math.cos(_degree)]]

    for val in _input:
        if val:
            x = float(val.x)
            y = float(val.y)
            z = float(val.z)
            values = [[x], [y], [z]]
            new_values = np.dot(y_matrix, values)
            val.x = format(float(new_values[0]), '.3f')
            val.y = format(float(new_values[1]), '.3f')
            val.z = format(float(new_values[2]), '.3f')

    return _input


def rotation_z(_degree, _input):
    z_matrix = [[math.cos(_degree), -math.sin(_degree), 0], [math.sin(_degree), math.cos(_degree), 0], [0, 0, 1]]
    for val in _input:
        if val:
            x = float(val.x)
            y = float(val.y)
            z = float(val.z)
            values = [[x], [y], [z]]
            new_values = np.dot(z_matrix, values)
            val.x = format(float(new_values[0]), '.3f')
            val.y = format(float(new_values[1]), '.3f')
            val.z = format(float(new_values[2]), '.3f')

    return _input
def contents_to_info_splitlines(contents):
    # reads the ATOM line. Then splits the info into respective frames and returns the data
    split_contents = []
    for lines in contents.splitlines():
        if lines.startswith("ATOM"):
            pdb_line = split_line_to_tuple(lines.strip())
            split_contents.append(pdb_line)
    return split_contents


def contents_to_info(contents):
    # reads the ATOM line. Then splits the info into respective frames and returns the data
    split_contents = []
    for lines in contents:
        if lines.startswith("ATOM"):
            pdb_line = split_line_to_tuple(lines.strip())
            split_contents.append(pdb_line)
    return split_contents


def space_returner(_input):
    i = 0
    space = ""
    while i < _input:
        space = space + " "
        i = i + 1
    return space

def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        if _cont[len(_cont) - 1].strip() != "END":
            f.write("END")
    return
def string_line_from_pdb_array(_pdb_row):
    # _pdb_copy = copy.deepcopy(_pdb_row)
    # # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    # _pdb_copy.atom = _pdb_copy.atom  # 1-4
    # _pdb_copy.serial = space_returner(4 - len(str(_pdb_copy.serial))) + str(_pdb_copy.serial)  # 7-11
    # _pdb_copy.atom_name = _pdb_copy.atom_name + space_returner(3 - len(_pdb_copy.atom_name))  # 13-16
    # _pdb_copy.alt_loc = space_returner(1 - len(_pdb_copy.alt_loc)) + _pdb_copy.alt_loc  # 17
    # _pdb_copy.res_name = space_returner(3 - len(_pdb_copy.res_name)) + _pdb_copy.res_name  # 18-20
    # _pdb_copy.chain = space_returner(1 - len(_pdb_copy.chain)) + _pdb_copy.chain  # 22
    # _pdb_copy.res_num = space_returner(4 - len(_pdb_copy.res_num)) + _pdb_copy.res_num  # 23-26
    # _pdb_copy.icode = space_returner(2 - len(_pdb_copy.chain)) + _pdb_copy.icode  # 27
    # _pdb_copy.x = space_returner(8 - len(_pdb_copy.x)) + _pdb_copy.x  # 31-38
    # _pdb_copy.y = space_returner(8 - len(_pdb_copy.y)) + _pdb_copy.y  # 39-46
    # _pdb_copy.z = space_returner(8 - len(_pdb_copy.z)) + _pdb_copy.z  # 47-54
    # _pdb_copy.occupancy = space_returner(6 - len(_pdb_copy.occupancy)) + _pdb_copy.occupancy  # 55-60
    # _pdb_copy.temp_fact = space_returner(6 - len(_pdb_copy.temp_fact)) + _pdb_copy.temp_fact  # 61-66
    # _pdb_copy.element = space_returner(4 - len(_pdb_copy.element)) + _pdb_copy.element  # 73-76
    # _pdb_copy.charge = space_returner(2 - len(_pdb_copy.charge)) + _pdb_copy.charge  # 77-78
    # content = _pdb_copy.atom + space_returner(3) + _pdb_copy.serial + space_returner(
    #     2) + _pdb_copy.atom_name + _pdb_copy.alt_loc + _pdb_copy.res_name + space_returner(
    #     1) + _pdb_copy.chain + _pdb_copy.res_num + _pdb_copy.icode + space_returner(
    #     3) + _pdb_copy.x + _pdb_copy.y + _pdb_copy.z + _pdb_copy.occupancy + _pdb_copy.temp_fact + space_returner(
    #     6) + _pdb_copy.element
    _pdb_copy = copy.deepcopy(_pdb_row)
    # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    _pdb_copy.atom = _pdb_copy.atom  # 1-4
    _pdb_copy.serial = space_returner(5 - len(str(_pdb_copy.serial))) + str(_pdb_copy.serial)  # 7-11
    _pdb_copy.atom_name = _pdb_copy.atom_name + space_returner(3 - len(_pdb_copy.atom_name))  # 13-16
    _pdb_copy.alt_loc = space_returner(1 - len(_pdb_copy.alt_loc)) + _pdb_copy.alt_loc  # 17
    _pdb_copy.res_name = space_returner(3 - len(_pdb_copy.res_name)) + _pdb_copy.res_name  # 18-20
    _pdb_copy.chain = space_returner(1 - len(_pdb_copy.chain)) + _pdb_copy.chain  # 22
    _pdb_copy.res_num = space_returner(4 - len(_pdb_copy.res_num)) + _pdb_copy.res_num  # 23-26
    _pdb_copy.icode = space_returner(2 - len(_pdb_copy.chain)) + _pdb_copy.icode  # 27
    _pdb_copy.x = space_returner(8 - len(_pdb_copy.x)) + _pdb_copy.x  # 31-38
    _pdb_copy.y = space_returner(8 - len(_pdb_copy.y)) + _pdb_copy.y  # 39-46
    _pdb_copy.z = space_returner(8 - len(_pdb_copy.z)) + _pdb_copy.z  # 47-54
    _pdb_copy.occupancy = space_returner(6 - len(_pdb_copy.occupancy)) + _pdb_copy.occupancy  # 55-60
    _pdb_copy.temp_fact = space_returner(6 - len(_pdb_copy.temp_fact)) + _pdb_copy.temp_fact  # 61-66
    _pdb_copy.element = space_returner(4 - len(_pdb_copy.element)) + _pdb_copy.element  # 73-76
    _pdb_copy.charge = space_returner(2 - len(_pdb_copy.charge)) + _pdb_copy.charge  # 77-78
    content = _pdb_copy.atom + space_returner(2) + _pdb_copy.serial

    if len(_pdb_copy.atom_name) < 4:
        content = content + space_returner(2) + _pdb_copy.atom_name
    elif len(_pdb_copy.atom_name) == 4:
        content = content + " " + _pdb_copy.atom_name

    content = content + _pdb_copy.alt_loc + _pdb_copy.res_name + space_returner(
        1) + _pdb_copy.chain + _pdb_copy.res_num + _pdb_copy.icode + space_returner(
        3) + _pdb_copy.x + _pdb_copy.y + _pdb_copy.z + _pdb_copy.occupancy + _pdb_copy.temp_fact + space_returner(
        8) + _pdb_copy.element + _pdb_copy.charge

    return content

def pdb_from_array_without_end(_pdb):
    array = []
    content = ''
    for x in _pdb:
        val = string_line_from_pdb_array(x)
        # array.append(val)
        content = content + val + '\n'
    # f = open(_filename, "w")
    # f.write(content + 'END')
    # f.close()
    return content

def pdb_from_array(_pdb):
    array = []
    content = ''
    for x in _pdb:
        val = string_line_from_pdb_array(x)
        # array.append(val)
        content = content + val + '\n'
    # f = open(_filename, "w")
    # f.write(content + 'END')
    # f.close()
    return content+ 'END'


def translate_xyz(_x_change, _y_change, _z_change, _input):
    temp_input = copy.deepcopy(_input)
    for val in temp_input:
        if val:
            val.x = format(float(val.x) + _x_change, '.3f')
            val.y = format(float(val.y) + _y_change, '.3f')
            val.z = format(float(val.z) + _z_change, '.3f')
    return temp_input


def translate_x(_x_change, _input):
    temp_input = copy.deepcopy(_input)
    for val in temp_input:
        if val:
            val.x = format(float(val.x) + _x_change, '.3f')
    return temp_input


def translate_y(_y_change, _input):
    temp_input = copy.deepcopy(_input)
    for val in temp_input:
        if val:
            val.y = format(float(val.y) + _y_change, '.3f')
    return temp_input


def translate_z(_z_change, _input):
    temp_input = copy.deepcopy(_input)
    for val in temp_input:
        if val:
            val.z = format(float(val.z) + _z_change, '.3f')
    return temp_input


def get_backbone_only(_agent):
    temp_agent = copy.deepcopy(_agent)
    output_temp_agent = []
    for values in temp_agent:
        if values.res_name == "GLY":
            if values.atom_name == "CA":
                output_temp_agent.append(values)
        elif values.atom_name == "CB":
            output_temp_agent.append(values)

    return output_temp_agent


# pdb_file = '/home/rajroy/Downloads/3HE4A_3HE4B.pdb'
#
# model= contents_to_info(read_pdb(pdb_file))
#
# final_array = ""
# for i in range(0,100):
#     print(i)
#     x_translate = random.randrange(co_start, co_end)
#     y_translate = random.randrange(co_start, co_end)
#     z_translate = random.randrange(co_start, co_end)
#     x_rotate =  math.radians(random.randrange(rot_start, rot_end))
#     y_rotate =  math.radians(random.randrange(rot_start, rot_end))
#     z_rotate = math.radians( random.randrange(rot_start, rot_end))
#     file_content_array = []
#     # model = []
#     # model = contents_to_info(read_pdb(model))
#
#     translated_array = translate_new(x_translate, y_translate, z_translate, copy.deepcopy(model))
#     rotated_x = []
#     rotated_x = rotation_x(x_rotate, translated_array)
#     rotated_y = []
#     rotated_y = rotation_y(y_rotate, rotated_x)
#     rotated_z = []
#     rotated_z = rotation_z(z_rotate, rotated_y)
#     final_array =rotated_z
#
#     str_pdb= pdb_from_array(final_array,"/home/rajroy/90_r.pdb")

def get_chains(_agent_1):
    temp = copy.deepcopy(_agent_1)
    chain_list = []
    for _chains in temp:
        chain_list.append(_chains.chain)
    chain_list = list(dict.fromkeys(chain_list))
    return chain_list


def distance_calculator(_cord_xyz_1, _cord_xyz_2):
    return (
                   ((float(_cord_xyz_1.x) - float(_cord_xyz_2.x)) ** 2) +
                   ((float(_cord_xyz_1.y) - float(_cord_xyz_2.y)) ** 2) +
                   ((float(_cord_xyz_1.z) - float(_cord_xyz_2.z)) ** 2)
           ) ** 0.5


def separate_by_chain(_pdb, _name):
    # print(_pdb)
    result = list(filter(lambda x: (x.chain == _name), _pdb))
    return result


def get_state(_agent_1, _agent_2, _max_l):
    temp_1 = copy.deepcopy(get_backbone_only(_agent_1))
    temp_2 = copy.deepcopy(get_backbone_only(_agent_2))
    out_put = np.zeros((_max_l, _max_l))
    for _agent_1_counter in range(0, len(temp_1)):
        for _agent_2_counter in range(0, len(temp_2)):
            out_put[_agent_1_counter][_agent_2_counter] = distance_calculator(temp_1[_agent_1_counter],
                                                                              temp_2[_agent_2_counter])

    return out_put

def backbone_rmsd (_agent_1, _agent_2):
    # temp_1 = copy.deepcopy(get_backbone_only(_agent_1))
    temp_1 = copy.deepcopy(_agent_1)
    # temp_2 = copy.deepcopy(get_backbone_only(_agent_2))
    temp_2 = copy.deepcopy(_agent_2)
    rmsd = 0
    for _agent_1_counter in range(0, len(temp_1)):
        # rmsd  += distance_calculator(temp_1[_agent_1_counter],temp_2[_agent_1_counter])**2
        rmsd += (
                ((float(temp_1[_agent_1_counter].x) - float(temp_2[_agent_1_counter].x)) ** 2) +
                ((float(temp_1[_agent_1_counter].y) - float(temp_2[_agent_1_counter].y)) ** 2) +
                ((float(temp_1[_agent_1_counter].z) - float(temp_2[_agent_1_counter].z)) ** 2)
        )
    return  (rmsd/len(temp_1))**0.5