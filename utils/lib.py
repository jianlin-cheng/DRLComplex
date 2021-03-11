import copy
import os

import numpy as np


class pdb_lines:
    # class fot he pdb details
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

def distance_calculation(_cord_1, _cord_2):
    return (((_cord_1.x - _cord_2.x) ** 2) + ((_cord_2.y - _cord_2.y) ** 2) + (
            (_cord_2.z - _cord_2.z) ** 2)) ** 0.5


def dist_map_maker(_agent_1, _agent_2):
    temp_dst = np.zeros((len(_agent_1),len(_agent_2)))
    for cord_1 in range(0, len(_agent_1)):
        for cord_2 in range(0, len(_agent_2)):
            temp_dst[cord_1][cord_2] = distance_calculation(_agent_1[cord_1], _agent_2[cord_2])
    return temp_dst

def correct_format(_pdb_row):
    # formats the class to string properly
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


def space_returner(_input):
    # returns proper space for the  pdbs
    i = 0
    space = ""
    while i < _input:
        space = space + " "
        i = i + 1
    return space


def pdb_lines_to_string(_pdb):
    # converts pdb class to a string
    out_string = ""
    for val in _pdb:
        out_string += correct_format(val) + "\n"
    out_string += "END"
    return out_string


def contents_to_info(_in_contents):
    # reads the ATOM line. Then splits the info into respective frames and returns the data
    split_contents = []
    contents = _in_contents.splitlines()
    for lines in contents:
        if lines.startswith("ATOM"):
            pdb_line = split_line_to_tuple(lines.strip())
            split_contents.append(pdb_line)
    return split_contents


def split_line_to_tuple(line):
    # reads string int pdb class
    a_pdb_line = pdb_lines()
    a_pdb_line.atom = line[0:6].strip()
    a_pdb_line.serial = line[6:12].strip()
    a_pdb_line.atom_name = line[12:16].strip()
    a_pdb_line.alt_loc = line[16].strip()
    a_pdb_line.res_name = line[17:20]  # for this not parsing everything as not needed
    a_pdb_line.chain = line[20:22].strip()
    a_pdb_line.res_num = line[22:26].strip()
    a_pdb_line.icode = line[26:30].strip()
    a_pdb_line.x =float( line[30:38].strip())
    a_pdb_line.y =float( line[38:46].strip())
    a_pdb_line.z = float(line[46:54].strip())
    a_pdb_line.occupancy = line[54:60].strip()
    a_pdb_line.temp_fact = line[60:66].strip()
    a_pdb_line.element = line[76:78].strip()
    a_pdb_line.charge = line[78:80].strip()
    return a_pdb_line


def separate_by_chain(_pdb, _name):
    #separates cahins from pdb
    result = list(filter(lambda x: (x.chain == _name), _pdb))
    return result


def file_reader(input_dir):
    #reads pdb
    _input_dir = copy.deepcopy(input_dir)
    contents = ""
    f = open(_input_dir, "r")
    if f.mode == 'r':
        contents = f.read()
        f.close()
    # return contents.split("END")  # CAN BE TER TOO
    return contents


def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        if _cont[len(_cont) - 1].strip() != "END":
            f.write("END")
    return


def specific_dir_reader(_input_dir):
    file_names = []
    i = 0
    for root, directories, files in os.walk(_input_dir):
        i = i + 1
        file_names.append(directories)

    return file_names[0]


# 3L8NA_3L8MB

# _input_pdb = "/home/rajroy/q3_het30/pdbs/3L8MA_3L8MB.pdb"
_input_pdb = "/home/rajroy/2AJ9.pdb"
output_file = "/home/rajroy/"

complex_model = file_reader(_input_pdb)  # reading pdb files with multiple chains

#
# model_2= separate_by_chain(contents_to_info(complex_model),"A")


# this seperates without chains uses waords like END or TER can be changed with requirements
# model_no_chain_1 = complex_model.split("TER")[0]
# pdb_model_1 = contents_to_info(model_no_chain_1)  # if theres 2 chains it will give you one of them
# _pdb_1 = list(filter(lambda x: (x.atom_name == "CA"), pdb_model_1)) # filtering all but CA atoms

# this can seperate by chains names
pdb_model_1 =  separate_by_chain(contents_to_info(complex_model),"A")
_pdb_1 = list(filter(lambda x: (x.atom_name == "CA"), pdb_model_1)) # filtering all but CA atoms


pdb_model_2 =  separate_by_chain(contents_to_info(complex_model),"B") # if theres 2 chains it will give you one of them
_pdb_2 = list(filter(lambda x: (x.atom_name == "CA"), pdb_model_2))


dist_map_maker(_pdb_1,_pdb_2)


# if you want to save pdb
# file_name_a = output_file + "/sample" + ".atom"
# write2File(file_name_a, pdb_lines_to_string(_pdb))


#For distance map generation