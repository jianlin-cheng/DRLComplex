import numpy as np

#accepts 3d_corrdinates of backbone
def distance_calculation(_cord_1, _cord_2):
    return (((_cord_1[0] - _cord_2[0]) ** 2) + ((_cord_2[1] - _cord_2[1]) ** 2) + (
            (_cord_2[2] - _cord_2[2]) ** 2)) ** 0.5


def dist_map_maker(_agent_1, _agent_2, _max_len):
    temp_dst = np.zeros((_max_len, _max_len))
    for cord_1 in range(0, len(_agent_1)):
        for cord_2 in range(0, len(_agent_2)):
            temp_dst[cord_1][cord_2] = distance_calculation(_agent_1[cord_1], _agent_2[cord_2])

    return temp_dst


agent_1 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0],[5, 0, 0], [6, 0, 0], [9, 0, 0]]).astype(float)
agent_2 =np.array ([[10, 0, 0], [12, 0, 0], [13, 0, 0],[15, 0, 0], [16, 0, 0], [19, 0, 0], [4, 0, 0], [8, 0, 0]]).astype(float)
#agent_2 will be in row and agent_1 in col
dist_map_maker(agent_1, agent_2, 10)
