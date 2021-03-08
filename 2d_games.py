import random
import time


def distance_calculation():
    return None


def state_generation():
    return None


def reward_calculation():
    return None


class cord_2d:
    x_cord = 0
    y_cord = 0

    pass


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

plt.style.use('fivethirtyeight')
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)




def render_3(i):
    cord_x_1 = []
    cord_y_1 = []
    cord_x_2 = []
    cord_y_2 = []


    cord_data_1 = open('cord_1.txt', 'r').read()
    lines = cord_data_1.split('\n')
    for line in lines:
        if len(line )> 1:
            x, y = line.split(",")

            cord_x_1.append(float(x))
            cord_y_1.append(float(y))
        # ax1.clear()
        # ax1.scatter(cord_x_1, cord_y_1, 'o', color='green')


    cord_data_2 = open('cord_2.txt', 'r').read()
    lines = cord_data_2.split('\n')
    for line in lines:
        if len(line) > 1:
            x, y = line.split(",")

            cord_x_2.append(float(x))
            cord_y_2.append(float(y))
    plt.cla()

    plt.plot(cord_x_1, cord_y_1, marker='o', color='g',linewidth=2)
    plt.plot(cord_x_2, cord_y_2, marker='x', color='b',linewidth=2)

def get_random_cords(_len):
    # l=_len
    cord_list = []
    cord_x=[]
    cord_y=[]
    cord_data= open('cord_1.txt','r').read()
    lines=cord_data.split('\n')
    for val in  range(0,l):
        temp_cord = cord_2d()
        temp_cord.x_cord = random.randrange(1, 100)
        temp_cord.y_cord = random.randrange(1, 100)
        cord_list.append(temp_cord)


    return cord_list

l = 10

episodes=100


ani=animation.FuncAnimation(plt.gcf(),render_3,interval=100)

plt.show()