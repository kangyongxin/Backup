# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from copy import deepcopy as dc
from collections import Counter
import matplotlib.pyplot as plt

#这个能用在以图形输入为基础的实验中吗
def reset():
    return 6
def env1(x,action):
    """
        12  13
         |   |
     6---7---8---9
         |   |
         2   3
    :param x:
    :param action:
    :return:
    """
    oo = []
    # oo.append(1)
    oo.append(2)
    oo.append(3)
    # oo.append(4)
    # oo.append(5)
    oo.append(6)
    oo.append(7)
    oo.append(8)
    oo.append(9)
    # oo.append(11)
    oo.append(12)
    oo.append(13)

    y = x
    if y != 2 and y != 6 and y != 12 and y!=13 and y!=3 and action == 0:#左侧没有状态
        y = y - 1
    elif y != 3 and y != 9 and y != 13 and y!=12 and y!=2 and action == 1:#右侧没有状态
        y = y + 1
    elif y !=6 and y!=2 and y!=3 and y!=9 and action == 2:#下面没有状态
        y = y - 5
    elif y !=6 and y!=12 and y!=13 and y!=9 and action == 3:#上面没有状态
        y = y + 5

    return y,oo

def env0(x,action):
    '''
     11---12---13
      |         |
      6--- 7--- 8--- 9 ---10
      |         |
      1--- 2----|
    :param x:
    :param action:
    :return:
    '''

    oo = []
    oo.append(1)
    oo.append(2)
    #oo.append(3)
    # oo.append(4)
    # oo.append(5)
    oo.append(6)
    oo.append(7)
    oo.append(8)
    oo.append(9)
    oo.append(10)
    oo.append(11)
    oo.append(12)
    oo.append(13)
    y = x
    if y != 1 and y != 6 and y != 11 and action == 0:
        y = y - 1
    elif y != 2 and y != 10 and y != 13 and action == 1:
        y = y + 1
    elif y > 5 and y != 8 and y != 9 and y != 10 and y != 7 and y != 12 and action == 2:
        y = y - 5
    elif y < 11 and y != 9 and y != 10 and y != 7 and y != 2 and action == 3:
        y = y + 5
    elif y == 8 and action == 4:
        y = y - 6
    elif y == 2 and action == 5:
        y = y + 6

    return y,oo

def step(x, action):
    y,_=env0(x,action)
    return y


def get(seq):
    print("trj,",seq)
    u_state = []# list of the unique state
    count_state = []# list of the number of state
    k = 0
    for i in seq:
        if u_state.count(i) == 0:
            count_state.append(seq.count(i))
            u_state.append(i)
            k = k + 1
    print("u_s",u_state)
    print("count_s",count_state)
    return u_state, count_state


def get1(seq):
    a = Counter(seq)

    return a


_, oo = env0(6,0)#初始化环境的
a = []
e = []
r = []
n_state= len(oo)
ll = 10000
trj_len =20
for j in range(ll):
    trj = []
    state = reset()
    for i in range(trj_len):
        trj.append(state)
        action = np.random.randint(0, 4)
        state = step(state, action)
    [d, g] = get(trj)
    [u_state, count_state]=get(trj)

    a.append(u_state)
    e.append(trj)
    r.append(count_state) #每条轨迹的统计特性。状态出现的次数

p = np.zeros((n_state, 1))
u = np.zeros((n_state, 1))
uo = np.zeros((n_state, 1))
for j in range(ll):#第j条轨迹
    for i in range(len(oo)):#某个状态
        for w in range(len(a[j])):#第w个 出现
            if a[j][w] == oo[i]:
                p[i] = p[i] + sum(r[j][0:w])#第j条轨迹中第i个状态出现之前所有的状态频率统计
                u[i] = u[i] + w #出现的顺序 越晚月重要？轨迹中
                uo[i] = uo[i] + r[j][w]#第 i 个状态在所有轨迹中出现的次数总和
print("oo", oo)
print("p", p)
print("u", u)
print("uo", p/uo)
# plt.plot(oo,p)
# plt.show()
plt.plot(oo, p/uo)
plt.show()