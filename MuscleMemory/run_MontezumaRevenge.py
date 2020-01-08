"""
1. just run it to test  whether we can encode the state in a simple way 

"""
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
#env = gym.make('FetchPush-v1')
#env = gym.make('HandManipulateBlock-v0')
def find_cnts(gray):
    a = np.array(gray>100,dtype = 'bool')
    c=np.nonzero(a)
    #print("c",c)
    cnt=[np.mean(c[0]),np.mean(c[1])]
    #print("cnt",cnt)
    return cnt
"""
"""
def state_to_obs(state):
    temp = np.zeros([210,160])
    temp [state[0]*10:(state[0]+1)*10,state[1]*10:(state[1]+1)*10] = np.ones([10,10])*255
    state_mask = temp.copy()
    return state_mask
def show_state(state):
    state_mask=state_to_obs(state)
    observation = obs[:,:,0]+state_mask #np.array([obs[:,:,0]+state_mask , obs[:,:,1]+state_mask , obs[:,:,2]+state_mask])
    #observation = observation.T
    #print(observation.shape)
    plt.imshow(observation,cmap = plt.cm.gray)
    plt.show()

def Obs2State(obs):
    b,g,r =cv2.split(obs)
    ret, thresh1 = cv2.threshold(b, 210, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(b, 190, 255, cv2.THRESH_BINARY)
    thresh3 = thresh2 -thresh1
    """
    #作掩码，掩去上面表分部分，值提取小人位置，大约 0～30的高度
    """
    mask = np.vstack((np.zeros([30,160]),np.ones([180,160])))
    thresh3 = thresh3 * mask
    # plt.subplot(2,2,1)
    # plt.imshow(thresh1,cmap = plt.cm.gray)
    # plt.subplot(2,2,2)
    # plt.imshow(thresh2,cmap = plt.cm.gray)
    # plt.subplot(2,2,3)
    # plt.imshow(thresh3,cmap = plt.cm.gray)

    cnt=find_cnts(thresh3)
    state =[ int(cnt[0]/10),int(cnt[1]/10)]
    print(state)
    show_state(state)
    return state


env=gym.make('MontezumaRevengeNoFrameskip-v4')

env.reset()
for i in range(100):
    #a= random.randint(0,num_action)
    act= env.action_space.sample()
    #print("action",act)
    obs,reward,done,_=env.step(act)
    #bgrimg = cv2.cvtColor(obs, cv2.COLOR_YUH2BGR)
    state = Obs2State(obs)

    if reward>0:
        print("get reward",reward)
        break
    if done:
        print("done")
        env.reset()
    env.render()