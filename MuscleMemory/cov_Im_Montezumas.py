"""
value the state in Montezumas

"""
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def random_policy():
    act = env.action_space.sample()
    return act

def find_cnts(gray):
    a = np.array(gray>100,dtype = 'bool')
    c=np.nonzero(a)
    #print("c",c)
    cnt=[np.mean(c[0]),np.mean(c[1])]
    #print("cnt",cnt)
    return cnt

def state_to_obs(state):
    temp = np.zeros([210,160])
    temp [state[0]*10:(state[0]+1)*10,state[1]*10:(state[1]+1)*10] = np.ones([10,10])*255
    state_mask = temp.copy()
    return state_mask
def show_state(states,obs):
    state_mask=np.zeros([210,160])
    for state in states:
        state_mask=state_mask+state_to_obs(state)
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

    #show_state(state,obs)
    return state
def stochastic_trj(seq):
    #print("trj,", seq)
    u_state = []  # list of the unique state
    count_state = []  # list of the number of state
    k = 0
    for i in seq:
        if u_state.count(i) == 0:
            count_state.append(seq.count(i))
            u_state.append(i)
            k = k + 1

    return u_state, count_state
def cov_calculate(pics):
    print(len(pics))
    average = sum(pics)
    for i in range(210):
        for j in range(160):
            average[i][j]=average[i][j]/len(pics)
    w=np.zeros((210,160))
    for k in range(len(pics)):
        for i in range(210):
            for j in range(160):
                w[i][j]=w[i][j]+(pics[k][i][j]-average[i][j])*(pics[k][i][j]-average[i][j])

    #print(len(average[0,0,:]))#/len(pics)
    return average,w
def average_var(pics):
    num_pic= len(pics)
    [H,W]= pics[0].shape
    print(num_pic,H,W)

    sum_pics = sum(pics)

    average= sum_pics/num_pic
    # for i in range(210):
    #     for j in range(160):
    #
    #         if average[i][j] == sum_pics[i][j]/num_pic :
    #             print("right")
    #         else :
    #             print("not equal")
    delta2_pics = []
    for pic in pics:
        delta2_pics.append((pic-average)*(pic-average))
    sum_delta2 = sum(delta2_pics)
    var_pics = sum_delta2/num_pic
    # w=np.zeros((210,160))
    # for k in range(len(pics)):
    #     for i in range(210):
    #         for j in range(160):
    #             w[i][j]=w[i][j]+(pics[k][i][j]-average[i][j])*(pics[k][i][j]-average[i][j])
    #
    #print(len(average[0,0,:]))#/len(pics)
    return average,var_pics

def main():
    n_trj=1
    # trj = []
    # a=[]
    # e=[]
    # r=[]
    # oo=[]
    # for i in range(21):
    #     for j in range(16):
    #         oo.append([i,j])
    # n_state= len(oo)
    # Im_s = np.zeros((n_state, 1))
    #Norm_Ims = np.zeros((n_state, 1))
    pics=[]
    for eps in range(n_trj):
        print(eps)
        obs=env.reset()
        obs = rgb2gray(obs)
        obs = np.array(obs)
        #print(obs)
        pics.append(obs)

        while True:
            act=random_policy()
            obs, reward, done, _ = env.step(act)
            obs=rgb2gray(obs)
            pics.append(obs)
            #state = Obs2State(obs)
            #print(state)
            #trj.append(state)
            env.render()
            if done:
                print("done")
                break
        # [u_state, count_state] = stochastic_trj(trj)
        # print(u_state)
        # print(count_state)
        # a.append(u_state)
        # e.append(trj)
        # r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        # for i in range(len(oo)):  # 某个状态
        #     for w in range(len(a[eps])):  # 第w个 出现
        #         if a[eps][w] == oo[i]:
        #             Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中
    #aver,var=cov_calculate(pics)
    aver, var = average_var(pics)
    plt.imshow(var,cmap = plt.cm.gray)
    plt.show()

    # plt.plot(cov_calculate(pics))
    # plt.show()
    #np.savetxt("im1.txt", Im_s)
def read_state():
    # f = open('im.txt')
    # a = []
    # while True:
    #     line = f.readline()
    #     if not line:
    #         break
    #     a.append(line)
    # f.close()
    env.reset()
    obs,_,_,_ = env.step(env.action_space.sample())

    oo = []
    for i in range(21):
        for j in range(16):
            oo.append([i, j])
    x = np.loadtxt("im1.txt")
    y = np.copy(x)
    print("x",x)
    Maxsort=sorted(x)
    index = np.argsort(-x)
    print("x",x)
    print("Maxsort",Maxsort)
    print(index)
    #print("after sort:",Maxsort)
    states = []
    for i in index[0:20]:
        states.append(oo[i])

    show_state(states,obs)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

if __name__ == "__main__":
    env = gym.make('MontezumaRevengeNoFrameskip-v4')
    main()
    #read_state()