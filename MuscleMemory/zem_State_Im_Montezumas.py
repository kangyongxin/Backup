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
    # 找到所有小人的像素点的位置，然后求平均，好像不太合理
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
    plt.imshow(observation,cmap = plt.cm.gray)
    plt.show()

def Obs2State(obs):
    b,g,r =cv2.split(obs)
    ret, thresh1 = cv2.threshold(b, 210, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(b, 190, 255, cv2.THRESH_BINARY)
    thresh3 = thresh2 -thresh1
    """
    #作掩码，掩去上面表分部分，只提取小人位置，大约 0～30的高度
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
    state =[ int(cnt[0]/10),int(cnt[1]/10)]#这个粒度有点儿大

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

def main():
    n_trj=20
    trj = []
    a=[]
    e=[]
    r=[]
    oo=[]
    for i in range(21):
        for j in range(16):
            oo.append([i,j])
            
    n_state= len(oo)
    Im_s = np.zeros((n_state, 1))
    Im_p = np.zeros((n_state, 1))
    #Norm_Ims = np.zeros((n_state, 1))
    for eps in range(n_trj):
        #print(eps)
        env.reset()
        while True:
            act=random_policy()
            obs, reward, done, _ = env.step(act)
            state = Obs2State(obs) #编码状态
            #print(state)
            trj.append(state)
            env.render()
            if done:
                print("done")
                break
        [u_state, count_state] = stochastic_trj(trj)
        #print(u_state)
        #print(count_state)
        a.append(u_state)
        e.append(trj)
        r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        for i in range(len(oo)):  # 某个状态
            for w in range(len(a[eps])):  # 第w个 出现
                if a[eps][w] == oo[i]:
                    Im_s[i] = Im_s[i] + sum(r[eps][w:len(a[eps])])# 出现的顺序 越晚月重要？轨迹中
                    Im_p[i]=Im_p[i]+sum(r[eps][0:w])#状态出现的次数
    for i in range(len(oo)):
        #Im_s[i] = Im_s[i]/Im_p[i]
        Im_s[i] = Im_p[i]/Im_s[i]
        #if Im_s[i]<60:
         #    Im_s[i]=0
    plt.plot(Im_p)
    plt.show()
    np.savetxt("im_zem_sp2.txt", Im_s)

def read_state():
    env.reset()
    obs,_,_,_ = env.step(env.action_space.sample())

    oo = []
    for i in range(21):
        for j in range(16):
            oo.append([i, j])
    x = np.loadtxt("im_zem_sp2.txt")
    y = np.copy(x)
    Maxsort=sorted(x)
    index = np.argsort(-x)

    states = []
    for i in index[0:20]:
        #只显示前20个
        states.append(oo[i])
    show_state(states,obs)


if __name__ == "__main__":
    env = gym.make('MontezumaRevengeNoFrameskip-v4')
    main()
    read_state()