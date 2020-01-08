import gym


env=gym.make('MontezumaRevengeNoFrameskip-v4')

env.reset()
for i in range(1000000):
    #a= random.randint(0,num_action)
    act= env.action_space.sample()
    #print("action",act)
    obs,reward,done,_=env.step(act)
    #bgrimg = cv2.cvtColor(obs, cv2.COLOR_YUH2BGR)


    if reward>0:
        print("get reward",reward)
        break
    if done:
        print("done")
        env.reset()
    env.render()