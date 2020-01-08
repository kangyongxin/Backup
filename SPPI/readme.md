readme file for  sppi in ijcai 2020

these notes are listed by data;
## 20191231
thermodynamic diagram

\# python -m baselines.SPPI.miniMaze.hotpic

## 20191114

To Do:
create a framework for SPPI: this framework will hold three part, 
+ the first one is for ablation experiments in minimaze
+ the second one is for tabular case
+ the third is for continue case

1.test for each env:


失败案例，只能在根目录下运行，否则要改动很多路径：\# python -m baselines.SPPI.minigrid.run_tests

\# python -m baselines.SPPI.miniMaze.run_tests

\# python -m baselines.SPPI.GridMaze.run_tests

\# python -m baselines.SPPI.MR.run_tests



2. q learning for tabular case:

先在每个文件夹中写一个可以用的，然后再把所有的汇总在一起部署实验 

miniMaze 中没有终点，给他个终点，试着让它跑起来

\# python -m baselines.SPPI.miniMaze.run_qlearning

这里用的是qlearning table , 要设计对应的 连续模块 


GridMaze 


MR


Atari

3.ac for tablular case:

miniMaze

Grid

MR

Atari


4.dqn for tablular case:



5.a2c


6.SP q-learning


7.SP ac


8.SP DQN


9.SP A2C


10 SIL


11 SIL +SP


12 SIL + Count based





