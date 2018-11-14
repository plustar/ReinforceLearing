# ReinforceLearning
Nankai ReinforceLearning homeworks
首先是一个基类Basic，其中储存的数据为地图的位置信息及不同位置的回报等信息。
类ViewGame继承Basic，根据Basic中的地图信息绘制窗口，ViewGame根据Basic中的action_map可视化移动小鸟，对于model中的不同的value表，
action_map也会随之更新。

 
现在已经学习的强化学习方法包括有模型与无模型两种，两者之间的根本性差别为计算所需的value表不同，以及针对不同的表，计算概率的函数有所
区别（虽然原理一样，但计算方法不一样）。针对这两个不同点，类WithModel和WithoutModel分别独立继承Basic类，分别储存价值表和行为价值
表以及对应的策略。策略函数分为两个函数，policy与policy_probability，因为在写程序的过程中发现对于相同的策略，有的需要当前状态执行动
作的概率，有的需要当前状态直接根据概率获取最优动作。
	对于策略迭代和值迭代，判断是否收敛的原则同书上所写，当value表中的最大变化值小于0.01时认为已收敛。策略迭代的迭代次数为5次，评估时
间不等；值迭代为47迭代后即收敛。

obstale_reward=-50 ,final_reward=200,
  策略迭代迭代次数5次（已收敛），最大评估次数100（实际运行时最大也就十几），参数theta=0.01, gamma=0.9, epsilon=0.5；（theta为value表稳定的阈值）；
	值迭代迭代次数47次（已收敛），参数theta=0.5, gamma=0.9, epsilon=0.5；
	对于无模型的方法，没有设定theta判定收敛，而是直接测固定迭代次数。Monte Carlo的方法均使用随机初始化，程序就不细说了。程序最后显示
的表为action_map，而不是value表。
	
obstale_reward=-100 ,final_reward=1000
	MonteCarloExploreStart迭代迭代次数10000次，gamma=0.9, epsilon=0.5；
	MCOffPolicy迭代迭代次数100000次，gamma=0.9, epsilon=0.5；（这个策略表现并不如其他两个）
	MCOnPolicy迭代迭代次数10000次，gamma=0.9, epsilon=0.5；

完成这次作业的过程中，由于使用了修改后的类框架，写程序的速度和调bug的速度比往常有明显的提升。
obstale_reward=-100 ,final_reward=500,
	Sarsa迭代次数50000次，参数alpha=0.5, gamma=0.9, epsilon=0.5；
	ExpectedSarsa迭代次数500次，参数alpha=0.5, gamma=0.9, epsilon=0.5；
	QLearning迭代次数2500次，参数alpha=0.5, gamma=0.9, epsilon=0.5；
	DoubleQLearning迭代次数4000次，参数alpha=0.5, gamma=0.9, epsilon=0.5；
	相同条件下，如果不考虑是我程序出错的问题，表现结果为
   ExpectedSarsa>QLearning>DoubleQLearning>Sarsa
