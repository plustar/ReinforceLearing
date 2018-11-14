import basic
import numpy as np


class MCOffPolicy(basic.WithoutModel):
    def __init__(self, obstacle_reward=-100, final_reward=500, loop_time=100000, epsilon=0.5, gamma=0.9):
        super(MCOffPolicy, self).__init__(obstacle_reward, final_reward, loop_time, epsilon)

        # self.epsilon = epsilon
        self.gamma = gamma
        self.init_map()
        self.loop_time = loop_time
        self.C = np.zeros(self.Q.shape)
        # print(self.reward_map)

    def generate_action_map(self):
        for i in range(10):
            for j in range(15):
                if (self.map2array(i, j) == self.obstacles_series).any():
                    continue
                else:
                    self.action_map[i, j] = self.policy([i, j], 'greedy')

    def loop(self):
        loop_time = 0
        while True:
            index = np.random.randint(self.map_size[0]*self.map_size[1])
            if (index == self.obstacles_series).any():
                # if False:
                continue
            else:
                loop_time += 1
                state = np.array(self.array2map(index))
                # state = np.array([0, 0])
                mc_chain = np.array([state])
                action_list = []
                # reward_list = []
                mini_loop = 0
                while True:
                    action = self.policy(state, 'epsilon_greedy')

                    state += self.action[int(action)]
                    mc_chain = np.append(mc_chain, state.reshape(1, 2), axis=0)
                    action_list.append(action)
                    mini_loop += 1
                    if self.is_normal_state(state) and mini_loop < 100:
                        continue
                    else:
                        break
                g = 0
                w = 1
                for i in range(len(action_list)):
                    j = len(action_list) - 1 - i
                    g = self.reward_map[mc_chain[j+1][0] + 1, mc_chain[j+1][1] + 1]+self.gamma*g
                    self.C[mc_chain[j][0], mc_chain[j][1], action_list[j]] += w
                    tmp = w*(g - self.Q[mc_chain[j][0], mc_chain[j][1], action_list[j]])
                    c = self.C[mc_chain[j][0], mc_chain[j][1], action_list[j]]
                    self.Q[mc_chain[j][0], mc_chain[j][1], action_list[j]] += tmp/c
                    if action_list[j] != self.policy(mc_chain[j], 'greedy'):
                        break
                    else:
                        if action_list[j] == self.policy(mc_chain[j], 'epsilon_greedy'):
                            b = 1-self.epsilon+self.epsilon/4
                        else:
                            b = self.epsilon/4
                        w /= b

            print("loop_time", loop_time)
            if loop_time > self.loop_time:

                break

if __name__ == "__main__":
    # agent = MCOffPolicy(-100, 500, 200000, 0.5, 0.9) # ok arg_max without random
    agent = MCOffPolicy(-100, 1000, 100000, 0.5, 0.9)  # not ok arg_max without random
    # agent = MCOffPolicy(-100, 1000, 200000, 0.5, 0.9) # ok arg_max with random
    # agent = MCOffPolicy(-100, 1000, 500000, 0.5, 0.9) # no ok if iterate time is more than 250000
    agent.loop()
    agent.generate_action_map()
    # print(agent.Q)
    print(agent.action_map)

    view = basic.ViewGame(agent.action_map)
    view.screen()
