import basic
import numpy as np


class MonteCarloExploreStart(basic.WithoutModel):
    def __init__(self, obstacle_reward=-100, final_reward=500, loop_time=100000, epsilon=0.5, gamma=0.9):
        super(MonteCarloExploreStart, self).__init__(obstacle_reward, final_reward, loop_time, epsilon)

        self.gamma = gamma
        self.Q_sum = np.zeros(self.Q.shape)
        self.Q_time = np.full(self.Q.shape, 1e-100)
        self.init_map()
        print(self.reward_map)

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
                if loop_time < 2000:
                    while True:
                        action = self.policy(state, 'random')

                        state += self.action[int(action)]
                        mc_chain = np.append(mc_chain, state.reshape(1, 2), axis=0)
                        action_list.append(action)
                        mini_loop += 1
                        if self.is_normal_state(state):
                            continue
                        else:
                            break
                else:
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
                gt = 0
                for i in range(len(action_list)):
                    j = len(action_list) - 1 - i
                    gt = self.reward_map[mc_chain[j + 1][0] + 1, mc_chain[j + 1][1] + 1] + self.gamma * gt
                    self.Q_time[mc_chain[j][0], mc_chain[j][1], action_list[j]] += 1
                    self.Q_sum[mc_chain[j][0], mc_chain[j][1], action_list[j]] += gt
                self.Q = self.Q_sum / self.Q_time

            print("loop_time", loop_time)
            if loop_time > self.loop_time:
                break

if __name__ == "__main__":
    agent = MonteCarloExploreStart(-100, 1000, 10000, 0.5, 0.9)
    agent.loop()
    agent.generate_action_map()
    # print(agent.Q)
    print(agent.action_map)

    view = basic.ViewGame(agent.action_map)
    view.screen()
