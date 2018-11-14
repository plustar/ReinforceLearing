import basic
import numpy as np


class DoubleQLearning(basic.WithoutModel):
    def __init__(self, obstacle_reward=-100, final_reward=500, loop_time=100000, alpha=0.5, gamma=0.9, epsilon=0.5):
        super(DoubleQLearning, self).__init__(obstacle_reward, final_reward, loop_time, epsilon)

        self.alpha = alpha
        self.gamma = gamma
        # self.epsilon = epsilon
        self.loop_time = loop_time
        self.Q1 = self.Q
        self.Q2 = self.Q
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
            # index = np.random.randint(self.map_size[0]*self.map_size[1])
            # if (index == self.obstacles_series).any():
            if False:
                continue
            else:
                loop_time += 1
                # state = np.array(self.array2map(index))
                state = np.array([0, 0])

                mini_loop = 0
                while True:
                    action = self.policy(state, 'epsilon_greedy')
                    state_ = state + self.action[int(action)]
                    if not self.is_obstacle(state_) and mini_loop < 50:
                        mini_loop += 1
                        # print(state_)
                        reward = self.reward_map[state_[0] + 1, state_[1] + 1]
                        if np.random.rand() < 0.5:
                            action_ = self.policy(state_, 'greedyQ1')
                            tmp = self.gamma * self.Q2[state_[0], state_[1], action_] - self.Q1[state[0], state[1], action]
                            # print(tmp, reward)
                            self.Q1[state[0], state[1], action] += self.alpha * (reward + tmp)
                            state = state_
                        else:
                            action_ = self.policy(state_, 'greedyQ2')
                            tmp = self.gamma * self.Q1[state_[0], state_[1], action_] - self.Q2[
                                state[0], state[1], action]
                            # print(tmp, reward)
                            self.Q2[state[0], state[1], action] += self.alpha * (reward + tmp)
                            state = state_
                    else:
                        reward = self.reward_map[state_[0] + 1, state_[1] + 1]
                        if np.random.rand() < 0.5:
                            self.Q1[state[0], state[1], action] += self.alpha * (reward - self.Q2[state[0], state[1], action])
                        else:
                            self.Q2[state[0], state[1], action] += self.alpha * (reward - self.Q1[state[0], state[1], action])
                        break

                    # print(mini_loop)

            print("loop_time", loop_time)
            if loop_time > self.loop_time:

                break

if __name__ == "__main__":
    agent = DoubleQLearning(-100, 500, 4000, 0.5, 0.9, 0.5)
    agent.loop()
    agent.generate_action_map()
    # print(agent.Q)
    print(agent.action_map)

    view = basic.ViewGame(agent.action_map)
    view.screen()
