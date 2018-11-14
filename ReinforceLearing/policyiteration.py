import basic
import numpy as np


class PolicyIteration(basic.WithModel):
    def __init__(self, obstacle_reward=-100, final_reward=500, loop_time=10,
                 max_evaluate_time=100, theta=1.0, gamma=0.9, epsilon=0.5):
        super(PolicyIteration, self).__init__(obstacle_reward, final_reward, epsilon)

        self.gamma = gamma
        self.theta = theta
        # self.epsilon = epsilon
        self.loop_time = loop_time
        self.max_evaluate_time = max_evaluate_time
        self.init_map()
        self.is_stable = False
        # print(self.reward_map)

    def generate_action_map(self):
        for i in range(10):
            for j in range(15):
                if (self.map2array(i, j) == self.obstacles_series).any():
                    continue
                else:
                    policy_name = 'greedy'
                    self.action_map[i, j] = self.policy(np.array([i, j]), policy_name)

    def policy_evaluation(self, policy_name, iterate_time):
        # self.value_map = np.zeros(self.value_map.shape)
        time = 0
        while True:
            delta = 0
            v_map = np.zeros(self.value_map.shape)
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    is_obstacle = (self.map2array(i, j) == self.obstacles_series).any()
                    # is_final = ([i, j] == self.waiting_bird_position).all()
                    if is_obstacle:
                        continue
                    else:
                        probability = self.policy_probability(np.array([i, j]), policy_name)
                        for k in range(self.action.shape[0]):
                            x = i + self.action[k][0]
                            y = j + self.action[k][1]
                            reward = self.reward_map[x+1, y+1]
                            if self.is_normal_state([x, y]):
                                value = self.value_map[x, y]
                            else:
                                value = 0
                            v_map[i, j] += probability[k]*(reward+self.gamma*value)
                        delta = max([delta, abs(v_map[i, j]-self.value_map[i, j])])
            self.value_map = v_map
            time += 1
            print('delta', delta)
            if delta < self.theta or time > iterate_time:
                print('evaluate_time ', time)
                break

    def policy_iteration(self):
        loop_time = 0
        while True:
            self.policy_evaluation('epsilon_greedy', self.max_evaluate_time)
            self.value_map[self.waiting_bird_position[0], self.waiting_bird_position[1]] = self.final_reward
            action_map_ = self.action_map.copy()
            self.generate_action_map()
            # print("loop_time ", loop_time)
            loop_time += 1
            # print((action_map_ - self.action_map) == 0)
            if loop_time > self.loop_time or ((action_map_ - self.action_map) == 0).all():
                print('iterate_time ', loop_time)
                self.is_stable = True
                break

if __name__ == "__main__":
    agent = PolicyIteration(-50, 200, 100, 10, 0.01, 0.9, 0.5)
    agent.policy_iteration()
    # print(agent.Q)
    print('value_map')
    print(agent.value_map)
    # print(agent.action_map)
    if agent.is_stable:
        view = basic.ViewGame(agent.action_map)
        view.screen()
