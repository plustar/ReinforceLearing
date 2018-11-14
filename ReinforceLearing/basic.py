import numpy as np
import pygame
import os
from pygame.locals import *
import time


class Basic:
    def __init__(self, obstacle_reward=-100, final_reward=500):
        self.map_size = np.array([10, 15])
        # whether the state is obstacle, maybe collide with obstacles series
        # self.map = np.zeros(self.map_size + 2)
        # get the reward of state, reduce compare time
        self.reward_map = -np.ones(self.map_size + 2, dtype=int)
        # actions down right left up
        self.action = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
        # used for showing final bird trajectory
        self.action_map = -np.ones(self.map_size)
        # set the obstacle reward
        self.obstacle_reward = obstacle_reward
        # set the final reward
        self.final_reward = final_reward
        # the obstacles in grid map, for plot window
        self.obstacles = np.array([[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                                   [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14],
                                   [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
                                   [6, 10], [6, 11], [6, 12], [6, 13], [6, 14]])
        # obstacles series mode, for easily accessible character
        self.obstacles_series = np.zeros(self.obstacles.shape[0])
        # set numpy print precision
        self.precision = np.set_printoptions(precision=0, linewidth=100, suppress=True)
        # if random ...
        self.seed = np.random.seed(0)
        self.flying_bird_position = np.array([0, 0])
        self.waiting_bird_position = np.array([9, 0])
        self.init_map()

    @staticmethod
    def map2array(i, j):
        return 10*j+i

    @staticmethod
    def array2map(args):
        # print(args, np.array([int(args % 10), int(args // 10)]))
        return np.array([int(args % 10), int(args // 10)])

    # init the map character, include reward and obstacle settings
    def init_map(self):
        i = 0
        for obstacle in self.obstacles:
            self.reward_map[obstacle[0] + 1, obstacle[1] + 1] = self.obstacle_reward
            # self.map[obstacle[0] + 1, obstacle[1] + 1] = 1
            self.obstacles_series[i] = self.map2array(obstacle[0], obstacle[1])
            i += 1
        self.reward_map[self.waiting_bird_position[0] + 1, self.waiting_bird_position[1] + 1] = self.final_reward
        self.reward_map[0, :] = self.obstacle_reward
        self.reward_map[:, 0] = self.obstacle_reward
        self.reward_map[-1, :] = self.obstacle_reward
        self.reward_map[:, -1] = self.obstacle_reward
        # print(self.reward_map)
        # print(self.obstacles_series)

    # def policy(self, state):
        # return self.action[0]

    def is_normal_state(self, state):
        # print(state[0]+1, state[1]+1, self.reward_map[state[0]+1, state[1]+1])
        if self.reward_map[state[0]+1, state[1]+1] != -1:
            return False
        else:
            return True

    def is_obstacle(self, state):
        # print(state[0]+1, state[1]+1, self.reward_map[state[0]+1, state[1]+1])
        if self.reward_map[state[0]+1, state[1]+1] == self.obstacle_reward:
            return True
        else:
            return False


class WithModel(Basic):
    def __init__(self, obstacle_reward, final_reward, epsilon):
        super(WithModel, self).__init__(obstacle_reward=obstacle_reward, final_reward=final_reward)
        self.value_map = np.ones(self.map_size)
        self.epsilon = epsilon

    def policy_probability(self, state, policy_name):
        if policy_name == 'greedy':
            probability = np.zeros(4)
            v = np.zeros(4)
            for i in range(4):
                x = state[0] + self.action[i][0]
                y = state[1] + self.action[i][1]
                if self.is_obstacle([x, y]):
                    v[i] = self.value_map[x, y]
                else:
                    v[i] = -1e1000
            max_ = np.where(v == v.max())[0]
            probability[max_] = 1
            return probability
        elif policy_name == 'epsilon_greedy':
            probability = np.full(4, self.epsilon / 4)
            v = np.zeros(4)
            for i in range(4):
                x = state[0] + self.action[i][0]
                y = state[1] + self.action[i][1]
                if self.is_normal_state([x, y]):
                    v[i] = self.value_map[x, y]
                else:
                    v[i] = -1e1000
            max_ = np.where(v == v.max())[0]
            probability[max_] = 1 - self.epsilon + self.epsilon / 4
            return probability

    def policy(self, state, policy_name):
        if policy_name == 'greedy':
            v = np.zeros(4)
            for i in range(4):
                x = state[0] + self.action[i][0]
                y = state[1] + self.action[i][1]
                if not self.is_obstacle([x, y]):
                    v[i] = self.value_map[x, y]
                else:
                    v[i] = -1e10000
            max_ = np.where(v == v.max())[0]
            return max_[0]


class WithoutModel(Basic):
    def __init__(self, obstacle_reward, final_reward, loop_time, epsilon):
        super(WithoutModel, self).__init__(obstacle_reward, final_reward)
        self.Q = np.zeros((self.map_size[0], self.map_size[1], 4))
        self.loop_time = loop_time
        self.epsilon = epsilon

    def policy(self, state, policy_name):
        if policy_name == 'epsilon_greedy':
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q[x, y, i]
            max_ = np.where(v == v.max())[0]
            # print(v)
            if np.random.rand() > self.epsilon:
                # print(max_[0])
                # print(max_)
                return max_[0]
            else:
                return np.random.randint(len(v))
        elif policy_name == 'greedy':
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q[x, y, i]
            max_ = np.where(v == v.max())[0]
            return max_[0]
            # p = np.random.randint(len(max_))
            # return max_[p]
        elif policy_name == 'soft_max':
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q[x, y, i]
            flat_value = np.exp(v) / np.exp(v).sum()
            return np.random.choice([0, 1, 2, 3], p=flat_value.ravel())
        elif policy_name == 'random':
            p = np.random.randint(4)
            return p
        elif policy_name == 'greedyQ1':
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q1[x, y, i]
            max_ = np.where(v == v.max())[0]
            return max_[0]
        elif policy_name == 'greedyQ2':
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q2[x, y, i]
            max_ = np.where(v == v.max())[0]
            return max_[0]

    def policy_probability(self, state, policy_name):
        if policy_name == 'greedy':
            probability = np.zeros(4)
            v = np.zeros(4)
            x = state[0]
            y = state[1]
            for i in range(4):
                if self.is_obstacle([x, y]):
                    v[i] = self.Q[x, y, i]
                else:
                    v[i] = -1e1000
            max_ = np.where(v == v.max())[0]
            probability[max_] = 1
            return probability
        if policy_name == 'epsilon_greedy':
            v = np.zeros(4)
            probability = np.full(4, self.epsilon/4)
            x = state[0]
            y = state[1]
            for i in range(4):
                v[i] = self.Q[x, y, i]
            max_ = np.where(v == v.max())[0]
            probability[max_] = 1-self.epsilon+self.epsilon / 4
            return probability


class ViewGame(Basic):
    def __init__(self, action_map):
        super(ViewGame, self).__init__()
        self.action_map = action_map
        self.viewer = None
        self.viewer_resolution = (400, 300)
        self.viewer_flag = 0
        self.viewer_depth = 32
        self.value_state_changed = False
        self.last_action = -5
        self.current_dir = os.path.split(os.path.realpath(__file__))[0]
        self.bird_file = self.current_dir + "/bird.png"
        self.obstacle_file = self.current_dir + "/obstacle.png"
        self.background_file = self.current_dir + "/background.png"
        self.grid_resolution = np.array([40, 20])
        self.flying_bird = None
        self.waiting_bird = None
        self.obstacle = None
        self.background = None

        # print(3)

    def grid_to_window(self, a, b):
        return a * self.grid_resolution[0], b * self.grid_resolution[1]

    def draw_flying_bird(self):
        self.viewer.blit(self.flying_bird,
                         self.grid_to_window(self.flying_bird_position[0],
                                             self.flying_bird_position[1]))

    def draw_background(self):
        self.viewer.blit(self.background, (0, 0))
        for obstacle in self.obstacles:
            self.viewer.blit(self.obstacle, self.grid_to_window(obstacle[0], obstacle[1]))

    def update_state(self):
        x = self.flying_bird_position[0]
        y = self.flying_bird_position[1]
        action = self.action_map[x, y]
        self.flying_bird_position += self.action[int(action)]

    def screen(self):

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.viewer_resolution,
                                                  self.viewer_flag, self.viewer_depth)
            self.flying_bird = pygame.image.load(self.bird_file).convert_alpha()
            self.waiting_bird = pygame.image.load(self.bird_file).convert_alpha()
            self.obstacle = pygame.image.load(self.obstacle_file).convert_alpha()
            self.background = pygame.image.load(self.background_file).convert_alpha()
            pygame.display.set_caption("Flying Bird")
            self.draw_background()
            self.viewer.blit(self.waiting_bird,
                             self.grid_to_window(self.waiting_bird_position[0],
                                                 self.waiting_bird_position[1]))

            while True:
                pygame.init()
                # self.viewer = pygame.display.set_mode(self.viewer_resolution,
                #                                      self.viewer_flag, self.viewer_depth)
                self.draw_background()
                self.viewer.blit(self.waiting_bird,
                                 self.grid_to_window(self.waiting_bird_position[0],
                                                     self.waiting_bird_position[1]))
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        os.sys.exit()
                keys = pygame.key.get_pressed()
                if keys[K_ESCAPE]:
                    # value_policy_pic = "value_policy.png"
                    # pygame.image.save(self.viewer, value_policy_pic)
                    os.sys.exit()

                is_normal = self.is_normal_state(self.flying_bird_position)
                is_not_final = not (self.flying_bird_position == self.waiting_bird_position).all()
                if is_not_final and is_normal:
                    self.draw_flying_bird()
                    self.update_state()
                else:
                    # value_policy_pic = "value_policy.png"
                    # pygame.image.save(self.viewer, value_policy_pic)
                    break

                pygame.display.update()
                time.sleep(0.2)
