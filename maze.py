# -*- coding: utf-8 -*-
"""
Created on 2022/4/11
@project: MCTS
@filename: maze
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""


import numpy as np
from mcts import Node, MCTS
from random import choice


class Maze():

    MapSize = 10
    Map = np.zeros((MapSize, MapSize))
    Map[1, 0:2] = 1
    Map[1:6, 4] = 1
    Map[0:9, 6] = 1
    Map[5:10, 2] = 1
    Map[3, 2:4] = 1
    Map[7, 3:5] = 1
    Map[6:10, 8] = 1
    Map[4, 8:] = 1
    Map[2, 6:8] = 1
    MOVES = ['W', 'E', 'N', 'S']
    num_moves = len(MOVES)
    MAX_TURNS = 100000
    # print(Map)


    def __init__(self, start, goal):
        super(Maze, self).__init__()
        assert start[0] < Maze.MapSize and start[1] < Maze.MapSize
        assert Maze.Map[start[0], start[1]] == 0
        assert goal[0] < Maze.MapSize and goal[1] < Maze.MapSize
        assert Maze.Map[goal[0], goal[1]] == 0
        self.turn = 0
        self.start = start
        self.goal = goal
        self.position = start
        self._reward = 0

    def next_state(self, action):
        tmp_posi = [self.position[0], self.position[1]]
        if action == 'W':
            tmp_posi[1] -= 1
        elif action == 'E':
            tmp_posi[1] += 1
        elif action == 'N':
            tmp_posi[0] -= 1
        elif action == 'S':
            tmp_posi[0] += 1
        if np.min(tmp_posi) < 0 or np.max(tmp_posi) >= Maze.MapSize:
            self._reward -= 2
        elif Maze.Map[tmp_posi[0], tmp_posi[1]] == 1:
            self._reward -= 2
        else:
            self.position = tmp_posi
            if self.position[0] == self.goal[0] and self.position[1] == self.goal[1]:
                self._reward += 100
            else:
                self._reward -= 1
        self.turn += 1
        return self.position

    @property
    def reward(self):
        return self._reward/100.0



    def terminal(self):
        if self.turn >= Maze.MAX_TURNS:
            return True
        elif self.position[0] == self.goal[0] and self.position[1] == self.goal[1]:
            return True
        else:
            return False


    def __repr__(self):
        s = "Posi: %d,%d; Moves: %s" % (self.position[0],self.position[1], self.turn)
        return s

    @staticmethod
    def get_vaild_positions():
        positions = []
        for ii in range(Maze.MapSize):
            for jj in range(Maze.MapSize):
                if Maze.Map[ii, jj] == 0:
                    positions.append([ii,jj])

        return positions





if __name__=="__main__":
    episode_num = 1000
    agent = MCTS()
    agent.num_action = Maze.num_moves
    agent.set_actions(Maze.MOVES)
    mean_reward = 0.0
    goal = [9,4]
    for ep in range(episode_num):
        agent.simulate_times += 1
        positions = Maze.get_vaild_positions()
        start = choice(positions)

        state = Maze(start=start, goal=goal)

        start_node = agent.start_node(state.position)
        score, path = agent.search([('start', start_node)])

        #search according to mcts path
        for action, node in path:
            if action != 'start':
                position = state.next_state(action)
                assert position == node.state
        # if not state.terminal():
        #     num_a = 0
        #     for n in agent.nodes:
        #         num_a += len(n.children)
        #     print('=================%d=============='%num_a)

        # simulate and expand
        while not state.terminal():
            action, node = path[-1]
            selected_action = agent.select_action(node)
            position = state.next_state(selected_action)
            new_node = agent.expand(node, selected_action, position)
            path.append((selected_action, new_node))

        # if len(agent.nodes) > 40:
        #     a = 1

        reward = state.reward
        mean_reward = 0.99*mean_reward + 0.01*reward
        agent.backup(path, reward)
        print("episode: %d, c_explore:%f, reward: %f[%f], steps:%d, node_len:%d, reached:%d"%(ep, agent.c_explore, reward, mean_reward, state.turn, len(agent.nodes), state.position[0]==state.goal[0] and state.position[1]==state.goal[1]))
        # actions = []
        # for action, node in path:
        #     if action == 'start':
        #         actions.append("start posi:%d,%d"%(node.state[0], node.state[1]))
        #     else:
        #         actions.append(node.state)
        # actions.append("goal posi:%d,%d"%(state.goal[0], state.goal[1]))
        # print(actions)
        agent.c_explore *= 0.999


