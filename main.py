from mcts import Node, MCTS
from maze import Maze
import numpy as np
# import hashlib


class State():
    NUM_TURNS = 5
    GOAL = 0
    MOVES = [2, -1, 3, -4]
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=[], turn=NUM_TURNS):
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self, action):
        nextmove = self.turn*action
        # nextmove = random.choice([x * self.turn for x in self.MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove], self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):
        r = 1.0 - (abs(self.value - self.GOAL) / self.MAX_VALUE)
        return r

    # def __hash__(self):
    #     return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if self.turn == other.turn and self.value == other.value:
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


if __name__=="__main__":
    episode_num = 10000
    agent = MCTS()
    agent.num_action = State.num_moves
    agent.set_actions(State.MOVES)

    for ep in range(episode_num):
        agent.simulate_times += 1
        state = State(value=np.random.randint(10)-5, turn=2 + np.random.randint(14))

        start_node = agent.start_node(state)
        score, path = agent.search([('start', start_node)])

        #search according to mcts path
        for action, node in path:
            if action is not 'start':
                state = state.next_state(action)
                assert state == node.state

        # simulate and expand
        while not state.terminal():
            action, node = path[-1]
            selected_action = agent.select_action(node)
            state = state.next_state(selected_action)
            new_node = agent.expand(node, selected_action, state)
            path.append((selected_action, new_node))

        reward = state.reward()
        agent.backup(path, reward)
        print("episode: %d, c_explore:%f, reward: %f, value:%d, node_len:%d"%(ep, agent.c_explore, reward, state.value, len(agent.nodes)))
        actions = []
        for action, node in path:
            if action == 'start':
                actions.append("start value:%d"%node.state.value)
            else:
                actions.append(action)
        actions.append("end value:%d"%path[-1][1].state.value)
        print(actions)
        agent.c_explore *= 0.999


# for node in agent.nodes:
#     print(node.state, node.children)












