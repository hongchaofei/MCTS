import numpy as np
from random import choice
from typing import List, Dict

class Node():

    def __init__(self, state=None, parent=None):
        self.visits = 0.0
        self.reward = 0.0
        self.state = state
        self.children: Dict[Node] = {}

    def add_child(self, action, child_node):
        if action in self.children:
            assert self.children[action] is child_node
        else:
            self.children[action] = child_node

    def set_state(self, state):
        self.state = state

    def update(self, reward):
        self.reward += reward
        self.visits += 1


    @property
    def ave_reward(self):
        return self.reward/(self.visits+1.0e-10)


    def ucb_score(self, c_explore, simulalte_times):
        exploit = self.reward/(self.visits+1.0e-10)
        explore = np.sqrt(np.log(simulalte_times)/(self.visits+1.0))
        score = exploit + c_explore*explore
        return score


    def fully_expanded(self, num_action):
        if len(self.children) == num_action:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.state[0] == other.state[0] and self.state[1] == other.state[1]:
            return True
        else:
            return False

    def __repr__(self):
        s = "Node[ state:%d; visits:%d; reward:%f]" % (self.state.value, self.visits, self.ave_reward)
        return s



class MCTS():

    def __init__(self):
        self.nodes = []
        self.actions = set()
        self.num_action = 0
        self.c_explore = 1.0
        self.simulate_times = 0.0

    def set_actions(self, actions):
        self.actions = set(actions)

    def start_node(self, state):
        for node in self.nodes:
            if node.state == state:
                return node
        new_node = Node(state=state)
        self.nodes.append(new_node)
        return new_node



    def search(self, search_path: List):
        _, node = search_path[-1]
        path_nodes = []
        for p in search_path:
            path_nodes.append(p[1])
        if node.fully_expanded(self.num_action):
            best_score = -10000000000000000000000000000
            best_path = None
            for action, child_node in node.children.items():
                if child_node not in path_nodes:
                    tmp_path = search_path + [(action, child_node)]
                    score, tmp_path = self.search(tmp_path)
                    if score > best_score:
                        best_score = score
                        best_path = tmp_path

            if best_path is None:
                best_score = node.ucb_score(self.c_explore, self.simulate_times) - 10
                best_path = search_path

        else:
            best_score = node.ucb_score(self.c_explore, self.simulate_times)
            best_path = search_path
        return best_score, best_path

    def select_action(self, node):

        if not node.fully_expanded(self.num_action):
            expanded_actions = node.children.keys()
            new_actions = self.actions.difference(expanded_actions)
            selected_action = choice(tuple(new_actions))
        else:
            selected_action = choice(tuple(self.actions))
        return selected_action


    def expand(self, parent_node, selected_action, state):
        child_node = Node(state)
        new_state = True
        for node in self.nodes:
            if node.state == state:
                child_node = node
                new_state = False
                break
        if new_state:
            self.nodes.append(child_node)

        parent_node.add_child(selected_action, child_node)
        return child_node


    def backup(self, path, reward):
        for action, node in path:
            node.update(reward)



