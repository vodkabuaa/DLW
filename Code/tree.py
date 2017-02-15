import numpy as np

class TreeModel(object):
    def __init__(self, decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0):
        """
        Here we only consider the decision nodes and periods.
        Since the last period is not uncertain, it will not a
        complete num_decision_nodes
        """
        self.decision_times = np.array(decision_times)
        #self.information_times = self.decision_times[:-2]
        self.prob_scale = prob_scale

        self.num_periods = len(decision_times) - 1
        self.num_decision_nodes = 2**self.num_periods - 1
        self.num_final_states = 2**(self.num_periods - 1)

        self.damage_by_state = np.zeros(self.num_decision_nodes)
        self.cost_by_state = np.zeros(self.num_decision_nodes)
        self.grad = np.zeros(self.num_decision_nodes)

        ### nodes probability
        self.final_states_prob = np.zeros(self.num_final_states)
        self.node_prob = np.zeros(self.num_decision_nodes)

        #### emissions
        self.emissions_per_period = np.zeros(self.num_periods)
        self.emissions_to_ghg = np.zeros(self.num_periods)

        ### Initialize the probability
        self._create_probs()
        

    def _create_probs(self):
        """Creates the probabilities of every nodes in the tree structure.

        """
        self.final_states_prob[0] = 1.0
        sum_probs = 1.0
        next_prob = 1.0

        ##Calculate the probability for the final state
        for n in range(1, self.num_final_states):
            next_prob = next_prob * self.prob_scale**(1.0 / n)
            self.final_states_prob[n] = next_prob
        self.final_states_prob /= np.sum(self.final_states_prob)

        self.node_prob[self.num_final_states-1:] = self.final_states_prob
        for period in range(self.num_periods-2, -1, -1): 
            for state in range(0, 2**period):
                pos = self.get_node(period, state)
                self.node_prob[pos] = self.node_prob[2 * pos + 1] + self.node_prob[2 * pos + 2]

    def get_node(self, period, state):
        """We can use the relationship between the period, state and index of
        these ndarrays to get the node number by O1.

        """
        if state >= 2**period:
            raise IndexError
        return 2**period + state - 1

    def get_state(self, node, period=None):
        if not period:
            period = self.get_period(node)
        return node - (2**period - 1)

    def get_period(self, node):
        if node >= self.num_decision_nodes: # can still be a too large node-number
            return self.num_periods

        for i in range(0, self.num_periods):
            if int((node+1) / 2**i ) == 1:
                return i

    def get_parent_node(self, child):
        if child == 0:
            return 0
        if child > self.num_decision_nodes:
            return child - self.num_final_states
        if child % 2 == 0:
            return int((child - 2) / 2)
        else:
            return int((child - 1 ) / 2)

    def get_path(self, node, period=None):
        if period is None:
            period = self.tree.get_period(node)
        path = [node]
        for i in range(0, period):
            parent = self.get_parent_node(path[i])
            path.append(parent)
        path.reverse()
        return path
    
    def reachable_end_states(self, node, period=None, state=None):
        if period is None:
            period = self.get_period(node)
        if state is None:
            state = self.get_state(node, period)

        k = self.num_final_states / 2**period
        return (k*state, k*(state+1)-1)

  

