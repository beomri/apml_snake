# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.qvalues = util.Counter()

        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            temp_values = util.Counter()
            temp_qvalues = util.Counter()
            for state in self.mdp.getStates():
                action_value = []
                for action in self.mdp.getPossibleActions(state):
                    pv_sum = 0
                    for next_state, next_prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        pv_sum += next_prob * (
                            self.discount * self.values[next_state] + self.mdp.getReward(state, action, next_state))
                        next_action_q = []
                        for next_action in self.mdp.getPossibleActions(next_state):
                            next_action_q.append(self.discount * self.qvalues[(next_state, next_action)])
                        max_action_qvalue = 0 if len(next_action_q) == 0 else max(next_action_q)
                        temp_qvalues[(state, action)] += next_prob * (
                            self.mdp.getReward(state, action, next_state) + max_action_qvalue)
                    action_value.append(pv_sum)
                max_action_value = 0 if len(action_value) == 0 else max(action_value)
                temp_values[state] = max_action_value
            self.values = temp_values
            self.qvalues = temp_qvalues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state, action)]

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            for next_state, next_prob in self.mdp.getTransitionStatesAndProbs(state, action):
                actions[action] += next_prob * self.values[next_state]
                # actions[action] += next_prob * (self.discount * self.values[next_state] + self.mdp.getReward(state, action, next_state))
        return actions.argMax()

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
