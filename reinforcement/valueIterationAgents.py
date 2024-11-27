# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from collections import defaultdict

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations): #k
            states = self.mdp.getStates()
            grid = util.Counter() # keep track of each state's value update
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                # Perform Function V_{k+1}(s)
                actionResults = defaultdict()
                for action in self.mdp.getPossibleActions(state):
                    currActionResult = 0
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state,action):
                        reward = self.mdp.getReward(state, action, nextState)
                        lastNewStateValue = self.discount*self.values[nextState]
                        newValue = probability * (reward + lastNewStateValue)
                        currActionResult += newValue
                    actionResults[action] = currActionResult
                _,v = max(actionResults.items(), key=lambda x: x[1])
                grid[state] = v # update values at k+1 iteration to the grid
            self.values = grid


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q*(s,a) = sum_{s'} T(s,a,s') [ R(s,a,s') + gamma max_{a'}Q*(s'a') ]
        q_value = 0
        for s_prime, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob * (self.mdp.getReward(state, action, s_prime) + self.discount * self.values[s_prime])
        return q_value
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # pi*(s) = argmax_a sum_s'( T(s,a,s') [ R(s,a,s') + gamma V*(s') ] ) = argmax_a Q(s,a)
        actions = self.mdp.getPossibleActions(state)
        bestAction = None
        bestQ = float('-inf')
        for action in actions:
          q = self.computeQValueFromValues(state, action)
          if bestQ < q:
            bestQ = q
            bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
