# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # the closer the next food is, the higher the score is
        closestFoodDist = float('inf')
        if len(newFood.asList()) != 0:
            closestFoodDist = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        else:
            closestFoodDist = 0
        foodScore = 1/(closestFoodDist+1)
        
        # the closer the next ghost is, the smaller the score is
        ghostScore = 0
        newGhostDist = min([manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])
        # if ghost is going to eat the pacman, set it to -inf to avoid it
        if newGhostDist < 1:
            ghostScore = float('-inf')

        return foodScore + ghostScore + successorGameState.getScore() 
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, next_action = self.minimax(gameState, self.index, 0)
        return next_action

    def minimax(self, gameState, agent, depth):
        num_agents = gameState.getNumAgents()
        if (depth == self.depth and agent % num_agents == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        if agent % num_agents == 0: # Pacman - maximize
            return self.maximize(gameState, agent % num_agents, depth)
        return self.minimize(gameState, agent % num_agents, depth) # Ghosts - minimize
    
    def minimize(self, state, agent, depth):
        best_value = float('inf')
        best_action = Directions.STOP
        for action in state.getLegalActions(agent):
            successor_state = state.generateSuccessor(agent, action)
            next_value, _ = self.minimax(successor_state, agent + 1, depth) # same depth, different ghosts
            if next_value < best_value:
                best_value, best_action = next_value, action
        return best_value, best_action

    def maximize(self, gameState, agent, depth):
        best_value = float('-inf')
        best_action = Directions.STOP
        for action in gameState.getLegalActions(agent):
            successor_state = gameState.generateSuccessor(agent, action)
            successor_value, _ = self.minimax(successor_state, agent + 1, depth + 1)
            if successor_value > best_value:
                best_value, best_action = successor_value, action
        return best_value, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_alpha_beta_pruning(gameState, self.index, depth=0, alpha=float('-inf'),beta=float('inf'))[1]

    def minimax_alpha_beta_pruning(self, gameState, agent, depth, alpha, beta):
        num_agents = gameState.getNumAgents()
        if (depth == self.depth and agent % num_agents == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        if agent % num_agents == 0: # Pacman - maximize
            return self.maximize(gameState, agent % num_agents, depth, alpha, beta)
        return self.minimize(gameState, agent % num_agents, depth, alpha, beta) # Ghosts - minimize
    
    def minimize(self, state, agent, depth, alpha, beta):
        best_value = float('inf')
        best_action = Directions.STOP
        for action in state.getLegalActions(agent):
            successor_state = state.generateSuccessor(agent, action)
            next_value, _ = self.minimax_alpha_beta_pruning(successor_state, agent + 1, depth, alpha, beta)
            if next_value < best_value:
                best_value, best_action = next_value, action
            if best_value < alpha: # must be strictly less than
                return best_value, best_action
            beta = min(beta, best_value)
        return best_value, best_action

    def maximize(self, gameState, agent, depth, alpha, beta):
        best_value = float('-inf')
        best_action = Directions.STOP
        for action in gameState.getLegalActions(agent):
            successor_state = gameState.generateSuccessor(agent, action)
            successor_value, _ = self.minimax_alpha_beta_pruning(successor_state, agent + 1, depth + 1, alpha, beta)
            if successor_value > best_value:
                best_value, best_action = successor_value, action
            if best_value > beta: # must be strictly greater than
                return best_value, best_action
            alpha = max(alpha, best_value)
        return best_value, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, next_action = self.expectimax(gameState, self.index, 0)
        return next_action

    def expectimax(self, gameState, agent, depth):
        num_agents = gameState.getNumAgents()
        if (depth == self.depth and agent % num_agents == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        if agent % num_agents == 0: # Pacman - maximize
            return self.max_value(gameState, agent % num_agents, depth)
        return self.expected_value(gameState, agent % num_agents, depth) # Ghosts
    
    def expected_value(self, state, agent, depth) -> tuple[float, None]:
        values = []
        for action in state.getLegalActions(agent):
            successor_state = state.generateSuccessor(agent, action)
            next_value, _ = self.expectimax(successor_state, agent + 1, depth) 
            values += [next_value]
        return sum(values)/len(values), None

    def max_value(self, gameState, agent, depth) -> tuple[float, Directions]:
        best_value = float('-inf')
        best_action = Directions.STOP
        for action in gameState.getLegalActions(agent):
            successor_state = gameState.generateSuccessor(agent, action)
            successor_value, _ = self.expectimax(successor_state, agent + 1, depth + 1)
            if successor_value > best_value:
                best_value, best_action = successor_value, action
        return best_value, best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currentPos = currentGameState.getPacmanPosition()
    currGhostPos = currentGameState.getGhostPositions()
    currentFood = currentGameState.getFood().asList()
    currentCapsules =  currentGameState.getCapsules()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    "*** YOUR CODE HERE ***"
    # next food distance - the closer the merrier
    foodScore = 1/(1+ min([manhattanDistance(currentPos, food) for food in currentFood])) if currentFood else 1
    
    # next ghost distance - the further the merrier 
    ghostScore = 0
    cloestGhostDist = min([manhattanDistance(currentPos, ghostPos) for ghostPos in currGhostPos])
    # if ghost is going to eat the pacman, set it to -inf to avoid it
    if cloestGhostDist < 1:
        ghostScore = float('-inf')

    # power capsules count - the more the merrier
    capsuleScore = 1/ (1+len(currentCapsules)) 

    # scare time - the longer the merrier
    scareScore = 3 * min(currentScaredTimes)

    return foodScore + ghostScore + currentGameState.getScore()  + capsuleScore + scareScore
    

# Abbreviation
better = betterEvaluationFunction
