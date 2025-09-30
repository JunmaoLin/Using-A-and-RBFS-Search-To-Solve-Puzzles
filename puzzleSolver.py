"""
puzzleSolver.py
Author: Junmao Lin
Description: This program should be able to take as input an
    8 or a 15 puzzle and output the set of moves required to solve the
    problem.
Usage: 
    python puzzleSolver.py <A> <N> <H> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>

    Arguments:
        A: the algorithm (A=1 for A* and A=2 for RBFS)
        N: the size of puzzle (N=3 for 8-puzzle and N=4 for 15-puzzle)
        H: for heuristics (H=1 for h1 and H=2 for h2)
        INPUT_FILE_PATH: the input filename
        OUTPUT_FILE_PATH: the output filename
"""

# Assuming all inputs are valid and solvable based on piazza.
import sys, time, heapq
# import puzzleGenerator 
import TileProblem
from dataclasses import dataclass
from typing import List, Optional


# Reading/Writing File -----------------------------------------------------------------------
def readInputFileNReturnGivenState(inputFilePath: str) -> List[List[Optional[int]]]:
    """
    Reads the input file and returns the given state as a 2D list.
    """
    givenState = []
    #Ex: 1,2,3
    #    4,5,6
    #    ,7,8
    #Read input file and store the state in givenState in 2D list
    with open(inputFilePath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() # Remove all leading and trailing whitespace a long with \n
            inputRow = line.split(',')
            appendingRow = []
            # Convert str into int, and if empty string, convert to None
            for i in range(len(inputRow)):
                appendingRow.append(int(inputRow[i]) if inputRow[i] != '' else None)
            givenState.append(appendingRow)
    return givenState

def writeToOutputFilePath(outputFilePath: str, solutionMoves: str) -> None:
    """
    Writes the solution moves to the output file.
    """
    with open(outputFilePath, 'w') as f:
        f.write(solutionMoves)
# End of Reading/Writing File ----------------------------------------------------------------

# Heuristics functions -----------------------------------------------------------------------
def h1ManhattanDistance(state: List[List[Optional[int]]], tileProblem: TileProblem) -> int:
    """
    H1: Manhattan Distance
    For each tile, compute how far it is from its goal position in terms of row column steps. 
    The heuristic value is the sum of these distances for all tiles (ignoring the blank).
    """
    totalManhattanDistance = 0
    size = tileProblem.size
    # N by N problem
    # 1, 2, 3
    # 4, 5, 6
    # 7, 8 
    # n-1 % size gives column
    # n-1 // size gives row
    for r in range(size):
        for c in range(size):
            tile = state[r][c]
            # Instead of looping over the goal state for R and C, we can compute directly to save time
            if tile is not None: 
                goalR = int((tile - 1) // size)
                goalC = int((tile - 1) % size)
                totalManhattanDistance += abs(r - goalR) + abs(c - goalC)
    return totalManhattanDistance

def h2MisplacedTiles(currentState: List[List[Optional[int]]], tileProblem: TileProblem) -> int:
    """
    H2: Misplaced Tiles
    Count how many tiles are not currently in their goal position (ignoring the blank)
    """
    countOfMisplacedTiles = 0
    size = tileProblem.size
    goalState = tileProblem.goal_state
    for r in range(size):
        for c in range(size):
            if currentState[r][c] is not None and currentState[r][c] != goalState[r][c]:
                countOfMisplacedTiles += 1
    return countOfMisplacedTiles
# End of Heuristics functions ----------------------------------------------------------------

# Creating Node Class ------------------------------------------------------------------------
@dataclass
class Node:
    """
    The Node class will store the current state, parent node, action taken to reach this state,
    g (cost from start to this node), h (heuristic estimate from this node to goal), and f (total estimated cost).
    """
    state: List[List[Optional[int]]]
    parent: Optional["Node"]
    action: Optional[str]
    g: int
    h: int

    @property
    # Access f by node.f instead of node.f()
    def f(self) -> int:
        return self.g + self.h
    
    # Get the path from root to this node
    def getPath(self) -> List[str]:
        if self.parent is None:
            return []
        return self.parent.getPath() + [self.action]
    
    # Use this to compare nodes in priority queue based on f value
    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f
    
    # For keeping track of explored states
    def getStateKey(self) -> str:
        return str([cell for row in self.state for cell in row])
    
# End of Node Class --------------------------------------------------------------------------

# Search Functions ---------------------------------------------------------------------------
def aStarSearch(tileProblem: TileProblem, heuristicFunction) -> tuple[List[str], int, float, float]:
    """
    A* Search Algorithm implementation based on "informed search" lecture slides.
    Returns the solution path as a list of actions, number of nodes expanded, and total time taken.
    """

    # Used for performance measurement
    startTime = time.time()
    nodeExpanded = 0

    # frontier = priority_queue() // Sort frontier on expected path cost, f(n) = g(n) + h(n)
    frontier = []

    # this is the root node
    startNode = Node(
        state = tileProblem.initial_state,
        parent = None,
        action = None,
        g = 0,
        h = heuristicFunction(tileProblem.initial_state, tileProblem)
    )

    # frontier = frontier + make-node(start)
    heapq.heappush(frontier, (startNode.f, 0, startNode))

    explored = set()
    tieBreaker = 1 # To avoid comparison issue in heapq when f values are the same

    # while not frontier.is-empty():
    while frontier:
        # current <- pop(frontier) // i.e., the top of the queue.
        nodeF, _, currentNode = heapq.heappop(frontier)
        nodeExpanded += 1

        # if goal-test(current) return success // goal test when node expands
        if tileProblem.goal_test(currentNode.state):
            endTime = time.time()
            totalTime = (endTime - startTime) * 1000 # in milliseconds
            nodeSize = sys.getsizeof(currentNode) # in bytes
            estimatedMemory = (len(frontier) + len(explored)) * nodeSize
            return currentNode.getPath(), nodeExpanded, totalTime, estimatedMemory
        
        # if current not in explored:
        currentStateKey = currentNode.getStateKey()
        if currentStateKey not in explored:
            # explored <- explored + current.state
            explored.add(currentStateKey)
            # for each action in current.actions():
            for action in tileProblem.actions(currentNode.state): # go through possible actions
                # new <- action(current.state)
                newState = tileProblem.result(currentNode.state, action)
                # new-node <- make-node(new, current, action)
                newNode = Node(
                    state = newState,
                    parent = currentNode,
                    action = action,
                    g = currentNode.g + tileProblem.step_cost(currentNode.state, action, newState),
                    h = heuristicFunction(newState, tileProblem)
                )
                # the newly created node will be added to the frontier to explore later
                # frontier = frontier + new-node
                heapq.heappush(frontier, (newNode.f, tieBreaker, newNode))
                tieBreaker += 1

    # return failure
    totalTimeSearch = (time.time() - startTime) * 1000 # in milliseconds
    try:
        nodeSize = sys.getsizeof(currentNode)
    except NameError:
        nodeSize = sys.getsizeof(Node)
    estimatedMemory = (len(frontier) + len(explored)) * nodeSize
    return [], nodeExpanded, totalTimeSearch, estimatedMemory

def rbfsSearch(tileProblem: TileProblem, heuristicFunction) -> tuple[List[str], int, float]:
    """
    Recursive Best-First Search (RBFS) Algorithm implementation based on "https://pages.mtu.edu/~nilufer/classes/cs5811/2012-fall/lecture-slides/cs5811-ch03-search-b-informed-v2.pdf".
    Returns the solution path as a list of actions, number of nodes expanded, and total time taken.
    """

    # Used for performance measurement
    startTime = time.time()
    nodesExpanded = [0]
    maxRecursionDepth = [0]
    currentDepth = [0]

    # function RBFS (problem, node, f-limit)
    def rbfs(node: Node, f_limit: int) -> tuple[Optional[Node], int]:
        nodesExpanded[0] += 1
        currentDepth[0] += 1
        if currentDepth[0] > maxRecursionDepth[0]:
            maxRecursionDepth[0] = currentDepth[0]
        
        # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
        if tileProblem.goal_test(node.state):
            currentDepth[0] -= 1
            return node, nodesExpanded[0]
        
        #successors ←[ ]
        successors = []

        # for each action in problem.ACTIONS(node.STATE) do
        for action in tileProblem.actions(node.state):
            # add CHILD-NODE(problem, node, action) into successors
            newState = tileProblem.result(node.state, action)
            newNode = Node(
                state = newState,
                parent = node,
                action = action, 
                g = node.g + tileProblem.step_cost(node.state, action, newState),
                h = heuristicFunction(newState, tileProblem)
            )
            successors.append(newNode)

        # if successors is empty then return failure, ∞
        if not successors:
            return None, float('inf')
        
        # for each s in successors do
        for s in successors:
            # /* update f with value from previous search, if any */
            # s.f←max (s.g + s.h, node.f ))
            s.backedUpF = max(s.g + s.h, node.f)

        # loop do
        while True:
            # best ←the lowest f -value in successors
            best = min(successors, key=lambda s: s.backedUpF)

            # if best.f > f-limit then return failure, best.f
            if best.backedUpF > f_limit:
                return None, best.backedUpF
            
            # alternative ←the second lowest f-value among successors
            alternatives = [s for s in successors if s != best]
            if alternatives:
                alternative = min(alternatives, key=lambda s: s.backedUpF)
                alternative_f = alternative.backedUpF
            else:
                alternative_f = float('inf')

            # result, best.f ←RBFS (problem, best, min(f-limit,alternative))
            result, best.backedUpF = rbfs(best, min(f_limit, alternative_f))

            # if result ̸= failure then return result
            if result is not None:
                return result, best.backedUpF
    
    startingNode = Node(
        state = tileProblem.initial_state,
        parent = None,
        action = None,
        g = 0,
        h = heuristicFunction(tileProblem.initial_state, tileProblem)
    )
    startingNode.backedUpF = startingNode.f

    solution, _ = rbfs(startingNode, float('inf'))

    endtime = time.time()
    totalTime = (endtime - startTime) * 1000 # in milliseconds

    nodeSize = sys.getsizeof(startingNode) # in bytes
    estimatedMemory = maxRecursionDepth[0] * nodeSize

    if solution is not None:
        return solution.getPath(), nodesExpanded[0], totalTime, estimatedMemory
    else:
        return [], nodesExpanded[0], totalTime, estimatedMemory

# End of Search Functions --------------------------------------------------------------------

# Main function ------------------------------------------------------------------------------
def main(argv):
    # Getting command line arguments
    algorithm = int(sys.argv[1]) # A: the algorithm (A=1 for A* and A=2 for RBFS)
    # The size here seems unnecessary since the size is already determined when we analyze the input file
    size = int(sys.argv[2])      # N: the size of puzzle (N=3 for 8-puzzle and N=4 for 15-puzzle) 
    heuristics = int(sys.argv[3])# H: for heuristics (H=1 for h1 and H=2 for h2)
    inputFilePath = sys.argv[4]  # <INPUT FILE PATH>
    outputFilePath = sys.argv[5] # <OUTPUT FILE PATH>

    givenState = readInputFileNReturnGivenState(inputFilePath)
    tileProblem = TileProblem.TileProblem(givenState)

    # Selecting heuristic function based on user input
    heuristicFunction = None
    heuristic = ""
    if heuristics == 1:
        heuristicFunction = h1ManhattanDistance
        heuristic = "H1 - Manhattan Distance"
    elif heuristics == 2:
        heuristicFunction = h2MisplacedTiles
        heuristic = "H2 - Misplaced Tiles"
    
    if algorithm == 1:
        print("A* Search Algorithm with " + heuristic + ":")
        solutionPath, nodesExpanded, totalTime, estimatedMemory  = aStarSearch(tileProblem, heuristicFunction)
    elif algorithm == 2:
        print("RBFS Search Algorithm with " + heuristic + ":")
        solutionPath, nodesExpanded, totalTime, estimatedMemory = rbfsSearch(tileProblem, heuristicFunction)
    
    if solutionPath:
        print("Solution found.")
    else:
        print("No solution found.")

    print("Output: " + str(solutionPath))
    print("Number of States Explored: " + str(nodesExpanded))
    print("Total Time Taken (milliseconds): " + str(totalTime))
    print("Solution Depth: " + str(len(solutionPath)))
    print("Estimated Memory Usage (bytes): " + str(estimatedMemory))

    # writeToOutputFilePath(outputFilePath, ', '.join(solutionPath))
    writeToOutputFilePath(outputFilePath, ','.join(solutionPath)) # Spaces between each move is removed to pass the format check


if __name__ == "__main__":
    main(sys.argv)
