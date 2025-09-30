from typing import List, Tuple, Optional, Iterable, Dict

Action = str  # One of: "U", "D", "L", "R"

class TileProblem:
    def __init__(self, initial_state: List[List[Optional[int]]]):
        """
        initial_state: 2D list representing the puzzle.
            Example (3x3 / 8-puzzle):
                [[1, 2, 3],
                 [4, 5, 6],
                 [None, 7, 8]]
        None represents the blank tile.
        """
        self.initial_state = initial_state
        self.size = len(initial_state)

        # Basic structural checks
        assert self.size >= 2, "Puzzle must be at least 2x2."
        for row in initial_state:
            assert len(row) == self.size, "Initial state must be an N x N grid."

        self.goal_state = self._make_goal_state()


    def actions(self, state: List[List[Optional[int]]]) -> List[Action]:
        """
        Return the list of legal actions from the given state.
        Actions move the BLANK (None): "U", "D", "L", "R".
        """
        r, c = self._find_blank(state)
        moves: List[Action] = []

        ############ write your code between these blocks

        # add legal actions based on blank position (r, c) and board size
        if r > 0 : #Up - Have space on the row above
            moves.append("U")
        if r < self.size - 1 : #Down - Have space on the row below
            moves.append("D")
        if c > 0 : #Left - Have space on the column to the Left
            moves.append("L")
        if c < self.size - 1: #Right - Have space on the column to the Right
            moves.append("R")

        ################################

        return moves

    def result(self, state: List[List[Optional[int]]], action: Action) -> List[List[Optional[int]]]:
        """
        Return the new state after applying the action to move the blank.
        """
        r, c = self._find_blank(state)

        newState = [row[:] for row in state] # Make a deep copy of the state to return later

        ############ write your code between these blocks

        # compute new blank position (nr, nc), assert it's in-bounds, and swap
        currentNumber = None
        if action == "U":
            currentNumber = newState[r-1][c]
            newState[r-1][c] = None
            newState[r][c] = currentNumber
        elif action == "D":
            currentNumber = newState[r+1][c]
            newState[r+1][c] = None
            newState[r][c] = currentNumber
        elif action == "L":
            currentNumber = newState[r][c-1]
            newState[r][c-1] = None
            newState[r][c] = currentNumber
        elif action == "R":
            currentNumber = newState[r][c+1]
            newState[r][c+1] = None
            newState[r][c] = currentNumber

        return newState

        ################################


    def goal_test(self, state: List[List[Optional[int]]]) -> bool:
        """Check if the puzzle is solved."""
        return state == self.goal_state

    # Placeholder function
    def step_cost(self, state: List[List[Optional[int]]], action: Action, next_state: List[List[Optional[int]]]) -> int:

        return 1

    # ---------- Helpers ----------

    def _make_goal_state(self) -> List[List[Optional[int]]]:
        """Return the solved puzzle configuration as a 2D list with blank at the last cell."""
        n = self.size
        tiles = list(range(1, n * n)) + [None]
        return [tiles[i * n:(i + 1) * n] for i in range(n)]

    def _find_blank(self, state: List[List[Optional[int]]]) -> Tuple[int, int]:
        """Locate the (row, col) of the blank tile (None)."""
        for r in range(self.size):
            for c in range(self.size):
                if state[r][c] is None:
                    return (r, c)
        raise ValueError("No blank tile (None) found in state")

    def flatten(state: List[List[Optional[int]]]) -> List[Optional[int]]:
        """Return a 1D list view of the state (useful for hashing/printing/debug)."""
        return [cell for row in state for cell in row]

    def __str__(self) -> str:
        return "\n".join(" ".join("_" if v is None else str(v) for v in row) for row in self.initial_state)
