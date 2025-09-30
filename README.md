Python Version Python 3.7+

Files needed to run my code:
    - puzzle1.txt file
    - puzzle2.txt file
    - puzzle3.txt file
    - TileProblem.py
    - puzzleSolver.py

For data confirmation, you can use the following files to crosscheck with my PDF report:
    - N3D10.txt
    - N3D20.txt
    - N4D10.txt
    - N4D20.txt

How to run my code:
    - python puzzleSolver.py <A> <N> <H> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
        - A: the algorithm (A=1 for A* and A=2 for RBFS)
        - N: the size of puzzle (N=3 for 8-puzzle and N=4 for 15-puzzle)
            - (The value of N is required, but actually does not get used, since I pulled the size of the puzzle through input file)
        - H: for heuristics (H=1 for h1 and H=2 for h2)
            - H1: Manhattan Distance
            - H2: Misplaced Tiles
        - INPUT_FILE_PATH: the input filename
        - OUTPUT_FILE_PATH: the output filename
        - EX: python puzzleSolver.py 1 3 1 puzzle1.txt solution.txt


    