# CS 441 - Programming Assignment 1 - Dan Jang
# Implementing a 3-by-3, 8-puzzle solver using best-first search & three A* heuristics

import random

def create8puzzle():
    # The 3x3 8-puzzle goal
    
    # Create a random 8-puzzle
    initpuzzle = []
    puzzle = []
    idx = 0
    
    while idx < 9:
        number = random.randint(0,8)
        if number not in initpuzzle:
            initpuzzle.append(number)
            idx += 1
    
    for n in range (3):
        puzzle.append(initpuzzle[(n*3):(n+1)*3])
        
    return puzzle

# For direction, 0 = left, 1 = right, 2 = up, 3 = down
def move(puzzle, x, y, direction):
    
    # If direction is left.
    if direction == 0:
        # If the blank tile is on the left edge, return none.
        if y != 0:
            return swap(puzzle, x, y, x, y-1)
        else:
            return None
    
    # If direction is right.
    if direction == 1:
        # If the blank tile is on the right edge, return none.
        if y != 2:
            return swap(puzzle, x, y, x, y+1)
        else:
            return None
    
    # If direction is up.
    if direction == 2:
        # If the blank tile is on the top edge, return none.
        if x != 0:
            return swap(puzzle, x, y, x-1, y)
        else:
            return None
     
     # If direction is down.   
    if direction == 3:
        # If the blank tile is on the bottom edge, return none.
        if x != 2:
            return swap(puzzle, x, y, x+1, y)
        else:
            return None

def swap(puzzle, x, y, n, k):
    
    curr = puzzle
    
    first = curr[x][y]
    second = curr[n][k]
    
    curr[x][y] = second
    curr[n][k] = first
    
    return curr

def search(puzzle, num):
    x,y = 0,0
    
    for idx in range(3):
        if num in puzzle[idx]:
            x = idx
            y = puzzle[idx].index(num)
            
    return x,y

if __name__ == '__main__':
    
    goal = [[1,2,3],[4,5,6],[7,8,0]]
    thepuzzle = create8puzzle()
    
    print("The 8-puzzle generated is:")
    for idx in range(3):
        print(thepuzzle[idx])
        
    print("The goal state of the 8-puzzle is:")
    for idx in range(3):
        print(goal[idx])
        
    print("The blank tile is at position: ", search(thepuzzle, 0))
    print(move(thepuzzle, 2, 1, 2))