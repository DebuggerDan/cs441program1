### CS 441 - Programming Assignment 1 - Dan Jang
### Implementing a 3-by-3, 8-puzzle solver using best-first search & three A* heuristics

# import random
import numpy as np
import math
import heapq as hq

def create8puzzle(idx):
    # The 3x3 8-puzzle goal
    
    # Create a random 8-puzzle where the blank tile is in a random position, as 'b'
    if idx == 0:
        puzzle = [5,1,0,7,8,6,2,4,3]
    if idx == 1:
        puzzle = [4,5,0,6,1,8,7,3,2]
    if idx == 2:
        puzzle = [2,8,4,5,1,0,7,3,6]
    if idx == 3:
        puzzle = [7,5,1,0,8,3,4,6,2]
    if idx == 4:
        puzzle = [3,8,4,6,7,0,1,5,2]
    # initpuzzle = []
    # puzzle = []
    # idx = 0
    
    # while idx < 9:
    #     number = random.randint(0,8)
    #     if number not in initpuzzle:
    #         initpuzzle.append(number)
    #         idx += 1
    
    # for n in range (3):
    #     puzzle.append(initpuzzle[(n*3):(n+1)*3])
    
        
    return puzzle

def create15puzzle(idx):
    # The 4x4 15-puzzle goal
    
    # Create a random 8-puzzle where the blank tile is in a random position, as 'b'
    if idx == 0:
        puzzle = [8,4,0,13,15,3,14,10,2,12,11,6,7,1,9,5]
    if idx == 1:
        puzzle = [5,9,7,15,0,2,10,8,14,11,6,4,13,3,1,12]
    if idx == 2:
        puzzle = [2,12,13,9,11,1,7,8,0,10,14,6,15,5,3,4]
    if idx == 3:
        puzzle = [8,14,2,15,12,13,5,1,4,9,6,0,3,10,7,11]
    if idx == 4:
        puzzle = [10,5,8,6,3,12,14,7,9,15,0,2,4,11,13,1]
        
    # initpuzzle = []
    # puzzle = []
    # idx = 0
    
    # while idx < 16:
    #     number = random.randint(0,15)
    #     if number not in initpuzzle:
    #         initpuzzle.append(number)
    #         idx += 1
    
    # for n in range (4):
    #     puzzle.append(initpuzzle[(n*4):(n+1)*4])
    
        
    return puzzle

# For direction, 0 = left, 1 = right, 2 = up, 3 = down
def move(currentmove, x, y, direction, dimension):
    
    # If direction is left.
    if direction == 0:
        # If the blank tile is on the left edge, return none.
        if y != 0:
            return swap(currentmove, x, y, x, y-1)
        else:
            return None
    
    # If direction is right.
    if direction == 1:
        # If the blank tile is on the right edge, return none.
        if y != dimension - 1:
            return swap(currentmove, x, y, x, y+1)
        else:
            return None
    
    # If direction is up.
    if direction == 2:
        # If the blank tile is on the top edge, return none.
        if x != 0:
            return swap(currentmove, x, y, x-1, y)
        else:
            return None
     
     # If direction is down.   
    if direction == 3:
        # If the blank tile is on the bottom edge, return none.
        if x != dimension - 1:
            return swap(currentmove, x, y, x+1, y)
        else:
            return None

def swap(swapmove, x, y, n, k):
    
    curr = swapmove
    
    first = curr[x][y]
    second = curr[n][k]
    
    curr[x][y] = second
    curr[n][k] = first
    
    return curr

def search(searchcurr, num, dimension):
    x,y = 0,0
    
    for idx in range(dimension):
        if num in searchcurr[idx]:
            x = idx
            y = searchcurr[idx].index(num)
            
    return x,y

## Solution Path / Route Helper Functions

# (Re)-Navigate the Solution Path
# Input: Puzzle Map, Current State, Move Costs
# Output: Move List
def navigate(prevstates, location, costs):
    # Initialize the route as a empty list/array
    route = []
    
    while location is not None:
        route.append(location)
        location = prevstates[str(location)]
    
    return route

# Main Print (For Results) Function
# Input: Move List, Search Type (Best-First-Search or A* Search), Heuristic Type (Manhattan, Misplaced, or Linear Conflict)
# Output: Number of moves to use for average moves calculation

def results(route, stype, htype):
    if route is not None:
        # Printing the Solution Path
        for idx in range(len(route)):
            print(route[idx])
            if idx != len(route)-1:
                print("End of Solution Path\n")
        
        print("Solution Path Statistics:")
        print("Move(s) Taken:" + str(len(route) - 1))
        return (len(route) - 1)
        
    if route is None:
        # Debug #5 / {Valid} Result Statement
        print("No Solution Path found for " + str(stype) + "-type search with " + str(htype) + "-type heuristic.")
        return None
    
## Heuristic Search Helper Functions

def identify(puzzle, pos):
    # convert = np.vectorize(np.int_)
    # puzzle = convert(puzzle)
    # result = int(np.where(puzzle == pos)[0])
    # curr_x = 0
    # curr_y = 0
    # for x in range(len(puzzle)):
    #     for y in range(len(puzzle)):
    #         # If we have found the location of the blank tile
    #         if puzzle[x][y] == pos:
    #             curr_x = x
    #             curr_y = y
                
    #             break
    
    #result = curr_x * curr_y
    result = puzzle.index(pos)

    return result

def difference(puzzle, goal, num, type, dimension):
    # If the type (1) is for row-difference
    if type == 1:
        x = identify(puzzle, num)
        y = identify(goal, num)
        rowresult = abs((x // dimension) - (y // dimension))
        return int(rowresult)
    
    # If the type (2) is for column-difference
    if type == 2:
        x = identify(puzzle, num)
        b = identify(goal, num)
        columnresult = abs((x % dimension) - (b % dimension))
        return int(columnresult)
    
    else:
        # Debug Statement #1
        print("Error [difference(puzzle, goal, num, type, dimension)]: Invalid type!")
        return None
        
# Heuristic #1 Implementation (Number of Misplaced Tiles)
# Implement A* Search using Heuristic 1 - The number of misplaced tiles (blank not included)
# Input: 3x3 8-puzzle (or 4x4 15-puzzle), heuristic 1, goal state, dimension of puzzle
# Output: Solution Path, if found.

def heuristic_one(puzzle1, goal, dimension):
    # Initalize heuristic #1 (cost) value to 0
    cost1 = 0
    
    # # For the 3x3 8-puzzle
    # if dimension == 3:
    #     for idx in range(1,9):
    #         #if difference(puzzle, goal, idx, 1, dimension) != 0 or difference(puzzle, goal, idx, 2, dimension) != 0:
    #             #cost += 1
    #         if puzzle[idx] != goal[idx]:
    #             cost += 1
    #     return cost
    
    # # For the 4x4 15-puzzle extra credit
    # if dimension == 4:
    #     for idx in range(1, 15):
    #         # if difference(puzzle, goal, idx, 1, dimension) != 0 or difference(puzzle, goal, idx, 2, dimension) != 0:
    #         #     cost += 1
    #         if puzzle[idx] != goal[idx]:
    #             cost += 1
    #     return cost
    
    for i in range(dimension * dimension):
        if puzzle1[i] != goal[i]:
            cost1 += 1
    return cost1
    
    # else:
    #     # Debug Statement #2a
    #     print("Error [heuristic_one(puzzle, goal, dimension)]: Invalid dimension!")
    #     return None

# Heuristic #2 Implementation (The Manhattan / City-Block Distance)
# Implement A* Search using Heuristic 2 - The Manhattan / City-Block Distance
# Input: 3x3 8-puzzle (or 4x4 15-puzzle), heuristic 2, goal state, dimension of puzzle
# Output: Solution Path, if found.

def heuristic_two(puzzle2, goal, dimension):
    # Initalize heuristic #2 (cost) value to 0
    cost2 = 0
    # Debug Print
    #print("HEURISTIC COST TEST BEFORE #3" + str(cost))
    # For the 3x3 8-puzzle
    # if dimension == 3:
    #     for idx in range(1, 9):
    #         cost += difference(puzzle, goal, idx, 1, dimension) + difference(puzzle, goal, idx, 2, dimension)
    #     # Debug Print
    #     #print("HEURISTIC COST TEST BEFORE #3" + str(cost))
    #     return cost

    # # For the 4x4 15-puzzle extra credit
    # if dimension == 4:
    #     for idx in range(1, 15):
    #         cost += difference(puzzle, goal, idx, 1, dimension) + difference(puzzle, goal, idx, 2, dimension)
    #     # Debug Print
    #     #print("HEURISTIC COST TEST BEFORE #3" + str(cost))
    #     return cost

    for i in range(dimension * dimension):
        if puzzle2[i] != 0:
            goalindex = goal.index(puzzle2[i])
            
            currrow = i // dimension
            currcolumn = i % dimension
            
            cost2 += abs(currrow - (goalindex // dimension)) + abs(currcolumn - (goalindex % dimension))

    return cost2

    # else:
    #     # Debug Statement #2b
    #     print(
    #         "Error [heuristic_two(puzzle, goal, dimension)]: Invalid dimension!")
    #     return None

# Heuristic #3 Implementation (The Euclidean Distance [from goal position{s}])
# Implement A* Search using Heuristic 3 - The sum of Euclidean (srqt(x^2 + y^2)) distances from goal position{s}
# Input: 3x3 8-puzzle (or 4x4 15-puzzle), heuristic 3, goal state, dimension of puzzle
# Output: Solution Path, if found.

def heuristic_three(puzzle3, goal, dimension):
    # Initalize heuristic #3 (cost) value to 0
    cost = 0

    # For the 3x3 8-puzzle
    if dimension == 3:
        for idx in range(1, 9):
            one = difference(puzzle3, goal, idx, 1, dimension)
            two = difference(puzzle3, goal, idx, 2, dimension)
            cost += math.sqrt(pow(one, 2) + pow(two, 2))
        return cost

    # For the 4x4 15-puzzle extra credit
    if dimension == 4:
        for idx in range(1, 15):
            one = difference(puzzle3, goal, idx, 1, dimension)
            two = difference(puzzle3, goal, idx, 2, dimension)
            cost += math.sqrt(pow(one, 2) + pow(two, 2))
        return cost

    # else:
    #     # Debug Statement #2c
    #     print(
    #         "Error [heuristic_three(puzzle, goal, dimension)]: Invalid dimension!")
    #     return None

# Row check helper function
# def check(puzzle, value):
#     rowcheck = []

def generate_moves(themoves, dimension):
    generatedmoves = []
    curr = themoves
    # Current position of blank tile
    # Debug Print #1
    #for idx in range(len(themoves)):
    #    print("row #" + str(idx) + ": " + str(themoves[idx]))
    
    #curr = puzzle.index(0)#identify(puzzle, 0)#search(puzzle, 0, dimension)
    #for idx in range(len(puzzle)):
    #curr = search(puzzle, 0, dimension)
    #curr = identify(puzzle, 0)
    #curr = list(zip(*np.where(puzzle == 0)))
    # curr_x = 0
    # curr_y = 0
    # for x in range(len(themoves)):
    #     for y in range (len(themoves)):
    #         # If we have found the location of the blank tile
    #         if curr[x][y] == 0:
    #             curr_x = x
    #             curr_y = y
                
    #             break
    blanktile = themoves.index(0)
    if blanktile >= dimension:
        # Move the blank tile up
        newmoves = themoves[:]
        newmoves[blanktile - dimension], newmoves[blanktile] = newmoves[blanktile], newmoves[blanktile - dimension]
        generatedmoves.append(newmoves)
    
    if blanktile < (dimension * dimension) - dimension:
        # Move the blank tile down
        newmoves = themoves[:]
        newmoves[blanktile + dimension], newmoves[blanktile] = newmoves[blanktile], newmoves[blanktile + dimension]
        generatedmoves.append(newmoves)
        
    if blanktile % dimension != 0:
        # Move the blank tile left
        newmoves = themoves[:]
        newmoves[blanktile - 1], newmoves[blanktile] = newmoves[blanktile], newmoves[blanktile - 1]
        generatedmoves.append(newmoves)
        
    if blanktile % dimension != dimension - 1:
        # Move the blank tile right
        newmoves = themoves[:]
        newmoves[blanktile + 1], newmoves[blanktile] = newmoves[blanktile], newmoves[blanktile + 1]
        generatedmoves.append(newmoves)
    
    return generatedmoves
    
    # #print(str(curr[curr_x][curr_y]))
    
    # # If the puzzle is 4x4
    # if dimension:
    #     # # If blank tile is not within left row
    #     # # if curr >= dimension:
    #     # if curr_x != 0:
    #     #     moves.append(move(puzzle, curr, curr - dimension))

    #     # If blank tile is not within left row, move left
    #     #if curr < dimension * (dimension - 1):
    #     if curr_x != dimension - 1:
    #         # For direction, 0 = left, 1 = right, 2 = up, 3 = down
    #         generatedmoves.append(move(themoves, curr_x, curr_y, 0, dimension))
            
    #     # If blank tile is not within right row, move right
    #     #if curr % dimension != dimension - 1:
    #     if curr_y != dimension - 1:
    #         #moves.append(puzzle[:curr] + [puzzle[curr + 1]] + puzzle[curr + 1:curr + dimension] + [0] + puzzle[curr + dimension + 1:])
    #         generatedmoves.append(move(themoves, curr_x, curr_y, 1, dimension))
            
    #     # If blank tile is not within first row, move up
    #     #if curr >= dimension:
    #     if curr_x != 0:
    #         #moves.append(puzzle[:curr - dimension] + [0] + puzzle[curr - dimension + 1:curr] + [puzzle[curr - dimension]] + puzzle[curr + 1:])
    #         generatedmoves.append(move(themoves, curr_x, curr_y, 2, dimension))
            
    #     # If blank tile is not within left row, move down
    #     #if curr % dimension != 0:
    #     if curr_y != 0:
    #         #moves.append(puzzle[:curr - 1] + [0] + puzzle[curr:curr + dimension - 1] + [puzzle[curr - 1]] + puzzle[curr + dimension:])
    #         generatedmoves.append(move(themoves, curr_x, curr_y, 3, dimension))
    
    # # else:
    # #     # Debug Statement #7
    # #     print("Error [generate_moves(puzzle, dimension)]: Invalid dimension!")
    # #     return None
    # # If the puzzle is 3x3
    # #if dimension == 3:
        
            
    # return generatedmoves
    
    # # If the puzzle is 3x3
    # if dimension == 3:
    #     if curr >= dimension:
    #         # Puzzle is 3x3
    #         moves.append(puzzle[:curr - dimension] + [0] + puzzle[curr - dimension + 1:curr] + [puzzle[curr - dimension]] + puzzle[curr + 1:])
    #     if curr < dimension * (dimension - 1):
    #         moves.append(puzzle[:curr] + [puzzle[curr + dimension]] + puzzle[curr + 1:curr + dimension] + [0] + puzzle[curr + dimension + 1:])
    #     if curr % dimension != 0:
    #         moves.append(puzzle[:curr - 1] + [0] + [puzzle[curr - 1]] + puzzle[curr + 1:])
    #     if curr % dimension != (dimension - 1):
    #         moves.append(puzzle[:curr] + [puzzle[curr + 1]] + [0] + puzzle[curr + 2:])
    
    #return moves

# A* Search Implementation /w Three Different Heuristics
# Input: Puzzle array, goal state, dimension of puzzle, the type of heuristic to use
# Output: List of moves, sorted by heuristic type & costs, if solution path found.

def a_asterisksearch(puzzle, goal, dimension, type, max=9999):
    # Map of moves and of path(s) taken
    puzzlemap = []
    
    hq.heappush(puzzlemap, (0, puzzle))
    
    # Empty list of explored states
    prevstates = {}
    
    # Costs of each path compiled
    costs = {}
    
    prevstates[str(puzzle)] = None
    costs[str(puzzle)] = 0
    
    initmap = []
    hq.heappush(initmap, (0, puzzle))
    init = hq.heappop(initmap)[1]
    costs[str(init)] = 0
    totalmoves = 0
    
    while puzzlemap:
        curr = hq.heappop(puzzlemap)[1]
        
        if curr == goal:
            return navigate(prevstates, curr, costs)
    
        # List of moves
        moves = []
        moves = generate_moves(curr, dimension)
        idx = []
        for idx in moves:
            if totalmoves >= max:
                # Result Statement / Debug Statement #10
                print("Error [a_asterisksearch(puzzle, goal, dimension, type)]: Max number of moves reached!")
                return None
            totalmoves += 1
            initmap = []
            hq.heappush(initmap, (0, puzzle))
            init = hq.heappop(initmap)[1]
            costs[str(init)] = 0
            currcost = costs[str(curr)] + 1
            if str(idx) not in costs or currcost < costs[str(idx)]:
                costs[str(idx)] = currcost
                prevstates[str(idx)] = curr
                
                # Heuristic #1 - Number of Misplaced Tiles
                if type == 1:
                    hq.heappush(puzzlemap, ((heuristic_one(idx, goal, dimension) + currcost), idx))
                
                # Heuristic #2 - Manhattan / City-Block Distance
                elif type == 2:
                    hq.heappush(puzzlemap, ((heuristic_two(idx, goal, dimension) + currcost), idx))
                
                # Heuristic #3 - Sum of Euclidean Distances 
                elif type == 3:
                    hq.heappush(puzzlemap, ((heuristic_three(idx, goal, dimension) + currcost), idx))
                
            # if idx >= max:
            #     # Result Statement / Debug Statement #8
            #     print("Error [a_asterisksearch(puzzle, goal, dimension, type)]: Max number of moves reached!")
            #     return None
    
    # Return none if no solution path found
    return None
    

    # # List of moves
    # moves = []
    
    # for idx in range(dimension + 1):
    #     moves.append(move(puzzle, search(puzzle, 0, dimension)[0], search(puzzle, 0, dimension)[1], idx))
        
    # # Remove invalid / None values
    # moves = [x for x in moves if x != None]
    
    # # If heuristic #1 is used for this A* Search (# of Misplaced Tiles)
    # if type == 1:
    #     # List of moves and heuristic #1 values
    #     heuristic1 = []
    #     for idx in moves:
    #         heuristic1.append((idx, heuristic_one(idx, goal, dimension)))
            
    #     # Sort list of moves by heuristic #1 values
    #     heuristic1.sort(key=lambda x: x[1])
        
    #     return heuristic1[0][0]
    
    # # If heuristic #2 is used for this A* Search (Manhattan / City-Block Distance)
    # if type == 2:
    #     # List of moves and heuristic #2 values
    #     heuristic2 = []
    #     for idx in moves:
    #         heuristic2.append((idx, heuristic_two(idx, goal, dimension)))

    #     # Sort list of moves by heuristic #2 values
    #     heuristic2.sort(key=lambda x: x[1])

    #     return heuristic2[0][0]
    
    # # If heuristic #3 is used for this A* Search (Sum of Euclidean Distances)
    # if type == 3:
    #     # List of moves and heuristic #3 values
    #     heuristic3 = []
    #     for idx in moves:
    #         heuristic3.append((idx, heuristic_two(idx, goal, dimension)))
            
    #     heuristic3.sort(key=lambda x: x[1])
        
    #     return heuristic3[0][0]
    
    # # If an invalid heuristic type is used
    # else:
    #     # Debug Statement #3
    #     print("Error [a_asterisksearch(puzzle, goal, dimension, type)]: Invalid type {of heuristic}!")
    #     return None

# Best First Search Implementation /w Three Different Heuristics
# Input: Puzzle array, goal state, dimension of puzzle, the type of heuristic to use
# Output: List of moves, sorted by heuristic type & costs, if solution path found.
def bestfirstsearch(givenpuzzle, goal, dimension, type, max):
    # Map of moves and of path(s) taken
    puzzlemap = []
    
    hq.heappush(puzzlemap, (0, givenpuzzle))
    
    # Empty list of explored states
    prevstates = {}
    
    # Costs of each path compiled
    costs = {}
    
    prevstates[str(givenpuzzle)] = None
    costs[str(givenpuzzle)] = 0
    
    initmap = []
    hq.heappush(initmap, (0, givenpuzzle))
    init = hq.heappop(initmap)[1]
    costs[str(init)] = 0
    totalmoves = 0
    
    while puzzlemap:
        curr = []
        curr = hq.heappop(puzzlemap)[1]
        
        # Debug Statement:
        #print(str(curr))
        #print(str(goal))
        
        if str(curr) == str(goal):
            return navigate(prevstates, curr, costs)
    
        # List of moves
        moves = []
        moves = generate_moves(curr, dimension)
        currentstate = []
        for currentstate in moves:
            if totalmoves >= max:
                # Result Statement / Debug Statement #9
                print("Error [bestfirstsearch(givenpuzzle, goal, dimension, type)]: Max number of moves reached!")
                return None
            priority = 0
            totalmoves += 1
            initmap = []
            hq.heappush(initmap, (0, givenpuzzle))
            init = hq.heappop(initmap)[1]
            costs[str(init)] = 0
            currcost = 0
            currcost = costs[str(curr)] + 1
            # Debug IDX Move Print
            #print("currentstate: ", currentstate)
            if str(currentstate) not in costs or currcost < costs[str(currentstate)]:
                costs[str(currentstate)] = currcost
                prevstates[str(currentstate)] = curr
                
                # Heuristic #1 - Number of Misplaced Tiles
                if type == 1:
                    priority = currcost + heuristic_one(currentstate, goal, dimension)
                    #hq.heappush(puzzlemap, ((heuristic_one(idx, goal, dimension) + currcost), idx))
                
                # Heuristic #2 - Manhattan / City-Block Distance
                elif type == 2:
                    priority = currcost + heuristic_two(currentstate, goal, dimension)
                    #hq.heappush(puzzlemap, ((heuristic_two(idx, goal, dimension) + currcost), idx))
                
                # Heuristic #3 - Sum of Euclidean Distances 
                elif type == 3:
                    priority = currcost + heuristic_three(currentstate, goal, dimension)
                    #hq.heappush(puzzlemap, ((heuristic_three(idx, goal, dimension) + currcost), idx))
                
                hq.heappush(puzzlemap, (priority, currentstate))
                
                # else:
                #     # Debug Statement #6
                #     print("Error [bestfirstsearch(puzzle, goal, dimension, type)]: Invalid type {of heuristic}!")
                #     return None
                
                prevstates[str(currentstate)] = curr
        # if idx >= max:
        #         # Result Statement / Debug Statement #8
        #         print("Error [a_asterisksearch(puzzle, goal, dimension, type)]: Max number of moves reached!")
        #         return None
        
    # Return none if no solution path found
    return None
    
    # for idx in range(dimension + 1):
    #     moves.append(move(puzzle, search(puzzle, 0, dimension)[0], search(puzzle, 0, dimension)[1], idx))
        
    # # Remove invalid / None values
    # moves = [x for x in moves if x != None]
    
    # # If heuristic #1 is used for this Best First Search (# of Misplaced Tiles)
    # if type == 1:
    #     # List of moves and heuristic #1 values
    #     heuristic1 = []
    #     for idx in moves:
    #         heuristic1.append((idx, heuristic_one(idx, goal, dimension)))
            
    #     # Sort list of moves by heuristic #1 values
    #     heuristic1.sort(key=lambda x: x[1])
        
    #     return heuristic1[0][0]
    
    # # If heuristic #2 is used for this Best First Search (Manhattan / City-Block Distance)
    # if type == 2:
    #     # List of moves and heuristic #2 values
    #     heuristic2 = []
    #     for idx in moves:
    #         heuristic2.append((idx, heuristic_two(idx, goal, dimension)))

    #     # Sort list of moves by heuristic #2 values
    #     heuristic2.sort(key=lambda x: x[1])

    #     return heuristic2[0][0]
    
    # # If heuristic #3 is used for this Best First Search (Sum of Euclidean Distances)
    # if type == 3:
    #     # List of moves and heuristic #3 values
    #     heuristic3 = []
    #     for idx in moves:
    #         heuristic3.append((idx, heuristic_three(idx, goal, dimension)))
            
    #     # Sort list of moves by heuristic #3 values
    #     heuristic3.sort(key=lambda x: x[1])
        
    #     return heuristic3[0][0]
    
    # # If an invalid heuristic type is used
    # else:
    #     # Debug Statement #4
    #     print("Error [bestfirstsearch(puzzle, goal, dimension, type)]: Invalid type {of heuristic}!")
    #     return None

def program(dimension, max):
    if dimension == 3:
        for iteration1 in range(5):
            #goal = [[1,2,3],[4,5,6],[7,8,0]]
            goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            print("The " + str(iteration1 + 1) + " Iteration 8-puzzle is:")
            for iteration in range(3):
                print(thepuzzle[iteration], thepuzzle[iteration+1], thepuzzle[iteration+2])
            
            print("The goal state of the 8-puzzle is:")
            for iteration in range(3):
                print(goal[iteration], goal[iteration+1], goal[iteration+2])
                        
            print(str(dimension) + "x" + str(dimension) + " Puzzle Best First & A* Searches: ")
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("Best First Search using Heuristic #1 (Misplaced Tiles):")
            # print("zzzzzzzz" + str(thepuzzle))
            # print("zzzzzzzz" + str(goal))
            bestresult1 = bestfirstsearch(thepuzzle, goal, dimension, 1, max)
            #print(bestresult1)
            results(bestresult1, 1, 1)
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("Best First Search using Heuristic #2 (Manhattan Distance):")
            bestresult2 = bestfirstsearch(thepuzzle, goal, dimension, 2, max)
            #print(bestresult2)
            results(bestresult2, 1, 2)
            # Debug Print
            #print("HEURISTIC COST TEST BEFORE #3")
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("Best First Search using Heuristic #3 (Sum of Euclidean Distances):")
            bestresult3 = bestfirstsearch(thepuzzle, goal, dimension, 3, max)
            #print(bestresult3)
            results(bestresult3, 1, 3)
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("A* Search using Heuristic #1 (Misplaced Tiles):")
            aresult1 = a_asterisksearch(thepuzzle, goal, dimension, 1, max)
            #print(aresult1)
            results(aresult1, 2, 1)
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("A* Search using Heuristic #2 (Manhattan Distance):")
            aresult2 = a_asterisksearch(thepuzzle, goal, dimension, 2, max)
            #print(aresult2)
            results(aresult2, 2, 2)
            
            thepuzzle = []
            thepuzzle = create8puzzle(iteration1)
            
            print("A* Search using Heuristic #3 (Sum of Euclidean Distances):")
            aresult3 = a_asterisksearch(thepuzzle, goal, dimension, 3, max)
            #print(aresult3)
            results(aresult3, 2, 3)
            
            # thepuzzle = []
            # thepuzzle = create8puzzle(iteration1)
    
    if dimension == 4:
        for iteration2 in range(5):
            goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            print("The " + str(iteration2 + 1) + " Iteration 15-puzzle is:")
            for iteration in range(4):
                print(thepuzzle[iteration], thepuzzle[iteration+1], thepuzzle[iteration+2], thepuzzle[iteration+3])
            
            print("The goal state of the 15-puzzle is:")
            for iteration in range(4):
                print(goal[iteration], goal[iteration+1], goal[iteration+2], goal[iteration+3])
                        
            print(str(dimension) + "x" + str(dimension) + " Puzzle Best First & A* Searches: ")
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("Best First Search using Heuristic #1 (Misplaced Tiles):")
            bestresult1 = bestfirstsearch(thepuzzle, goal, dimension, 1, max)
            #print(bestresult1)
            results(bestresult1, 1, 1)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("Best First Search using Heuristic #2 (Manhattan Distance):")
            bestresult2 = bestfirstsearch(thepuzzle, goal, dimension, 2, max)
            #print(bestresult2)
            results(bestresult2, 1, 2)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("Best First Search using Heuristic #3 (Sum of Euclidean Distances):")
            bestresult3 = bestfirstsearch(thepuzzle, goal, dimension, 3, max)
            #print(bestresult3)
            results(bestresult3, 1, 3)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("A* Search using Heuristic #1 (Misplaced Tiles):")
            aresult1 = a_asterisksearch(thepuzzle, goal, dimension, 1, max)
            #print(aresult1)
            results(aresult1, 2, 1)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("A* Search using Heuristic #2 (Manhattan Distance):")
            aresult2 = a_asterisksearch(thepuzzle, goal, dimension, 2, max)
            #print(aresult2)
            results(aresult2, 2, 2)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)
            
            print("A* Search using Heuristic #3 (Sum of Euclidean Distances):")
            aresult3 = a_asterisksearch(thepuzzle, goal, dimension, 3, max)
            #print(aresult3)
            results(aresult3, 2, 3)
            
            thepuzzle = []
            thepuzzle = create15puzzle(iteration2)

if __name__ == '__main__':

    max = 9999

    program(3, max)
    program(4, max)
    
