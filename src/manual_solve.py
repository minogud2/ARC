#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_b94a9452(x):
    """
    1. Description: The aim is to find a square, flip the colours and reduce the matrix to the dimension of
    the square. Method: a) Find coloured shape in matrix i.e. all non-zero values, b) reduce the dimensions
    of the matrix to the size of the shape i.e. delete all non-zero values, c) reverse the colors of the shape
    d) return new shape.

    2. Solved Correctly? Yes, all training and test grids are solved correctly. See output.
    """
    i, shape_size = 0,0    # find size to reshape the vector
    while i <len(x) and (shape_size == 0):
        if np.count_nonzero(x[i]) > shape_size:
            shape_size = np.count_nonzero(x[i])
        i+=1
    # Remove all zeros and reshape
    x = x[np.nonzero(x)].reshape(shape_size,shape_size)
    # identify unique values and swap the order.
    unique_values =np.unique(x)
    x = np.where(x == unique_values[0], unique_values[1], unique_values[0])
    return x

def solve_6c434453(x):
    """
    1. Description: The aim is to find blue square boxes with an empty centre and convert to a red cross.
    Method: a) create two vectors, one for the square and another for the red cross. b) flatten the matrix and
    find each value of blue (i.e. 1) in the matrix. Loop through the array and see if it matches the correct index
    values that would correspond to the blue square. c) If they correspond, replace with the red cross.
    d) Reshape the flattened array to the original grid shape.

    2. Solved Correctly? Yes, all training and test grids are solved correctly. See output.
    """
    shp = x.shape # To reshape at the end.
    x = x.flatten() # convert to 1d array.
    sq_shape = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]) # Square shape
    plus_shape = np.array([0,2,0,2,2,2,0,2,0]) # Red Triangle Shape.
    for i in range(0, len(x)):
        shape_index = np.array([0,1,2,10,11,12,20,21,22]) # index of square shape across a flattened vector
        if ((shape_index[-1] + i) < len(x)): # Don't exceed the bounds
            if x[i] == 1: # Check each element in blue to see if it is a square shape.
                shape_index = shape_index + i
                if (x[shape_index] == sq_shape).all(): # checks new index of square shape + i.
                    x[shape_index] = plus_shape # assign the plus shape.
    return x.reshape(shp)

def solve_1bfc4729(x):
    """
    1. Description: Aim is to find two coloured squares in each half of the grid and then flood fill the horizonal line
    across from the square to the same colour as the square. Then fill the perimiter of each half with the same color of
    each square found in each half of the grid space.

    2. Solved Correctly? Yes, all training and test grids are solved correctly. See output.
    """
    nums = x[x>0] # create an array from the 2 values above 0 in either half of the matrix.
    x[0], x[-1] = nums[0], nums[1] # fill first array with num[0] and last array with num[1]
    x[np.where(x == nums[0])[0]] = nums[0] # fill array with num[0] at index where num[0] is located
    x[np.where(x == nums[1])[0]] = nums[1] # fill array with num[1] at index where num[1] is located
    x[:len(x)//2,0], x[:len(x)//2,-1] = nums[0], nums[0] # fill half the perimeter with num[0]
    x[len(x)//2:,0], x[len(x)//2:,-1] = nums[1], nums[1] # fill remaining perimeter with num[1]
    return x

""" Summary
    All solutions were hard coded. Only numpy was imported to resolve each of the problems identified.
    All solutions identified were based on my own inferences for solving the problem and would likely
    fail when applied to other ARC tests. This in itself shows the complexity of designing algorithms
    to resolve such problems.

    There are commonalities between function 1 and 2 as there is a need  to search for a square shape
    within the grid, but solutions applied were very different as the search space was more cluttered
    in function 2, whereas in function 1, the dimensions could be reduced immediately. In all examples
    the key shapes for transformation could be easily identified i.e. finding the square.
"""

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)

def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()
