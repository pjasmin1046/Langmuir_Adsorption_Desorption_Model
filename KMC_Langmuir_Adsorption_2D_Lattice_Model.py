# The following is a Kinetic Monte-Carlo Simulation of gas adsorption/desorption on a 2D hypothetical substrate.
# The algorithm is modeled after and adapted from https://www.youtube.com/watch?v=KfwhMDZVS1U implemented in Python
# instead of Matlab. This simulation models the simplest case of monolayer adsorption where all sites are energetically
# equivalent and there is no diffusion present.

# Adsorption/Desorption events are considered for a 10x10 lattice on which 6 possible events can occur:
#   E1 = Adsorption
#   E2 - E6 = Desorption with 0 - 4 near neighboring atoms

# At each iteration the total number of available locations, rates, and probabilities for each event are tabulated
# and random number generation is used to select which event occurs and at what location. Theta is defined as the
# % surface coverage and is tabulated at each iteration. Graphical output of theta vs t(sec) is provided along with the
# probability of each of the 6 possible events (P1 - P6) vs time throughout the simulation.

# The optional debug parameter prints intermediate results to console throughout the simulation.


import numpy as np
import random
import math
import pandas as pd
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def AdsorptionPeriodicity(LATTICE, latticeLength, action_x, action_y):
    # Maintain periodicity in x direction
    if action_x == 2:
        LATTICE[latticeLength + 1, action_y] = 1

    elif action_x == latticeLength + 1:
        LATTICE[1, action_y] = 1

    # Maintain periodicity in y direction
    if action_y == 2:
        LATTICE[action_x, latticeLength + 1] = 1

    elif action_y == latticeLength + 1:
        LATTICE[action_x, 1] = 1

def DesorptionPeriodicity(LATTICE, latticeLength, action_x, action_y):
    # Maintain periodicity in x direction
    if action_x == 2:
        LATTICE[latticeLength + 1, action_y] = 0

    elif action_x == latticeLength + 1:
        LATTICE[1, action_y] = 0

    # Maintain periodicity in y direction
    if action_y == 2:
        LATTICE[action_x, latticeLength + 1] = 0

    elif action_y == latticeLength + 1:
        LATTICE[action_x, 1] = 0

def findTheta(LATTICE, latticeLength, debug = False):
    # Creating a subarray by slicing from index 2 to latticeLength + 1
    subarray = LATTICE[1:latticeLength+1, 1:latticeLength+1]

    total = subarray.size
    occupied = np.count_nonzero(subarray == 1)
    theta = (occupied / total)

    if debug:
        print(f'Total: {total}\nOccupied: {occupied}')

    return theta

def Langmuir_No_Diffusion(numRuns, latticeLength, debug = False):
    # Rate Constants
    rA = 1  # s^-1
    rD = 2  # s^-1
    alpha = 1  # factor denoting atomic bond strength & near neighbor interactions

    # Desorption rates accounting for influence from neighbors
    r_E2 = rD * (alpha ** 0)
    r_E3 = rD * (alpha ** 1)
    r_E4 = rD * (alpha ** 2)
    r_E5 = rD * (alpha ** 3)
    r_E6 = rD * (alpha ** 4)

    rMatrix = [rA, r_E2, r_E3, r_E4, r_E5, r_E6]

    # Define lattice and pertinent lists/counter variables
    LATTICE = np.zeros((latticeLength + 2, latticeLength + 2))  # maintains periodicity in the X-axis
    P = np.zeros((6, 1))
    tList = []
    thetaList = []

    # Counts number of each event which occurs
    E1counter = 0
    E2counter = 0
    E3counter = 0
    E4counter = 0
    E5counter = 0
    E6counter = 0

    # Stores each event probability at each time step (for graphing purposes)
    P1List = []
    P2List = []
    P3List = []
    P4List = []
    P5List = []
    P6List = []

    if debug:
        print("---- FUNCTION START ----\n")
        print("\n-- Constants & Rates --")
        print(f'rA: {rA}\nrD: {rD}\nAlpha: {alpha}\nRate Matrix: {rMatrix}\n')
        print(f'Initial Lattice:\n {LATTICE}\n')
        print(f'Initial P Matrix:\n {P}\n')

    # KMC Loop Initialization
    t = 0 # Time (sec)

    for i in range(1, numRuns):
        # Lists to store all locations at which the particular event can occur at a given iteration
        E1 = []
        E2 = []
        E3 = []
        E4 = []
        E5 = []
        E6 = []

        # Stores number of locations at which each event can occur (6x1 matrix)
        count = np.zeros((6, 1))

        if debug:
            print(f'\n-- Loop Iteration {i} @ t = {t} -- \n')
            print(f'Initial E1 @ iteration {i}: {E1}\nInitial E2 @ iteration {i}: {E2}\nInitial E3 @ iteration {i}: {E3}\nInitial E4 @ iteration {i}: {E4}\nInitial E5 @ iteration {i}: {E5}\nInitial E6 @ iteration {i}: {E6}\nInitial Count Matrix @ iteration {i}:\n {count}\n')

        # The following populates E1 - E6 Lists and the count matrix
        for x in range(1, latticeLength + 1):
            for y in range(1, latticeLength + 1):

                # Determines locations at which E1 can occur
                if LATTICE[x, y] == 0:
                    count[0, 0] += 1
                    E1.append([x, y])

                else:
                    # If E1 cannot occur, determine how many neighbors are near the desorbing atom to determine which desorption event occurs
                    neighbors = LATTICE[x - 1, y] + LATTICE[x + 1, y] + LATTICE[x, y - 1] + LATTICE[x, y + 1]
                    # if debug:
                    #     print("ELSE")
                    #     print(f'# neighbors @ ({x}, {y}) in iteration {i}: {neighbors}')

                    if neighbors == 0:
                        count[1, 0] += 1
                        E2.append([x, y])

                    elif neighbors == 1:
                        count[2, 0] += 1
                        E3.append([x, y])

                    elif neighbors == 2:
                        count[3, 0] += 1
                        E4.append([x, y])

                    elif neighbors == 3:
                        count[4, 0] += 1
                        E5.append([x, y])

                    elif neighbors == 4:
                        count[5, 0] += 1
                        E6.append([x, y])

        # The following two lists are used to find the coordinates at which the randomly selected desorption event will occur
        lengthList = [E1, E2, E3, E4, E5, E6]
        totalE = E1 + E2 + E3 + E4 + E5 + E6

        if debug:
            print(f'\nLength List @ iteration {i}: {lengthList}\n')
            print(f'totalE (len = {len(totalE)} @ iteration {i}: {totalE}\n')
            # print(f'Final E1 (len = {len(E1)}):\n {E1}\nFinal E2 (len = {len(E2)}):\n {E2}\nFinal E3 (len = {len(E3)}):\n {E3}\nFinal E4 (len = {len(E4)}):\n {E4}\nFinal E5 (len = {len(E5)}):\n {E5}\nFinal E6 (len = {len(E6)}):\n {E6}\n')


        # Total Rate Calculation
        R = 0
        for g in range(0, 6):
            R = R + (count[g, 0] * rMatrix[g])

        # Event-wise probability -> P matrix
        P[0, 0] = (count[0, 0] * rMatrix[0]) / R
        # if debug:
        #     print(f'Count: \n{count}\nP: \n{P}\n')

        for j in range(1, 6):
            # if debug:
            #     print(f'@ j = {j}, P[j, 0] = {P[j, 0]}\nCount[j + 1, 0] = {count[j + 1, 0]}\nrMatrix[j + 1] = {rMatrix[j + 1]}\nR = {R}')
            P[j, 0] = (count[j, 0] * rMatrix[j] / R)
            # if debug:
            #     print(f'P[j + 1, 0]: {P[j + 1, 0]}\n')

        # pSum MUST == 1
        pSum = P[0,0] + P[1,0] + P[2,0] + P[3,0] + P[4,0] + P[5,0]

        # Append to probability lists from above (for graphing purposes)
        P1List.append(P[0, 0])
        P2List.append(P[1, 0])
        P3List.append(P[2, 0])
        P4List.append(P[3, 0])
        P5List.append(P[4, 0])
        P6List.append(P[5, 0])

        # Random number to choose event
        rd = random.random()

        if debug:
            print(f'\nR @ iteration {i}: {R}\n')
            print(f'Final Count Matrix @ iteration {i}:\n {count}\n')
            print(f'Rate Matrix @ iteration {i}:\n {rMatrix}\n')
            print(f'Final P Matrix @ iteration {i}:\n {P}\n adding to {pSum}\n')
            print(f'\nRANDOM @ iteration {i}: {rd}\n')

        if np.abs(1 - pSum) > 0.02:
            print(f'-- PROBABILITY SUMMATION WARNING: P matrix adds to {pSum}, not 1 -- \n')



        # Determining Event Occurrence

        # Determine if E1 occurs

        # Evaluates which event occurs based on whether rd is between the probability of events which have already
        # been considered, and the event currently being considered.
        
        baseP = P[0,0]

        if debug:
            print(f'BaseP: {baseP}\n')

        # EVENT 1 OCCURS
        if rd < baseP:
            # Random number to use in E1 location selection
            rd_temp = random.random()

            # temp determines which of the available locations for E1 is used
            temp = math.floor((rd_temp * count[0, 0]) + 1)
            if temp >= count[0, 0]:
                # if debug:
                #     print(f'COUNT [0, 0]: {count[0, 0]}\nUNALTERED TEMP: {temp}')
                temp = count[0, 0] - 1

            if debug:
                print(f'\n- EVENT 1 OCCURS - \n')
                print(f'temp: {temp}\nE1 @ temp: {E1[int(temp)]}')

            # Coordinates of Adsorption
            action_x = E1[int(temp)][0]
            action_y = E1[int(temp)][1]

            if debug:
                print(f'Action_X: {action_x}\nAction_Y: {action_y}\n')

            # Update LATTICE structure and appropriate counter variable accordingly
            LATTICE[action_x, action_y] = 1
            E1counter += 1

            # Maintain periodicity if necessary
            AdsorptionPeriodicity(LATTICE, latticeLength, action_x, action_y)

        else:
            # DETERMINES WHICH DESORPTION EVENT (2-6) OCCURS
            for p in range(1,6):

                if debug:
                    print(f'@ p = {p}, BaseP = {baseP}\nP[p, 0] = {P[p, 0]}\nBaseP + P[p, 0] = {baseP + P[p, 0]}')
                    print(rd)

                # Determines which desorption event occurs by comparing the combined probability of all previously
                # considered events (baseP) to the sum of baseP and the current event probability being considered.
                # This determines which event should occur.
                if ((rd > baseP) and (rd < (baseP + P[p, 0]))):
                    rd_temp = random.random()

                    # temp determines which of the available locations for E[p] is used
                    temp = math.floor((rd_temp * count[p, 0]) + 1)
                    if temp > count[p, 0]:
                        temp = count[p, 0] - 1

                    # loc determines where in the totalE list locations for the selected event begin
                    loc = 0
                    loc += sum(len(lengthList[e]) for e in range(0, p-1)) - 1

                    if debug:
                        print(f'\n- EVENT {p + 1} OCCURS - \n')
                        print(f'temp: {temp}\n')
                        print(f'loc: {loc}\n')
                        print(f'totalE (len = {len(totalE)} @ iteration {i}: {totalE}\n')
                        print(f'E[loc + temp]: {totalE[int(loc + temp)]}\n')

                    # Desorption event coordinates
                    action_x = totalE[int(loc + temp)][0]
                    action_y = totalE[int(loc + temp)][1]

                    if debug:
                        print(f'Action_X: {action_x}\nAction_Y: {action_y}\n')

                    # Update Lattice structure and maintain periodicity if necessary
                    LATTICE[action_x, action_y] = 0
                    DesorptionPeriodicity(LATTICE, latticeLength, action_x, action_y)

                    # Update appropriate counter variable
                    if p + 1 == 2:
                        E2counter += 1

                    elif p + 1 == 3:
                        E3counter += 1

                    elif p + 1 == 4:
                        E4counter += 1

                    elif p + 1 == 5:
                        E5counter += 1

                    elif p + 1 == 6:
                        E6counter += 1

                    # Add current event probability to cummulative probability to avoid multiple events falsely occurring in the same timestep
                    baseP += P[p, 0]
                    if debug:
                        print(f'BaseP updates to {baseP}\n')

                else:
                    # If considered event does not occur, add its probability to cumulative probability and iterate to next possible event
                    baseP += P[p, 0]
                    if debug:
                        print(f'BaseP updates to {baseP}\n')

        #
        theta = findTheta(LATTICE, latticeLength, debug = True)
        tList.append(t)
        thetaList.append(theta)
        t += 1 / R

        if debug:
            print(f'\nTheta: {theta}')
            print(f'Time updates by {1/R} to {t} sec\n')
            print(f'LATTICE @ end of iteration {i}:\n {LATTICE}\n')

    # Plotly graphing procedure

    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Theta vs Time', 'Probability vs Time'))

    # Add trace for Theta vs Time
    fig.add_trace(go.Scatter(x=tList, y=thetaList, mode='lines', name='Theta'), row=1, col=1)

    # Add traces for each probability line

    fig.add_trace(go.Scatter(x=tList, y=P1List, mode='lines', name=f'P{1}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tList, y=P2List, mode='lines', name=f'P{2}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tList, y=P3List, mode='lines', name=f'P{3}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tList, y=P4List, mode='lines', name=f'P{4}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tList, y=P5List, mode='lines', name=f'P{5}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tList, y=P6List, mode='lines', name=f'P{6}'), row=2, col=1)

    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis_title='Time (sec)',
    )

    # Show the plot
    fig.show()

    # Print final counts of events that occurred.
    print(f'E1 Count: {E1counter}\nE2 Count: {E2counter}\nE3 Count: {E3counter}\nE4 Count: {E4counter}\nE5 Count: {E5counter}\nE6 Count: {E6counter}\nTotal: {E1counter + E2counter + E3counter + E4counter + E5counter + E6counter}\n')




Langmuir_No_Diffusion(20000, 10, debug = True)






