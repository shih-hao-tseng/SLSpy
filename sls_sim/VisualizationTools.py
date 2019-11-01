import matplotlib.pyplot as plt
import numpy as np

def matrix_list_multiplication (matrix_A=None, list_B=[]):
    AB = []
    for t in range(len(list_B)):
        AB.append(np.dot(matrix_A,list_B[t]))
    return AB

def plot_graph(adjMtx, nodeCoords, colour):
    '''
    Visualizes graph
      adjMtx     : adjacency matrix
      nodeCoords : x,y coordinates of each node (in order)
      colour     : colour of nodes, either a letter or a RGB coordinate
    '''
    # TODO

def plot_graph_animation(adjMtx, nodeCoords, slsParams, x, Bu, waitTime, logScale):
    '''
    Plots topology of graph and animates states values at nodes
    Inputs
      adjMtx     : adjacency matrix
      nodeCoords : x,y coordinates of each node (in order)
      slsParams  : SLSParams containing parameters
      x, Bu      : state and actuation values at nodes
      waitTime   : amount of time to wait between steps
      logScale   : whether to use the same scale as heat map plotter
    '''
    # TODO

def plot_heat_map (x=None, Bu=None, myTitle='title'):
    '''
    Plots log-based heat map for x, u
    Inputs
    x, Bu    : state and actuation values at nodes
    myTitle  : overall title of the heat maps
    '''

    plt.figure()
    plt.suptitle(myTitle)

    logmin = -4
    logmax = 0

    plt_x  = np.concatenate(x, axis=1)
    plt_Bu = np.concatenate(Bu,axis=1)

    plt_x  = np.log10(np.absolute(plt_x))
    plt_Bu = np.log10(np.absolute(plt_Bu))

    # cut at the min
    plt_x  = np.clip(plt_x,  logmin - 1, logmax + 1)
    plt_Bu = np.clip(plt_Bu, logmin - 1, logmax + 1)

    if Bu is None:  # or pure zero?
        # don't subplot; plot only x
        plt.pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        plt.colorbar()
        plt.title('log10(|x|)')
        plt.xlabel('Time')
        plt.ylabel('Space')
        
    else:
        plt.subplot(1,2,1)
        plt.pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        plt.colorbar()
        plt.title('log10(|x|)')
        plt.xlabel('Time')
        plt.ylabel('Space')

        plt.subplot(1,2,2)
        plt.pcolor(
            plt_Bu,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        plt.colorbar()
        plt.title('log10(|u|)')
        plt.xlabel('Time')

    plt.show()

def plot_line_chart(list_x=[], list_y=[], title='title', xlabel='xlabel', ylabel='ylabel'):
    plt.figure()
    plt.plot(list_x,list_y,'o-')
    plt.gca().invert_xaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_time_trajectory(x=None, Bu=None, xDes=None):
    '''
    Plots time trajectories and errors (useful for trajectory tracking)
    Inputs
       x, Bu : state and actuation values at nodes
       xDes  : desired trajectory
    '''
    # TODO

def plot_vertex(node, nodeCoords, colour):
    '''
    Adds colour & label to specified vertex / node
      node       : the node we're plotting
      nodeCoords : x,y coordinates of each node (in order)
      colour     : colour of node, either a letter or a RGB coordinate
    '''
    # TODO