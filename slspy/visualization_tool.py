from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np

def matrix_list_multiplication (matrix_A=None, list_B=[]):
    AB = []
    for t in range(len(list_B)):
        AB.append(np.dot(matrix_A,list_B[t]))
    return AB

def keep_showing_figures ():
    show()

def plot_heat_map (x=None, Bu=None, myTitle='title', outputFileName=None):
    '''
    Plots log-based heat map for x, u
    Inputs
    x, Bu    : state and actuation values at nodes
    myTitle  : overall title of the heat maps
    '''

    figure()
    suptitle(myTitle)

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
        pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        title('log10(|x|)')
        xlabel('Time')
        ylabel('Space')
        
    else:
        subplot(1,2,1)
        pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        title('log10(|x|)')
        xlabel('Time')
        ylabel('Space')

        subplot(1,2,2)
        pcolor(
            plt_Bu,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        title('log10(|u|)')
        xlabel('Time')

    if outputFileName is not None:
        # output as csv file
        np.savetxt(outputFileName+'-x.csv', plt_x.round(2).T, fmt='%.1f')
        np.savetxt(outputFileName+'-Bu.csv', plt_Bu.round(2).T, fmt='%.1f')

    show(block=False)

def plot_line_chart(list_x=[], list_y=[], title='title', xlabel='xlabel', ylabel='ylabel',line_format='o-', invert_x=False):
    figure()
    plot(list_x,list_y,line_format)
    if invert_x:
        gca().invert_xaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    show(block=False)

def plot_time_trajectory(x=None, Bu=None, xDes=None):
    '''
    Plots time trajectories and errors (useful for trajectory tracking)
    Inputs
       x, Bu : state and actuation values at nodes
       xDes  : desired trajectory
    '''
    figure()

    # nothing to plot
    if x is None:
        return

    TMax = len(x)
    if TMax < 1:
        return

    Nx = x[0].shape[0]

    def get_max_from_list (value, list_x):
        for tmp in list_x:
            val = np.max(tmp)
            if value < val:
                value = val

    def get_min_from_list (value, list_x):
        for tmp in list_x:
            val = np.min(tmp)
            if value > val:
                value = val

    #maxy = max([max(vec(x)) max(vec(Bu)) max(vec(xDes))]) + 2;
    maxy = np.max(x[0])
    get_max_from_list (maxy, x)
    get_max_from_list (maxy, Bu)
    get_max_from_list (maxy, xDes)
    maxy += 2

    #miny = min([min(vec(x)) min(vec(Bu)) max(vec(xDes))]) - 2;
    miny = np.min(x[0])
    get_min_from_list (miny, x)
    get_min_from_list (miny, Bu)
    get_min_from_list (miny, xDes)
    miny -= 2

    err = []
    maxe = None
    mine = None
    for i in range(len(xDes)):
        if i < len(x):
            val = np.absolute(xDes[i]-x[i])
            err.append(val)
            maxv = np.max(val)
            if maxe is None:
                maxe = maxv
            elif maxe < maxv:
                maxe = maxv
            minv = np.min(val)
            if mine is None:
                mine = minv
            elif mine > minv:
                mine = minv

    TMax_series = np.arange(1,TMax + 1)

    for node in range(Nx):
        subplot(Nx, 2, node * 2 + 1)
        step(TMax_series, x[node])
        step(TMax_series, xDes[node])
        step(TMax_series, Bu[node])
        xticks([])
        ylabel('%d' % (node+1))
        ylim((miny,maxy))

        subplot(Nx, 2, node * 2 + 2)
        step(TMax_series, err[node])
        xticks([])
        ylabel('%d' % (node+1))
        if mine != maxe:
            ylim((mine,maxe))
    
    subplot(Nx, 2, Nx * 2 - 1)
    legend(['x', 'xDes', 'u'])
    xlabel('time step')

    subplot(Nx, 2, Nx * 2)
    legend('error')
    xlabel('time step')

    show(block=False)

def plot_vertex(node, nodeCoords, colour):
    '''
    Adds colour & label to specified vertex / node
      node       : the node we're plotting
      nodeCoords : x,y coordinates of each node (in order)
      colour     : colour of node, either a letter or a RGB coordinate
    '''
    txtOffset = 0.08
    coord_x = nodeCoords[node][0]
    coord_y = nodeCoords[node][1]

    plot(
        coord_x, coord_y, 'o',
        markersize=10,
        markerfacecolor=colour,
        markeredgecolor='k'
    )
    text(
        coord_x + txtOffset, coord_y + txtOffset,
        '%d' % node
    )

def plot_graph(adjMtx, nodeCoords, colour):
    '''
    Visualizes graph
      adjMtx     : adjacency matrix
      nodeCoords : x,y coordinates of each node (in order)
      colour     : colour of nodes, either a letter or a RGB coordinate
    '''
    for ia,ib in np.ndindex(adjMtx.shape):
        if adjMtx[ia,ib] > 0:
            plot(
                [nodeCoords[ia][0],nodeCoords[ib][0]],
                [nodeCoords[ia][1],nodeCoords[ib][1]],
                '-ok'
            )

    numNodes = len(nodeCoords)
    
    minCoords = None
    maxCoords = None

    for node in range(numNodes):
        plot_vertex(node, nodeCoords, colour)
        minCoords = nodeCoords[node] if minCoords is None else np.minimum(minCoords,nodeCoords[node])
        maxCoords = nodeCoords[node] if maxCoords is None else np.maximum(maxCoords,nodeCoords[node])

    # dynamic axis limits
    xlowerlim = minCoords[0] - 0.5
    xupperlim = maxCoords[0] + 0.5
    ylowerlim = minCoords[1] - 0.5
    yupperlim = maxCoords[1] + 0.5

    axis([xlowerlim,xupperlim,ylowerlim,yupperlim])

    show(block=False)

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