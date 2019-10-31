import matplotlib.pyplot as plt
import numpy as np

def Plot_Heat_Map (x=None, Bu=None, myTitle='title'):
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