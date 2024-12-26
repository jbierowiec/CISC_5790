import numpy as np
import matplotlib.pyplot as plt

def dtw_distance(X, Y):
    n, m = len(X), len(Y)
    dtw = np.zeros((n+1, m+1))
    
    # Initialize borders with infinity, except for the starting point (0, 0)
    for i in range(1, n+1):
        dtw[i, 0] = np.inf
    for i in range(1, m+1):
        dtw[0, i] = np.inf
    dtw[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(X[i-1] - Y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match

    # Backtrack to find the optimal path
    i, j = n, m
    path = [(i, j)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            if dtw[i-1, j] == min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]):
                i -= 1
            elif dtw[i, j-1] == min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
        
    path.reverse()
    return dtw[n, m], path

def dtw_distance_and_path(X, Y):
    n, m = len(X), len(Y)
    dtw = np.zeros((n+1, m+1))
    
    # Initialize borders with infinity, except for the starting point (0, 0)
    for i in range(1, n+1):
        dtw[i, 0] = np.inf
    for i in range(1, m+1):
        dtw[0, i] = np.inf
    dtw[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(X[i-1] - Y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match

    # Backtrack to find the optimal path
    i, j = n, m
    path = [(i, j)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            if dtw[i-1, j] == min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]):
                i -= 1
            elif dtw[i, j-1] == min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
        
    path.reverse()
    return dtw, path

def plot_dtw_and_path(X, Y):
    dtw, path = dtw_distance_and_path(X, Y)
    
    # Plotting the DTW matrix
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # Distance matrix
    ax[0].imshow(dtw[1:, 1:].T, origin='lower', cmap='viridis')  # Transpose to switch axes
    ax[0].set_title('DTW Distance Matrix')
    ax[0].set_xlabel('X Sequence')
    ax[0].set_ylabel('Y Sequence')
    for i in range(len(X)):
        for j in range(len(Y)):
            ax[0].text(i, j, f'{dtw[i+1, j+1]:.0f}', ha='center', va='center', color='white')

    # Path plot
    ax[1].imshow(dtw[1:, 1:].T, origin='lower', cmap='viridis')  # Transpose to switch axes
    ax[1].set_title('Optimal Warping Path')
    ax[1].set_xlabel('X Sequence')
    ax[1].set_ylabel('Y Sequence')
    path_x, path_y = zip(*path)
    ax[1].plot(np.array(path_x)-1, np.array(path_y)-1, color='red')  # adjust indices for display and switch x, y for plotting
    
    plt.tight_layout()
    plt.show()

X = [32, 36, 27, 37, 35, 40, 34, 33, 25, 29]
Y = [31, 32, 32, 30, 37, 39, 29, 34, 25, 26]

plot_dtw_and_path(X, Y)

distance, optimal_path = dtw_distance(X, Y)
print("DTW Distance:", distance)
print("Optimal Path:", optimal_path)
