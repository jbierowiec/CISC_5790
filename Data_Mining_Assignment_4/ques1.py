import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Function to normalize RGB values
def normalize_rgb(rgb_array):
    return rgb_array / 255

# Function to implement k-means clustering and recolor the image
def k_means_clustering(img, k, initial_centroids, max_iter=50):
    # Normalize the image data
    img_normalized = normalize_rgb(img.reshape((-1, 3)))
    # Initialize KMeans with given centroids and max iterations
    kmeans = KMeans(n_clusters=k, init=np.array(initial_centroids), max_iter=max_iter, n_init=1)
    # Fit KMeans
    kmeans.fit(img_normalized)
    # Predict clusters
    labels = kmeans.predict(img_normalized)
    # Recolor the image based on cluster assignment
    recolored_img = np.array([color_scheme[label] for label in labels]).reshape(img.shape)
    # Calculate and return SSE
    sse = kmeans.inertia_
    return recolored_img, sse

# Color scheme for recoloring the pixels
color_scheme = [
    [60, 179, 113],    # SpringGreen
    [0, 191, 255],     # DeepSkyBlue
    [255, 255, 0],     # Yellow
    [255, 0, 0],       # Red
    [0, 0, 0],         # Black
    [169, 169, 169],   # DarkGray
    [255, 140, 0],     # DarkOrange
    [128, 0, 128],     # Purple
    [255, 192, 203],   # Pink
    [255, 255, 255],   # White
]

# Initial centroids for different values of k
initial_centroids_dict = {
    2: [(0, 0, 0), (0.1, 0.1, 0.1)],
    3: [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2)],
    6: [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5, 0.5)],
    10: [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4),
         (0.5, 0.5, 0.5), (0.6, 0.6, 0.6), (0.7, 0.7, 0.7), (0.8, 0.8, 0.8), (0.9, 0.9, 0.9)]
}

# Load the image
# This code can change based on the programmer's file structure
img = skimage.io.imread('/Users/janbierowiec/DataMining/project_4/image.png')

# Dictionary to store results for different values of k
results = {}

# Run k-means clustering for each k and store the recolored image and final SSE
for k, initial_centroids in initial_centroids_dict.items():
    # Normalize initial centroids to be in the range [0, 1]
    normalized_centroids = [(x/255, y/255, z/255) for x, y, z in initial_centroids]
    recolored_img, sse = k_means_clustering(img, k, normalized_centroids)
    results[k] = {'recolored_img': recolored_img, 'sse': sse}

    # Plotting each of the recolored images
    skimage.io.imshow(recolored_img)
    plt.title(f'k={k}, SSE={sse:.2f}')
    plt.savefig(f'k = {k}')
    plt.close()
