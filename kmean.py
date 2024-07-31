
import numpy as np
import matplotlib.pyplot as plt

def kMeans_init_centroids(X, K):
   
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids
def find_closest_centroid(X,centroids):

    K= centroids.shape[0]
    idx=np.zeros(X.shape[0],dtype=int)

    for i in range(X.shape[0]):
        
        distance = [] 
        for j in range(centroids.shape[0]):

            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)

        idx[i] =np.argmin(distance)

    return idx

def compute_centroids(X, idx, K):
    
    m, n = X.shape
    
    
    centroids = np.zeros((K, n))
    
   
    for i in range(K):
        points = X[idx == i] 
        centroids[i] = np.mean(points, axis = 0)
        
        
        
   
    
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):

    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

  
    for i in range(max_iters):

        print("K-Means iteration %d/%d" % (i, max_iters-1))

        idx = find_closest_centroid(X, centroids)

        if plot_progress:
            
            previous_centroids = centroids

        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx


original_img = plt.imread('img.jpeg')
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
K = 16
max_iters = 10


initial_centroids = kMeans_init_centroids(X_img, K)


centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
# Find the closest centroid of each pixel
idx = find_closest_centroid(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()