# K-means-algorithm-image-resize
Here we implement the K-means algorithm to reduce the storage size of an image.

<img width="1424" alt="Screenshot 2024-07-31 at 7 14 12 PM" src="https://github.com/user-attachments/assets/8114a534-cbb9-48d7-97aa-fa33b7d8da10">


# K-Means Image Compression

This project demonstrates how the K-means clustering algorithm can be applied to image compression. The key idea is to reduce the number of colors in an image by clustering similar colors together and representing each cluster with a single color, the centroid of that cluster.

## How It Works

1. **Initialization**:
   The process begins by initializing the centroids randomly. The function `kMeans_init_centroids` selects `K` random data points from the image to serve as the initial centroids.

   ```python
   def kMeans_init_centroids(X, K):
       randidx = np.random.permutation(X.shape[0])
       centroids = X[randidx[:K]]
       return centroids
   ```

2. **Finding Closest Centroids**:
   Each pixel in the image is assigned to the nearest centroid. This is done by calculating the Euclidean distance between the pixel's color and each centroid's color. The pixel is assigned to the cluster with the closest centroid. This step is implemented in the `find_closest_centroid` function.

   ```python
   def find_closest_centroid(X, centroids):
       K = centroids.shape[0]
       idx = np.zeros(X.shape[0], dtype=int)
       for i in range(X.shape[0]):
           distance = [np.linalg.norm(X[i] - centroids[j]) for j in range(centroids.shape[0])]
           idx[i] = np.argmin(distance)
       return idx
   ```

3. **Computing New Centroids**:
   After assigning all pixels to the nearest centroid, the algorithm recalculates the position of each centroid. The new position is the mean of all pixels assigned to that centroid. This step is performed in the `compute_centroids` function.

   ```python
   def compute_centroids(X, idx, K):
       m, n = X.shape
       centroids = np.zeros((K, n))
       for i in range(K):
           points = X[idx == i]
           centroids[i] = np.mean(points, axis=0)
       return centroids
   ```

4. **Iterative Optimization**:
   The K-means algorithm iteratively updates the centroids by repeating the process of finding the closest centroids and computing new centroids until the centroids stabilize or a maximum number of iterations is reached. This iterative process is implemented in the `run_kMeans` function.

   ```python
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
   ```

5. **Image Compression**:
   Once the K-means algorithm has converged, each pixel in the image is assigned the color of the nearest centroid. The result is an image with `K` unique colors, effectively compressing the image's color space. The reshaped image can then be displayed and saved.

   ```python
   original_img = plt.imread('tiger.png')
   X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
   K = 16
   max_iters = 10

   initial_centroids = kMeans_init_centroids(X_img, K)
   centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
   idx = find_closest_centroid(X_img, centroids)
   X_recovered = centroids[idx, :]
   X_recovered = np.reshape(X_recovered, original_img.shape)

   fig, ax = plt.subplots(1, 2, figsize=(16, 16))
   ax[0].imshow(original_img)
   ax[0].set_title('Original')
   ax[0].set_axis_off()
   ax[1].imshow(X_recovered)
   ax[1].set_title(f'Compressed with {K} colors')
   ax[1].set_axis_off()
   plt.show()
   ```

## Results

The original image and the compressed image are displayed side by side. The compressed image retains the overall visual appearance of the original but with significantly fewer colors. This technique is particularly useful for reducing the file size of images while maintaining visual quality, which can be beneficial for web applications and storage.

### Notes

- The choice of `K` (the number of clusters) significantly affects the quality of the compressed image. A higher `K` retains more details but results in less compression.
- The algorithm's efficiency can be improved by optimizing the centroid initialization and convergence criteria.

This project is a practical demonstration of K-means clustering for image compression. It highlights the trade-off between compression and image quality and serves as a simple yet powerful example of clustering in image processing.
