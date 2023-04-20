import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans as torch_KMeans

_USE_MPS = True if torch.backends.mps.is_built() else False
print("Use MPS? ", _USE_MPS)

# Flatten the images in the dataset to a 1D array
def flatten_images(images):
    return images.view(images.size(0), -1).numpy()

# Extract features from the DataLoader
def extract_features(trainloader):
    all_features = []
    for data in trainloader:
        images, labels = data
        flattened_images = flatten_images(images)
        all_features.extend(flattened_images)

    all_features = np.array(all_features)
    return all_features

def get_centroids(n_clusters, trainloader, device='', save=False, vis=False):
    all_features = extract_features(trainloader)
    
    if device=='cpu':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(all_features)
        centroids = kmeans.cluster_centers_ # (K, D) array of K centroids, each of dimension D=784=28*28
    elif _USE_MPS:
        kmeans = torch_KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
        kmeans.fit_predict(torch.from_numpy(all_features).to(device))
        centroids = kmeans.centroids.cpu().numpy()
    
    if vis:
        print("Visualize the centroids...")
        # randomly pick 3 centroids and visualize in subplot
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        for i in range(3):
            ex = centroids[np.random.randint(0, n_clusters)]
            ax[i].imshow(ex.reshape(28, 28), cmap='gray')
        plt.show()
    
    # save centroids as pt file
    if save:
        # create folder if not exist
        import os
        DIR = './centroids'
        MODE = 'torch' if _USE_MPS else 'sklearn'
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        SAVENAME = f'{DIR}/{MODE}-k={n_clusters}.pt'
        torch.save(centroids, SAVENAME)
        print(f"Centroids saved to {SAVENAME} (shape: {centroids.shape})")
    
    return kmeans, centroids
