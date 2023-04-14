
# kmeans clustering on train data
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]) # transforms.ToTensor() converts the image to a tensor and transforms.Normalize() normalizes the tensor

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

image, label = trainset[0] 
print(image.shape) # torch.Size([1, 28, 28])
print(label) 
trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])
# Final sizes are 50000, 10000, 10000
print(f'Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}')
batchsize = 32
# Shuffle the data at the start of each epoch (only useful for training set)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

# Flatten the images in the dataset to a 1D array
def flatten_images(images):
    return images.view(images.size(0), -1).numpy()

# Extract features from the DataLoader
all_features = []
for data in trainloader:
    images, labels = data
    flattened_images = flatten_images(images)
    all_features.extend(flattened_images)

all_features = np.array(all_features)

n_clusters = 10  # Choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_features)

centroids = kmeans.cluster_centers_ # (K, D) array of K centroids, each of dimension D=784=28*28
ex = kmeans.cluster_centers_[0]
plt.imshow(ex.reshape(28, 28), cmap='gray')

# Assign each sample to its corresponding cluster
cluster_assignments = kmeans.predict(all_features)

# Create a dictionary to store the samples for each cluster
clusters = {i: [] for i in range(n_clusters)}

# Iterate through the dataset and cluster assignments to populate the clusters dictionary
for i, (image, label) in enumerate(trainset):
    cluster = cluster_assignments[i]
    clusters[cluster].append((image, label))

# Choose representative samples for each cluster
deduplicated_data = []
for cluster, samples in clusters.items():
    representative_sample = samples[np.random.randint(len(samples))]  # You can use a different method to choose a representative sample
    deduplicated_data.append(representative_sample)

# Create a new DataLoader with the deduplicated data
deduplicated_dataset = torch.utils.data.TensorDataset(torch.stack([x[0] for x in deduplicated_data]), torch.tensor([x[1] for x in deduplicated_data]))
deduplicated_loader = torch.utils.data.DataLoader(deduplicated_dataset, batch_size=10, shuffle=True, num_workers=2)

