# Import libraries
import time
import os
import numpy as np
import argparse

import torch
from torchvision import datasets, transforms
print('torch ver: ', torch.__version__)

from utils.prune import data_prune
from utils.cluster import get_centroids
from utils.dataset import DictDataset
from utils.engine import *
from utils.models import myMLP

DEVICE = 'mps' if torch.backends.mps.is_built() else 'cpu'

parser = argparse.ArgumentParser()
# evaluation set-up params
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')

parser.add_argument('--dataset', type=str, default='fmnist', choices=['mnist','fmnist'], help="dataset")
parser.add_argument('--n_clusters', type=int, default=20, help="number of centroids to use")
parser.add_argument('--ratio', type=int, default=100, help="percent data to keep")
parser.add_argument('--rank_type', type=str, default='random', choices=['random','cossim'], help="data ranking criteria")
parser.add_argument('--prune_type', type=str, default='common', 
                    choices=['common', # keep the common images
                             'diverse', # keep the diverse images
                             'stratified' # balance data load across ranks
                             ], help="data pruning criteria")
parser.add_argument('--save', action="store_true", help="save loader")
# instantiate
args = parser.parse_args()
print(args)

def get_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]) 
    if args.dataset == 'mnist':
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    elif args.dataset == 'fmnist':
        trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
        testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    image, label = trainset[0] 
    print("image shape: ", image.shape) # torch.Size([1, 28, 28])
    print("# labels: ", len(testset.classes))
    print(testset.class_to_idx)

    # Final sizes are 50000, 10000, 10000
    trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])
    print(f'Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}')
    # Shuffle the data at the start of each epoch (only useful for training set)
    batchsize = 128 # was 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)
    
    return train_loader, val_loader, test_loader

def preprocess(train_loader, centroids, n_clusters):
    # Assign each image to a cluster
    flatten = lambda x : x.view(x.size(0), -1).numpy()
    assignments = []
    for image, target in train_loader:
        feat = flatten(image)
        closest = np.argmax(centroids @ feat.T, axis=0)
        assignments.extend(closest)


    # create a nested list according to assignments
    def fetch(data_loader, indices):
        all_data = list(data_loader.dataset)  # Convert the dataset to a list
        fetched_images = [all_data[i] for i in indices]
        return fetched_images

    # group (image, target) pairs 
    clustered_dataset = [[] for i in range(n_clusters)]
    for i in range(n_clusters):
        idx = np.where(np.array(assignments) == i)[0]
        print(f"cluster {i} has {len(idx)} images")
        imgs = fetch(train_loader, idx)
        clustered_dataset[i] = imgs

    return clustered_dataset

def create_loader(train_loader):
    # check if centroid exists
    DIR = './centroids'
    SAVENAME = f'{DIR}/{args.dataset}/torch-k={args.n_clusters}.pt'
    if os.path.exists(SAVENAME):
        centroids = torch.load(SAVENAME)
    else:
        kmeans, centroids = get_centroids(args.n_clusters, train_loader, 
                                        device=DEVICE, save=True, vis=False,
                                        dataset=args.dataset)
    
    clustered_dataset = preprocess(train_loader, centroids, args.n_clusters)
    
    data_dict_50 = data_prune(clustered_dataset, 
                            centroids,
                            rnd=args.rank_type=='random',
                            diverse=args.prune_type=='diverse',
                            ratio=args.ratio/100)

    custom_dataset = DictDataset(data_dict_50)
    # Create the dataloader for the train dataset
    dedup_train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=128, shuffle=True)
    torch.save(dedup_train_loader, f'./loaders/{args.dataset}/rank={args.rank_type}_prune={args.prune_type}_k={args.n_clusters}@{args.ratio}%.pt')

def evals(dedup_train_loader, test_loader):
    rn = np.random.randint(0, 50)
    torch.manual_seed(rn)
    np.random.seed(rn)
    # shared hparams
    epochs = 30
    lr = 5e-4
    loss_fn = torch.nn.CrossEntropyLoss()

    def driver(loader, mode):
        print(f"Training {mode} model...")
        model = myMLP().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trained_model = train(model, loader, optimizer,
                    loss_fn, device=DEVICE, epochs=epochs)
        
        return trained_model

    _MODE = 'dedup' if args.ratio < 100 else 'dense'
    start = time.time()
    dedup_model = driver(dedup_train_loader, mode=_MODE)
    end = time.time()
    dedup_time = end - start

    dedup_acc = evaluate(dedup_model, test_loader, device=DEVICE) * 100

    return dedup_acc, dedup_time


def main():

    train_loader, val_loader, test_loader = get_loaders()
    print(f"k={args.n_clusters}")

    cfg = f'rank={args.rank_type}_prune={args.prune_type}_k={args.n_clusters}@{args.ratio}%'
    if not os.path.exists(f'./loaders/{args.dataset}/{cfg}.pt'):
        print(f"Creating {cfg} loader...")
        create_loader(train_loader)
    
    print(f"Loading {cfg} loader...")
    dedup_train_loader = torch.load(f'./loaders/{args.dataset}/{cfg}.pt')

    acc, time = evals(dedup_train_loader, test_loader)
    print(cfg)
    print(f"Accuracy: {acc:.2f}%, Time: {time:.2f} sec")






if __name__ == '__main__':
    main()
