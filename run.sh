#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate 499

# sweep ratio values
for r in 10 25 50 75; do 
    python main.py \
        --ratio $r \
        --n_clusters 20 \
        --rank_type random \
        --prune_type common \
        --dataset mnist
done

# sweep n_clusters values
for k in 10 20 30 40 50; do
    python main.py \
        --ratio 25 \
        --n_clusters $k \
        --rank_type random \
        --prune_type common \
        --dataset mnist
done