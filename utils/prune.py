import random
import numpy as np

def data_prune(clustered_dataset, centroids, rnd=False, diverse=False,ratio=0.5):
    #create dataset with 50% of data
    data_dict_50 = {}
    index = 0
    for i in range(len(centroids)):
        num_items = len(clustered_dataset[i])
        if num_items < 2:
            continue
        num_selected = np.ceil(len(clustered_dataset[i]) * ratio)
        print(f"num_items: {num_items} ==> num_selected: {num_selected}")
        if rnd:
            #select random indices
            selected_indices = random.sample(range(num_items), int(num_selected))
        else:
            flatten = lambda x : x.view(x.size(0), -1).numpy()
            # similarity of centroid to each image
            sim = []
            for j in range(num_items):
                # sim.append(np.linalg.norm(centroids[i] - flatten(clustered_dataset[i][j][0])))
                val = flatten(clustered_dataset[i][j][0]) @ centroids[i]
                # get the value of array
                sim.append(val.item())

            assert len(sim) == num_items, "similarity list length is not equal to number of items"
            sorted_indices = np.argsort(sim) # sort similarity, ascending
            if diverse: # retain diverse samples
                selected_indices = sorted_indices[:int(num_selected)]
            else: # retain common samples
                selected_indices = sorted_indices[-int(num_selected):]


        for j in selected_indices:
            data_dict_50[index] = {'images': clustered_dataset[i][j][0], 'labels': clustered_dataset[i][j][1]}
            index = index+1

    print('new train data length', len(data_dict_50))
    return data_dict_50