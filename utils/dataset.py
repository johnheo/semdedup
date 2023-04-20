#for creating custom dataset
from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.keys = list(data_dict.keys())
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # Get the image data and label for the current item
        key = self.keys[index]
        img = self.data_dict[key]['images']
        label = self.data_dict[key]['labels']

        if self.transform is not None:
            img = self.transform(img)

        return img, label