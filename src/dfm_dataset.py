import numpy as np
import pandas as pd
import torch.utils.data

class Market_User(torch.utils.data.Dataset):
    def __init__(self, dataset_path, sep=':', header=0, train=True):
        data = pd.read_csv(dataset_path, sep=sep, low_memory=False, header=header).to_numpy()
        if train:
            self.items = data[:, 1:6].astype(np.int) % 100
            self.targets = self.__preprocess_target(data[:, 6].astype(np.float32))
            self.field_dims = np.max(self.items, axis=0) + 1
            self.user_field_idx = np.array((0, ), dtype=np.long)
            self.item_field_idx = np.array((1,), dtype=np.long)
            self.visited = None
        else:
            items = np.repeat(data[:, 1:5], repeats=75, axis=0)
            markets = np.tile(np.arange(0, 75), int(len(items) / 75))
            self.items = np.hstack((items, np.atleast_2d(markets).T))
            self.targets = np.repeat(data[:, 20] % 100, repeats=75, axis=0)
            self.visited = np.repeat(data[:, 5:20], repeats=75, axis=0)
            self.field_dims = np.max(self.items, axis=0) + 1
            self.user_field_idx = np.array((0,), dtype=np.long)
            self.item_field_idx = np.array((1,), dtype=np.long)


    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        if self.visited is None:
            return self.items[index], self.targets[index]
        else:
            return self.items[index], self.targets[index], self.visited[index]

    def __preprocess_target(self, target):
        target[target == 1] = 1
        target[target == 2] = 1
        target[target == 3] = 1
        target[target == 4] = 1
        target[target == 5] = 1
        return target