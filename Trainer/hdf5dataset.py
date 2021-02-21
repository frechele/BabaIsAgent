import h5py
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path

        with h5py.File(self.file_path, 'r') as f:
            self.length = f['feature'].shape[0]

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            feature = f['feature'][index, :, :, :]
            prob = f['prob'][index]
            value = f['val'][index]

        return torch.from_numpy(feature).float(), torch.FloatTensor(prob), torch.FloatTensor(value)

    def __len__(self):
        return self.length
