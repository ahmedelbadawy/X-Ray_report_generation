"""Loads chest X-rays from an HDF5 file in batches for fast encoding."""
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def to_pil(arr):
    arr = np.asarray(arr)
    img = Image.fromarray(arr)
    return img.convert('RGB')                # ALBEF expects 3 channels


class CXRDataset(Dataset):
    def __init__(self, h5_path, transform):
        self.h5_path = h5_path        
        self.transform = transform
        self.h5 = None
        with h5py.File(h5_path, 'r') as f:      # read length once, file closed after
            self.n = len(f["cxr"])
                     

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.h5 is None:   # to run inside each worker 
            self.h5 = h5py.File(self.h5_path, 'r')
        arr = self.h5["cxr"][i]     
        return self.transform(to_pil(arr))  