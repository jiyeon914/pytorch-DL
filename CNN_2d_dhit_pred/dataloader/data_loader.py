import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange

def read_data(cfg):
    nt = cfg.nt; data_dir = cfg.data_dir; nx = ny = cfg.data_res
    start = cfg.start; end = start + cfg.seq_length + nt; use_data = end - start + 1
    num_train, num_val, num_test = 500, 100, 50
    T = 3*10**1; dt = 2.5*10**-3; n = int(T/dt); K = 20

    def load_data(num_samples, data_type):
        data = np.zeros([num_samples, use_data, ny, nx, 1], dtype=np.float64)
        for num in range(num_samples):
            file_path = f"{data_dir}/{data_type} data/{data_type}{num+1} 2D HIT n={nx} T={T} dt={dt:.4f} K={K} data.h5"
            with h5py.File(file_path, 'r') as fr:
                data[num] = fr['w'][start:start+use_data]
        data = torch.from_numpy(data).transpose(2, 3).transpose(2, 4)
        return data
    data_train = load_data(num_train, 'Training')
    data_val = load_data(num_val, 'Validation')
    data_test = load_data(num_test, 'Test')

    training_rms = torch.std(data_train[:, 0], dim=(2, 3), correction=0)
    scaling_factor = torch.mean(training_rms[:, 0])
    data_train /= scaling_factor
    data_val /= scaling_factor
    data_test /= scaling_factor
    return data_train, data_val, data_test, scaling_factor

def get_batch(data, cfg, shuffle=True):
    nt = cfg.nt; start = cfg.start; end = start + cfg.seq_length + nt; use_data = end - start +1
    batch_size = cfg.batch_size; length = data.shape[0]

    data = data.to(torch.float32)
    input_field = rearrange(data[:, :use_data-nt], "s t c h w -> (t s) c h w")
    target = rearrange(data[:, nt:], "s t c h w -> (t s) c h w")
    data_cat = TensorDataset(input_field, target)
    dataloader = DataLoader(data_cat, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader
