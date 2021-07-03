import os
import tonic
import numpy as np
import multiprocessing
from tqdm.auto import tqdm
from functools import partial
from torch.utils.data.sampler import Sampler

# custom sampler for torch dataloader
class custom_sampler(Sampler):
    """Samples elements from a given list of indices.
    
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices
     
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

# create timesurfaces from a recording and keeps track of where they're located on a grid
# for use with a spatial histogram
def extract_spatial_features(data, transform, sensor_size, ordering, surface_dimensions, folder, save_as_double, add_noise, x_index, y_index, K):
    recording, label, index = data
    surfs = []
    cells = []
    cell_index = 0
    for i in range(sensor_size[0] // K +1):
        for j in range(sensor_size[1] // K +1):
            xs = recording[:,x_index]
            ys = recording[:,y_index]
            selection = recording[(xs >= i*K) & (xs < i*K+K) & (ys >= j*K) & (ys < j*K+K)]
            if len(selection) > 0:
                surfaces = transform(selection, sensor_size=sensor_size, ordering=ordering)
                surfs.extend(surfaces)
                cells.extend([cell_index] * surfaces.shape[0])
            cell_index += 1
    
    cells = np.array(cells)
    surfs = np.array(surfs)
    surfs = surfs.reshape(1, -1, surfs.shape[1], np.prod(surface_dimensions))
    if add_noise:
        surfs[0][:] = [ts+np.random.uniform(0,1e-6) for ts in surfs[0]]
    data_type = np.float64 if save_as_double else np.float32
    labels_file_name = './' + folder + 'labels/' + str(index) + '.npy'
    np.save(labels_file_name, np.array(np.int32([label])))
    cells_file_name = './' + folder + 'cells/' + str(index) + '.npy'
    np.save(cells_file_name, cells.astype(np.int32))
    file_name = './' + folder + 'data/' + str(index) + '.npy'
    np.save(file_name, surfs.astype(data_type))
    return surfs.shape

# create timesurfaces from a recording
# for use with a normal histogram
def extract_features(data, transform, sensor_size, ordering, surface_dimensions, folder, save_as_double, add_noise):
    recording, label, index = data
    surfs = transform(recording, sensor_size, ordering)
    surfs = surfs.reshape(1, -1, surfs.shape[1], np.prod(surface_dimensions))
    if add_noise:
        surfs[0][:] = [ts+np.random.uniform(0,1e-6) for ts in surfs[0]]
    data_type = np.float64 if save_as_double else np.float32
    labels_file_name = './' + folder + 'labels/' + str(index) + '.npy'
    np.save(labels_file_name, np.array(np.int32([label])))
    file_name = './' + folder + 'data/' + str(index) + '.npy'
    np.save(file_name, surfs.astype(data_type))
    return surfs.shape

# writes all recordings in a dataset into timesurfaces sequentially
def write_dataset(dataset, indices, folder, transform, surface_dimensions, save_as_double, add_noise=False, split_grid=True, K=10):
    
    dataloader = tonic.datasets.DataLoader(dataset, sampler=custom_sampler(indices), shuffle=False)
        
    if split_grid:
        for f in [folder, folder+'data/', folder+'labels/', folder+'cells/']:
            if not os.path.exists(f): os.makedirs(f)
        
        number_of_cells = (dataset.sensor_size[0] // K+1) * (dataset.sensor_size[1] // K+1)
        x_index = dataset.ordering.find("x")
        y_index = dataset.ordering.find("y")
        return [extract_spatial_features((events.squeeze().numpy(), target, index), transform=transform, sensor_size=dataset.sensor_size, ordering=dataset.ordering, folder=folder, surface_dimensions=surface_dimensions, save_as_double=save_as_double, add_noise=add_noise, x_index=x_index, y_index=y_index, K=K)\
            for index, (events, target) in enumerate(tqdm(iter(dataloader)))], number_of_cells
    else:
        for f in [folder, folder+'data/', folder+'labels/']:
            if not os.path.exists(f): os.makedirs(f)
        
        return [extract_features((events.squeeze().numpy(), target, index), transform=transform, sensor_size=dataset.sensor_size, ordering=dataset.ordering, folder=folder, surface_dimensions=surface_dimensions, save_as_double=save_as_double, add_noise=add_noise)\
                for index, (events, target) in enumerate(tqdm(iter(dataloader)))], 0