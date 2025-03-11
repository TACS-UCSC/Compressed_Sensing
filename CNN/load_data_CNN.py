
import torch
import random
import numpy as np
import einops
from pathlib import Path

def shape2coordinates(spatial_shape, max_value=1.0):
    """Create coordinates from a spatial shape.
    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.
    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, max_value, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)

def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates

def set_seed(seed=33):
    """Set all seeds for the experiments.
    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

print("Load data for 99 sparsity")
def get_dynamics_data(
    data_dir_mask = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/processed_data/New_norm_masked/channel_masked/ch_mask99.npy",
    data_dir_original = "/glade/derecho/scratch/nasefi/compressed_sensing/Beta_channel-turbulence_n/data_directories/processed_data/New_norm_original/chorigin_all.npy", 
    # remove_clouds = True,
    # dataset_name,
    # ntrain,
    # ntest,
    # seq_inter_len = 20,
    # seq_extra_len = 20,
    # sub_from=1,
    # sub_tr=1,
    # sub_te=1,
    # same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.
    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    Raises:
        NotImplementedError: _description_
    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    with open(data_dir_mask, 'rb') as f:
        data_masked = np.load(f) 
    with open(data_dir_original, 'rb') as f:
        data_original = np.load(f)

    print("data_masked", data_masked.shape)
    print("data_original", data_original.shape)

    # #change the size from [10000,256,256] to [10000, 1, 256,256]

    # data_masked =np.expand_dims(data_masked, axis=1)
    # data_original= np.expand_dims(data_original, axis=1)

    print("ex_masked", data_masked.shape)
    print("ex_original",type(data_original), data_original.shape)


    u_train_m = torch.Tensor(data_masked[:7000])
    u_train_o = torch.Tensor(data_original[:7000])
    u_test_m = torch.Tensor(data_masked[7000:])
    u_test_o = torch.Tensor(data_original[7000:]) #


    print("Curious", type(u_train_m.shape), u_train_m.shape)   # CNN_u_train_m torch.Size([6999, 1, 256, 256]
    print("curiousss", u_train_o.shape)
    print("curiousnn", u_test_m.shape)     
    print("curiousddd", u_test_o.shape)


    spatial_shape = [256, 256]
    grid_tr_m = shape2coordinates(spatial_shape)
    grid_tr_o = shape2coordinates(spatial_shape)
    grid_te_m = shape2coordinates(spatial_shape)
    grid_te_o = shape2coordinates(spatial_shape)

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates
    grid_tr_m = einops.repeat(
        grid_tr_m, "... -> b ... t", t=u_train_m.shape[-1], b=u_train_m.shape[0]
    )

    grid_tr_o = einops.repeat(
        grid_tr_o, "... -> b ... t", t=u_train_o.shape[-1], b=u_train_o.shape[0]
    )

    # grid_tr_extra = einops.repeat(
    #     grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    # )
    grid_te_m = einops.repeat(
        grid_te_m, "... -> b ... t", t=u_test_m.shape[-1], b=u_test_m.shape[0]
    )

    grid_te_o = einops.repeat(
        grid_te_o, "... -> b ... t", t=u_test_o.shape[-1], b=u_test_o.shape[0]
    )

    return u_train_m, u_train_o, u_test_m, u_test_o



print("Finish")

u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()


print("FNew_u_train_m", u_train_m.shape)    # torch.Size([7000, 1, 256, 256]
print("FNew_u_train_0", u_train_o.shape)    
print("FNew_u_test_m", u_test_m.shape)   
print("FNew_u_test_o", u_test_o.shape)

# FNew_u_train_m torch.Size([7000, 1, 256, 256])
# FNew_u_train_0 torch.Size([7000, 1, 256, 256])
# FNew_u_test_m torch.Size([3000, 1, 256, 256])
# FNew_u_test_o torch.Size([3000, 1, 256, 256])
# FNew_tr torch.Size([7000, 256, 256, 2, 256])
# FNew_te torch.Size([3000, 256, 256, 2, 256])
