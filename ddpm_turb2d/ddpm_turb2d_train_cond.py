from importlib import reload
import logging
import sys
import yaml
import os

with open("./setup_turb2d.yaml", "r") as f:
    setup = yaml.safe_load(f)

sys.path.append(setup["repo_dir"])
output_dir = setup["output_dir"]
models_dir = setup["models_dir"]
data_dir = setup["data_dir"]
logging_dir = setup["logging_dir"]
torch_device = setup["torch_device"]
device = setup["torch_device"]

if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import models
reload(models)
from models import simple_unet
from models import fno2d
reload(simple_unet)
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding
from models import loss_functions
reload(loss_functions)
from models.loss_functions import LOSS_FUNCTIONS
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler
import metrics
reload(metrics)
from metrics import rfft_abs_mirror_torch
import yaml
import pickle
from models.simple_unet import SimpleUnet
from datetime import datetime
from scipy.stats import norm
from pprint import pformat

loaded_data = False
# Generate a string with the current date and time
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Diffusion process parameters
with open("./ddpm_turb2d_config.yml", 'r') as h:
    hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

timesteps = hyperparam_dict["timesteps"]
beta_start = hyperparam_dict["beta_start"]
beta_end = hyperparam_dict["beta_end"]
batch_size = hyperparam_dict["batch_size"]
epochs = hyperparam_dict["epochs"]
loss_function = hyperparam_dict["loss_function"]
loss_function_start = hyperparam_dict["loss_function_start"]
loss_function_start_batch = hyperparam_dict["loss_function_start_batch"]
loss_args_start = hyperparam_dict["loss_args_start"]
loss_args_end = hyperparam_dict["loss_args_end"]
beta_scheduler = hyperparam_dict["beta_scheduler"]
ddpm_arch = hyperparam_dict["ddpm_arch"]
ddpm_params = hyperparam_dict["ddpm_params"]
train_type = hyperparam_dict["train_type"]
lr = hyperparam_dict["lr"]
data_type = hyperparam_dict["data_type"]
model_name = hyperparam_dict["model_name"]

if model_name is None:
    model_name = f"ddpm_arch-{ddpm_arch}_time-{current_time}_timesteps-{timesteps}_epochs-{epochs}"

## Configure logging to log to both file and stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
## this will print to stdout and log to file
printlog = logging.info
printlog("-"*40)
printlog(f"Running ddpm_turb2d.py for {model_name}...")
printlog(f"loaded ddpm_turb2d_config: {pformat(hyperparam_dict)}")
printlog("-"*40)

model_dir = f"{models_dir}/{model_name}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

Nx=256
Ny=256
numchannels = 1
lead = 0 # same time predictions

## Load training data...change these to the correct locations
train_sparse_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_input_sparse.pkl"
train_pred_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_pred_sparse.pkl"
truth_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_truth.pkl"

## Pipeline: we will take the trian_pred_sparse, add some noise, then send it back to truth_train
with open(train_sparse_loc, "rb") as f:
    train_input_sparse = pickle.load(f)

with open(train_pred_loc, "rb") as f:
    train_pred_sparse = pickle.load(f)

with open(truth_loc, "rb") as f:
    truth_train = pickle.load(f)

train_input_sparse = train_input_sparse.permute((0,3,1,2))
train_pred_sparse = train_pred_sparse.permute((0,3,1,2))
truth_train = truth_train.permute((0,3,1,2))

# if data_type == "delta_mean":
#     data = data - data.mean()
#     test_data = test_data - test_data.mean()

## pulling shape of the data, and generating x/y grid, between 0 and 1
xs = torch.linspace(0, 1, Nx)
ys = torch.linspace(0, 1, Ny)

reload(models)
import models.simple_unet
reload(simple_unet)
import models.fno2d
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding, FNO2D_grid_tembedding_cond

# Model and optimizer
if ddpm_arch == "unet":
    model = SimpleUnet(**ddpm_params).to(device)
elif ddpm_arch == "unet_cond":
    model = SimpleUnetCond(**ddpm_params).to(device)
elif ddpm_arch == "fno2d":
    model = FNO2D_grid_tembedding_cond(**ddpm_params).to(device)
    model.gridx = xs.to(device)
    model.gridy = ys.to(device)
    
# optimizer = optim.Adam(model.parameters(), lr=1e-2) ## for fno2d
optimizer = optim.AdamW(model.parameters(), lr=lr)

t = torch.randint(0, timesteps, (batch_size,), device=device).long()

hyperparam_dict["epochs_run"] = 0
# Training loop
# We use truth_data as the data to train the model
# We recover the original data from the noised truth_data, then applied to partially noised fno sparse recovery in inference.
data = truth_train
cond_data = train_pred_sparse
# _ = model(data[:10], cond_data[:10], t) # Test the model with a small batch to see if it works
_ = model(data[:batch_size], cond_data[:batch_size], t) # Test the model with a small batch to see if it works

##############
## TRAINING ##
##############

if True:
    printlog(f"Training {model_name}...")
    if beta_scheduler == "linear":
        # betas and alphas for the diffusion process, linear noise scheduler
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        # cosine beta scheduler, https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

    # Plot alphas_cumprod
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(timesteps), c2n(alphas_cumprod), label='alphas_cumprod')
    plt.xlabel('Timesteps')
    plt.ylabel('Alphas Cumulative Product')
    plt.title(f'Alphas Cumulative Product over Timesteps\nbeta_start: {beta_start}, beta_end: {beta_end}')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.savefig(f"{model_dir}/alphas_cumprod.png", dpi=200)
    plt.close()

    if "epochs_run" not in hyperparam_dict:
        hyperparam_dict["epochs_run"] = 0

    loss_batch = []
    loss_epoch = []
    ibatch = 0
    for epoch in range(epochs):  # Adjust number of epochs
        printbatch = 0
        for batch_num in range(0, data.shape[0], batch_size):
            data_batch = data[batch_num:batch_num+batch_size].to(device)
            cond_batch = cond_data[batch_num:batch_num+batch_size].to(device)
            batch_size = data_batch.shape[0]

            if train_type == "noise":
                # Sample timesteps
                t = torch.randint(0, timesteps, (batch_size,), device=device)

                # Add noise
                noise = torch.randn_like(data_batch)
                noisy_data = torch.sqrt(alphas_cumprod[t].view(-1, 1, 1, 1)) * data_batch + torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1)) * noise

                # Train the model to predict the noise
                optimizer.zero_grad()
                predicted_noise = model(noisy_data, cond_batch, t)
                
                ## due to nature of absolute loss allowing for amplitude flips in spectral space, we start with mse to get close initially
                if ibatch <= loss_function_start_batch or loss_function_start_batch == -1:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](predicted_noise.permute((0,2,3,1)), noise.permute((0,2,3,1)), **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_function](predicted_noise.permute((0,2,3,1)), noise.permute((0,2,3,1)), **loss_args_end) # Loss is between predicted noise and actual noise

                loss.backward()
                optimizer.step()

            if train_type == "tauFull":
                # Sample timesteps...full from timesteps to 0
                tau0 =  torch.tile(torch.tensor(0, device=device), (batch_size,))
                tau1 = torch.tile(torch.tensor(timesteps-1, device=device), (batch_size,))
                rev_steps = timesteps - 0

                ## adding noise for each of the tau0 and tau1 timesteps
                noise = torch.randn_like(data_batch)
                data_noise_rev = torch.sqrt(alphas_cumprod[tau0].view(-1, 1, 1, 1)) * data_batch + torch.sqrt(1 - alphas_cumprod[tau0].view(-1, 1, 1, 1)) * noise # weighted noise addition to the data_batch, cumulative product up to tnoise, which we are trying to have the model match
                data_noise = torch.sqrt(alphas_cumprod[tau1].view(-1, 1, 1, 1)) * data_batch + torch.sqrt(1 - alphas_cumprod[tau1].view(-1, 1, 1, 1)) * noise ## weighted noise addition to the data_batch, cumulative product up to tnoise+rev_steps, which we will reverse

                # Train the model to predict the noise
                optimizer.zero_grad()
                
                x=data_noise

                for tr in range(rev_steps):
                    # timestep = torch.tensor([t], device=device)
                    tau = tau1 - tr # starting from the last timestep, reverse noise
                    predicted_noise = model(x, tau)
                    alpha_t = alphas[tau].view(-1, 1, 1, 1)
                    alpha_bar_t = alphas_cumprod[tau].view(-1, 1, 1, 1)
                    x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)
                    
                    ## this noise addition in the reverse process gives strange results during training (no convergence)
                    teff = (torch.max(torch.ones(tau.shape).to(device), tau)).int()
                    beta_t = betas[teff].view(-1, 1, 1, 1)
                    ## adding beta_t noise to reverse process only if tau > 0
                    step_add_noise = (tau.view(-1, 1, 1, 1) > 0).int()
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x, requires_grad=True) * step_add_noise # Add noise except for the last step

                ## due to nature of absolute loss allowing for amplitude flips in spectral space, we start with mse to get close initially
                if ibatch < loss_function_start_batch:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0,2,3,1)), data_noise_rev.permute((0,2,3,1)), **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0,2,3,1)), data_noise_rev.permute((0,2,3,1)), **loss_args_end) # Loss is between predicted noise and actual noise

                loss.backward()
                optimizer.step()

            loss_batch.append([ibatch, loss.item()])

            if ibatch >= printbatch:
                printlog(f"Epoch [{epoch+1}/{epochs}], ibatch {ibatch+1}, loss_use: {loss_use}, Loss: {loss.item():.8f}")
                printbatch = ibatch + 10
            
            ibatch += 1

        loss_epoch.append([ibatch, loss.item()])

        loss_batch_arr = np.array(loss_batch)
        loss_epoch_arr = np.array(loss_epoch)
        ## plot epoch and batch loss
        plt.plot(loss_batch_arr[:,0], loss_batch_arr[:,1], color = "blue", label = "batch loss", alpha = 0.5)
        plt.scatter(loss_epoch_arr[:,0], loss_epoch_arr[:,1], color = "red", label = "epoch loss")
        plt.xlabel("Batch number")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend(loc = "upper right")
        plt.savefig(f"{model_dir}/loss_batch_epoch.png", dpi = 200)
        plt.close()

        # Save the trained model and hyperparameters used
        torch.save(model.state_dict(), f"{model_dir}/{model_name}_epoch-{epoch+1}.pt")
        hyperparam_dict["epochs_run"] += 1
        with open(f"{model_dir}/config.yml", 'w') as h:
            yaml.dump(hyperparam_dict, h, default_flow_style=False)

else:
    ## loading previous model
    model_name = "ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79"
    model_loc = "/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79.pt"
    model.load_state_dict(torch.load(model_loc))

    # Diffusion process parameters
    with open("/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/config.yml", 'r') as h:
        hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

    timesteps = hyperparam_dict["timesteps"]
    beta_start = hyperparam_dict["beta_start"]
    beta_end = hyperparam_dict["beta_end"]
    batch_size = hyperparam_dict["batch_size"]
    epochs = hyperparam_dict["epochs"]
    # loss_function = hyperparam_dict["loss_function"]
    # loss_function_start = hyperparam_dict["loss_function_start"]
    # loss_function_start_batch = hyperparam_dict["loss_function_start_batch"]
    # loss_args = hyperparam_dict["loss_args"]
    beta_scheduler = hyperparam_dict["beta_scheduler"]
    epochs_run = hyperparam_dict["epochs_run"]

    if beta_scheduler == "linear":
        # betas and alphas for the diffusion process, linear noise scheduler
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        # cosine beta scheduler, https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

#############
## TESTING ##
#############

## number of initial random states to reverse sample from
seeds = 40
stausteps = 0
ftausteps = timesteps
timesteps_rev_list = np.arange(stausteps, ftausteps)
its = np.arange(len(timesteps_rev_list))
its_timesteps = np.array(list(zip(its, reversed(timesteps_rev_list))))
num_rev_steps = len(its_timesteps)
# ts_idx = timesteps_rev_list[::int(len(timesteps_rev_list)/20)] ## indices of interest to plot
# ts_idx = ts_idx + [len(timesteps_rev_list)-1] ## add the last timestep
idx_skip = int(len(timesteps_rev_list)/40)
its_timesteps_plot_og = its_timesteps[::idx_skip]
its_timesteps_plot = np.concatenate((its_timesteps_plot_og, its_timesteps[-idx_skip:]), axis = 0)
its_timesteps_plot2 = its_timesteps_plot_og
## normal zero reverse last timestep
itrevfinal = np.where(its_timesteps[:,1] == 0)[0][0]

model.eval()
probe_data = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny))
probe_noise = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny))

seeds_idx = np.arange(seeds)

## run on test data
test_input_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_input_sparse.pkl"
test_pred_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_pred_sparse.pkl"
truth_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_truth.pkl"

## data to use for prediction
with open(test_input_loc, "rb") as f:
    test_input_sparse = pickle.load(f)

with open(test_pred_loc, "rb") as f:
    test_pred_sparse = pickle.load(f)

with open(truth_loc, "rb") as f:
    truth_test = pickle.load(f)

# bug with code saving two channels
test_pred_sparse = test_pred_sparse[:,:,:,[0]]

test_input_sparse = test_input_sparse.permute((0,3,1,2))
test_pred_sparse = test_pred_sparse.permute((0,3,1,2))
truth_test = truth_test.permute((0,3,1,2))


data_test = test_pred_sparse
cond_data_test = test_pred_sparse

## running inference
with torch.no_grad():
    for s in range(seeds):
        printlog(f"Seed: {s+1}/{seeds}")
        # x = torch.randn(1, 3, 128, 128).to(device) # Start with random noise

        # Add noise
        noise = torch.randn_like(data_test[[s]])
        ## full cumulative noise up to timestep timesteps-1
        noisy_data = torch.sqrt(alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * data_test[[s]] + torch.sqrt(1 - alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * noise
        x = noisy_data

        for it, t in its_timesteps:
            timestep = torch.tensor([t], device=device)
            predicted_noise = model(x, cond_data_test[[s]], timestep)
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

            ## this step doesnt work with cosine beta noise scheduler...replace with something else? or dont add noise at all
            if beta_scheduler == "linear":
                if t > stausteps:
                    teff = max(1, t)
                    beta_t = betas[teff]
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x) # Add noise except for the last step

            probe_data[s, it] = x
            probe_noise[s, it] = predicted_noise

## testing analysis, comparisons
## swapping x/y so spectrums are computed correctly
probe_pred_data = probe_data.permute(0, 1, 2, 3, 5, 4)
probe_pred_noise = probe_noise.permute(0, 1, 2, 3, 5, 4)

train_input_sparse_use = test_input_sparse[seeds_idx].permute(0, 1, 3, 2)
train_pred_sparse_use = test_pred_sparse[seeds_idx].permute(0, 1, 3, 2)
train_truth_use = truth_test[seeds_idx].permute(0, 1, 3, 2)

path_outputs_model = f"{output_dir}/{model_name}"
if not os.path.exists(path_outputs_model):
    printlog(f"Creating directory: {path_outputs_model}")
    os.makedirs(path_outputs_model)
    
path_outputs_model_timesteps = f"{path_outputs_model}/fno_ddpm_timesteps"
if not os.path.exists(path_outputs_model_timesteps):
    printlog(f"Creating directory: {path_outputs_model}")
    os.makedirs(path_outputs_model_timesteps)

chs = ["vorticity"]
yscales = ["linear", "log"]

true_sparse_pred_diff = f"{path_outputs_model_timesteps}/true_sparse_pred_diff"

if not os.path.exists(true_sparse_pred_diff):
    os.makedirs(true_sparse_pred_diff)

# Plot probe_pred_data, data, and data_true side by side for the first 3 seeds

itau1, tau1 = its_timesteps[0]
itau2, tau2 = its_timesteps[-10]
itau3, tau3 = its_timesteps[-1]

## writing figure for the prediction steps, including sparse, sparse fno, sparse fno ddpms, sparse fno ddpms final, and truth
for istep, s in enumerate(seeds_idx,0):
    print(f"Plotting timestep: {istep}")

    str_step = "0"*(6-len(str(istep)))+str(istep)

    fig, axs = plt.subplots(1, 6, figsize=(18, 5))

    axs[0].imshow(train_input_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[0].set_title("Sparse 99%")
    axs[1].imshow(train_pred_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[1].set_title("FNO on Sparse")
    axs[2].imshow(probe_pred_data[s, 0, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[2].set_title(r"FNO+DDPM $\tau$ = %i"%tau1)
    axs[3].imshow(probe_pred_data[s, -10, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[3].set_title(r"FNO+DDPM $\tau$ = %i"%tau2)
    axs[4].imshow(probe_pred_data[s, -1, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[4].set_title(r"FNO+DDPM $\tau$ = %i"%tau3)
    axs[5].imshow(train_truth_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[5].set_title("Truth")

    plt.suptitle(f"Comparison for Seed {s}\n full tau, conditioning/{timesteps}")
    plt.tight_layout()
    plt.savefig(f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}.png", dpi = 300)
    plt.close()

png_locs_txt = ""
for istep, s in enumerate(seeds_idx,0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    png_loc = f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}.png"
    png_locs_txt += f"{png_loc}\n"

true_sparse_pred_diff_txt_loc = f"{path_outputs_model_timesteps}/true_sparse_pred_diff_timestep.txt"    
with open(true_sparse_pred_diff_txt_loc, "w") as f:
    f.write(png_locs_txt)

# Create gif from list of png files
with open(true_sparse_pred_diff_txt_loc, "r") as f:
    png_files = f.read().strip().split("\n")
    png_files_str = " ".join(png_files)

os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {path_outputs_model_timesteps}/true_sparse_pred_diff.mp4')


## writing figure for the prediction steps, sparse fno, sparse fno ddpms final, and truth
for istep, s in enumerate(seeds_idx,0):
    print(f"Plotting timestep: {istep}")

    str_step = "0"*(6-len(str(istep)))+str(istep)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(train_pred_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[0].set_title("FNO on Sparse")
    axs[1].imshow(probe_pred_data[s, -1, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[1].set_title(r"FNO+DDPM $\tau$ = %i"%tau3)
    axs[2].imshow(train_truth_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[2].set_title("Truth")

    plt.suptitle(f"Comparison for Seed {s}\n full tau, conditioning/{timesteps}")
    plt.tight_layout()
    plt.savefig(f"{true_sparse_pred_diff}/true_pred_timestep-{str_step}.png", dpi = 300)
    plt.close()

os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {path_outputs_model_timesteps}/true_pred_timestep.mp4')



## spectrum plots, for each of the 6 arrays used for plotting the imshows above
# Compute the y average Fourier transform for each of the 6 arrays used for plotting the imshows above
def compute_y_avg_fft(data):
    data_fft = np.fft.rfft(data, axis=1)
    data_fft_avg = np.mean(np.abs(data_fft), axis=0)
    return data_fft_avg

# Prepare the data for FFT computation
arrays_to_plot = [
    train_input_sparse_use[:, 0, :, :].cpu().numpy(),
    train_pred_sparse_use[:, 0, :, :].cpu().numpy(),
    probe_pred_data[:, -400, 0, 0, :, :].cpu().numpy(),
    probe_pred_data[:, -10, 0, 0, :, :].cpu().numpy(),
    probe_pred_data[:, -1, 0, 0, :, :].cpu().numpy(),
    train_truth_use[:, 0, :, :].cpu().numpy()
]

labels = {
    "Sparse 99%": "blue",
    "FNO on Sparse": "orange",
    r"FNO+DDPM $\tau$ = %i" % tau1: "green",
    r"FNO+DDPM $\tau$ = %i" % tau2: "red",
    r"FNO+DDPM $\tau$ = %i" % tau3: "purple",
    "Truth": "black"
}

# Plot the y average Fourier transform
fig, ax = plt.subplots(figsize=(10, 6))

for d, (label, color) in zip(arrays_to_plot, labels.items()):
    for s in range(d.shape[0]):
        y_avg_fft = compute_y_avg_fft(d[s])  # Compute FFT for the first seed
        # freqs = np.fft.rfftfreq(d.shape[2])
        ks = np.arange(y_avg_fft.shape[0])
        ax.plot(ks[2:-1], y_avg_fft[2:-1], color=color, alpha = .3)

for d, (label, color) in zip(arrays_to_plot, labels.items()):
    ax.plot([], [], label = label, color=color, alpha = 1)

ax.set_xlabel('k')
ax.set_ylabel('Amplitude')
ax.set_title('Vertically Average Spectrum')
ax.legend()
ax.grid(alpha = .3)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{path_outputs_model_timesteps}/y_avg_fft_comparison.png", dpi=300)
plt.close()

## plot images of the probe_data
for s in seeds_idx[:2]:
    plot_timestep = f"{path_outputs_model_timesteps}/s-{s}"
    if not os.path.exists(plot_timestep):
        os.makedirs(plot_timestep)

    for it, t in its_timesteps_plot:
        fig, axs = plt.subplots(1, 1, figsize=(8, 5))
        axs = [axs]

        for ich, ch in enumerate(chs,0):

            lims = [
                torch.min(probe_data[s, it, 0, ich]).cpu().numpy(),
                0,
                torch.max(probe_data[s, it, 0, ich]).cpu().numpy()
            ]

            im = axs[ich].imshow(probe_data[s, it, 0, ich].cpu().numpy(), cmap='coolwarm')
            axs[ich].set_title(f"{ch}")
            cbar = fig.colorbar(im, orientation='vertical', ticks = lims, fraction=0.046, pad=0.04)

        plt.suptitle(f"{model_name}\nrev timestep: {t}")
        plt.savefig(f"{plot_timestep}/probe_revStep_s-{s}_t-{pthstr(t)}.png")
        plt.close()

## spectrum plots of the probe_data
for revfinal in [len(its_timesteps)]:
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs = [axs]
    startk = 2

    true_data_fft = rfft_abs_mirror_torch(data, axis=3).mean(axis = 0)
    true_data_fft_mean = torch.mean(true_data_fft, dim = 1)

    ks = np.arange(true_data_fft_mean.shape[1])
    axs[0].plot([],[], color = "black", linewidth = 2, label = f"actual")
    axs[0].plot([],[], color = "black", linewidth = 2, label = f"true final reverse ", linestyle = "--")
    
    # for ich, ch in enumerate(chs,0):
    # for it, t in its_timesteps[ts_idx]:
    for it, t in its_timesteps_plot:
        ## taking the mean of the fft over the seeds
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis = 0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1)

        if (it % (len(its_timesteps)//3)) == 0:
            axs[0].plot([],[], color = cm.rainbow(it/its_timesteps.shape[0]), label = f"reverse t: {t}")

        axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)

    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=axs[-1])

    numticks = 10+1
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')
    ## actual
    axs[0].plot(ks[startk:], c2n(true_data_fft_mean[0,startk:]), color = "black", linewidth = 3)

    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis = 0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1)
    axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]), color = "black", linewidth = 3, linestyle = "--")

    axs[0].set_title(f"{chs[0]}")

    for ax in axs:
        ax.set_xlabel('k')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=0.3)

    axs[0].legend()

    plt.suptitle(f"Probe data FFT\nReverse timesteps from {ftausteps} to {stausteps}")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    for yscale in yscales:
        for ax in axs:
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrum_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)

    plt.close()


yscales_diff = ["linear", "symlog"]
## spectrum difference
for revfinal in [len(its_timesteps)]:
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs = [axs]
    startk = 2

    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis = 0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim = 1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    for it, t in its_timesteps_plot:
        ## taking the mean of the fft over the seeds
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis = 0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1).to(device)

        axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]-true_data_fft_mean[0,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)
    
    ## plot final spectrum as black on each axis
    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis = 0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1).to(device)
    axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]-true_data_fft_mean[0,startk:]), color = "black", linewidth = 3, linestyle = "--")

    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=axs[-1])

    numticks = 10+1
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    axs[0].set_title(f"{chs[0]}")

    for ax in axs:
        ax.set_xlabel('k')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=0.3)

    plt.suptitle(f"Probe data FFT, difference from true\nReverse timesteps from {ftausteps} to {stausteps}")

    for yscale in yscales_diff:
        for ax in axs:
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumDiff_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)
            
    plt.close()

## wave number error at varying tau
yscales_diff = ["linear", "symlog"]
for revfinal in [len(its_timesteps)]:
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis = 0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim = 1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    startk = 2
    ks_keep = ks[startk:]

    chs_ks_timesteps = np.zeros((len(chs), len(ks), len(its_timesteps)))

    for it, t in its_timesteps:
        print(f"{t}, {np.min(c2n(probe_data_fft_mean[0]))}")
        ## taking the mean of the fft over the seeds
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis = 0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1).to(device)
        chs_ks_timesteps[0,:,it] = c2n((probe_data_fft_mean[0]-true_data_fft_mean[0,:]).abs())
    
    # ## taking the mean of the fft over the seeds
    # probe_data_fft = rfft_abs_mirror_torch(probe_data[:, i, 0], axis=3).mean(axis = 0)
    # probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1).to(device)[:,startk:]
    # print(np.min(c2n(probe_data_fft_mean[0])))
    # chs_ks_timesteps[0,:,it] = c2n(probe_data_fft_mean[0])
    # chs_ks_timesteps[1,:,it] = c2n(probe_data_fft_mean[1])
    # chs_ks_timesteps[2,:,it] = c2n(probe_data_fft_mean[2])

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs = [axs]

    for ich, ch in enumerate(chs,0):
        for k in ks_keep:
            axs[ich].plot(its, chs_ks_timesteps[ich,k,its], color = cm.rainbow_r(k/ks[-1]), alpha = .5)

    # ## plot final spectrum as black on each axis
    # probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis = 0)
    # probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1).to(device)
    # axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]-true_data_fft_mean[0,startk:]), color = "black", linewidth = 3, linestyle = "--")
    # axs[1].plot(ks[startk:], c2n(probe_data_fft_mean[1,startk:]-true_data_fft_mean[1,startk:]), color = "black", linewidth = 3, linestyle = "--")
    # axs[-1].plot(ks[startk:], c2n(probe_data_fft_mean[2,startk:]-true_data_fft_mean[2,startk:]), color = "black", linewidth = 3, linestyle = "--")

    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow_r'), ax=axs[-1])

    numticks = 10+1
    kticks_use = np.linspace(ks[-1], ks[0], num=numticks).astype(int)
    kticks_use_norm = (kticks_use)/ks[-1]
    cbar.set_ticks(kticks_use_norm)
    cbar.set_ticklabels([f"{k}" for k in kticks_use])
    cbar.set_label('Wave number k')

    axs[0].set_title(f"{chs[0]}")

    for ax in axs:
        ax.set_xlabel('1 - tau (inverse diffusion step)')
        ax.set_ylabel('Amplitude Error')
        ax.grid(alpha=0.3)

    plt.suptitle(f"Probe data FFT, difference from true vs timesteps, for each k")

    for yscale in yscales_diff:
        for ax in axs:
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/kDiff_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)
            
    plt.close()

## probe noise plots
for revfinal in [len(its_timesteps)]:
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs = [axs]
    startk = 2

    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis = 0)
    true_data_fft_mean = torch.mean(true_data_fft, dim = 1)

    ks = np.arange(true_data_fft_mean.shape[1])
    # axs[0].plot([],[], color = "black", linewidth = 2, label = f"actual")
    # axs[0].plot([],[], color = "black", linewidth = 2, label = f"true final reverse ", linestyle = "--")
    
    # for ich, ch in enumerate(chs,0):
    # for it, t in its_timesteps[ts_idx]:
    for it, t in its_timesteps_plot:
        ## taking the mean of the fft over the seeds
        probe_noise_fft = rfft_abs_mirror_torch(probe_noise[:, it, 0], axis=3).mean(axis = 0)
        probe_noise_fft_mean = torch.mean(probe_noise_fft, dim = 1)

        if (it % (len(its_timesteps)//3)) == 0:
            axs[0].plot([],[], color = cm.rainbow(it/its_timesteps.shape[0]), label = f"reverse t: {t}")

        axs[0].plot(ks[startk:], c2n(probe_noise_fft_mean[0,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)


    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=axs[-1])

    numticks = 10+1
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')
    ## actual
    # axs[0].plot(ks[startk:], c2n(true_data_fft_mean[0,startk:]), color = "black", linewidth = 3)

    probe_noise_fft = rfft_abs_mirror_torch(probe_noise[:, itrevfinal, 0], axis=3).mean(axis = 0)
    probe_noise_fft_mean = torch.mean(probe_noise_fft, dim = 1)
    # axs[0].plot(ks[startk:], c2n(probe_noise_fft_mean[0,startk:]), color = "black", linewidth = 3, linestyle = "--")

    axs[0].set_title(f"{chs[0]}")

    axs[0].legend()
    for ax in axs:
        ax.set_xlabel('k')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f"Probe data noise spectrum\nReverse timesteps from {ftausteps} to {stausteps}")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()

    for yscale in yscales:
        for ax in axs:
            ax.set_yscale(yscale)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumNoise_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)

    plt.close()

## noise distribution plots
for revfinal in [len(its_timesteps)]:
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs = [axs]

    bins = 500
    pdf_range = (-4,4)
    for it, t in its_timesteps_plot2:
        # taking the mean of the fft over the seeds
        # probe_noise
        # Calculate the normalized probability distribution function
        for ich, ch in enumerate(chs,0):
            probe_noise_flat = c2n(probe_noise[:, it, 0, ich]).flatten()
            hist, bin_edges = np.histogram(probe_noise_flat, bins=bins, range = pdf_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            hist = hist/(hist.sum()*bin_width) # normalized
            axs[ich].plot(bin_centers, hist, color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)

    x = np.linspace(pdf_range[0], pdf_range[1], bins)
    pdf = norm.pdf(x, loc=0, scale=1)
    axs[0].plot(x, pdf, color="black", alpha=1.0, label="Gaussian")

    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=axs[-1])

    numticks = 10+1
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    for ich, ch in enumerate(chs,0):
        axs[0].set_title(f"{chs[0]}")
        axs[0].set_xlim(pdf_range)

    axs[0].legend()
    for ax in axs:
        ax.set_xlabel('Noise')
        ax.set_ylabel('PDF')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f"hist data noise spectrum\nReverse timesteps from {ftausteps} to {stausteps}")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    for yscale in ["linear", "log"]:
        for ax in axs:
            ax.set_yscale(yscale)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/histNoise_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)

    plt.close()






## finding the pdf of the diffusion process noise 
printlog(f"Model: {model_name} pipeline finished.")

## old script
"""
## number of initial random states to reverse sample from
seeds = 100
stausteps = -timesteps
ftausteps = timesteps
timesteps_rev_list = np.arange(stausteps, ftausteps)
its = np.arange(len(timesteps_rev_list))
its_timesteps = np.array(list(zip(its, reversed(timesteps_rev_list))))
num_rev_steps = len(its_timesteps)
ts_idx = [0,1,2,3,timesteps-3,timesteps-2,timesteps-1,num_rev_steps-3, num_rev_steps-2, num_rev_steps-1] ## indices of interest to plot

## normal zero reverse last timestep
itrevfinal = np.where(its_timesteps[:,1] == 0)[0][0]


model.eval()
probe_data = torch.zeros((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny))

with torch.no_grad():
    for s in range(seeds):
        if s % 100 == 0:
            printlog(f"Seed {s}")
        # x = torch.randn(1, 3, 128, 128).to(device) # Start with random noise

        # Add noise
        noise = torch.randn_like(data[[s]])
        noisy_data = torch.sqrt(alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * data[[s]] + torch.sqrt(1 - alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * noise
        x = noisy_data

        for it, t in its_timesteps:
            timestep = torch.tensor([t], device=device)
            predicted_noise = model(x, timestep)
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

            if t > stausteps:
                teff = max(1, t)
                beta_t = betas[teff]
                x = x + torch.sqrt(beta_t) * torch.randn_like(x) # Add noise except for the last step

            probe_data[s, it] = x

## to run analysis later
with open(f"{model_dir}/probe_data_seeds-{seeds}_sts-{stausteps}_fts-{ftausteps}.pkl", "wb") as f:
    pickle.dump(probe_data, f)

## test plots
# path_outputs_model = f"{output_dir}/{model_name}"
path_outputs_model = f"{output_dir}/{model_name}_modRev1"
if not os.path.exists(path_outputs_model):
    os.makedirs(path_outputs_model)
    
path_outputs_model_timesteps = f"{path_outputs_model}/timesteps"
if not os.path.exists(path_outputs_model_timesteps):
    os.makedirs(path_outputs_model_timesteps)

for s in [0,1,2]:
    for it, t in its_timesteps[ts_idx]:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(probe_data[s, it, 0, 0].cpu().numpy(), cmap='coolwarm')
        axs[1].imshow(probe_data[s, it, 0, 1].cpu().numpy(), cmap='coolwarm')
        axs[-1].imshow(probe_data[s, it, 0, 2].cpu().numpy(), cmap='coolwarm')
        plt.suptitle(f"t: {t}")
        plt.savefig(f"{path_outputs_model_timesteps}/probe_s-{s}_t-{pthstr(t)}.png")
        plt.close()

for revfinal in [itrevfinal, len(its_timesteps)]:
    for yscale in ["linear", "log"]:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        startk = 2

        true_data_fft = rfft_abs_mirror_torch(data[s], axis=2)
        true_data_fft_mean = torch.mean(true_data_fft, dim = 1)

        ks = np.arange(true_data_fft_mean.shape[1])
        axs[0].plot([],[], color = "black", linewidth = 2, label = f"actual")
        axs[0].plot([],[], color = "black", linewidth = 2, label = f"true final reverse ", linestyle = "--")

        # for it, t in its_timesteps[ts_idx]:
        for it, t in its_timesteps[:revfinal+1:len(its_timesteps)//20]:
            ## taking the mean of the fft over the seeds
            probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis = 0)
            probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1)

            if (it % (len(its_timesteps)//3)) == 0:
                axs[0].plot([],[], color = cm.rainbow(it/its_timesteps.shape[0]), label = f"reverse t: {t}")

            axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)
            axs[1].plot(ks[startk:], c2n(probe_data_fft_mean[1,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)
            axs[-1].plot(ks[startk:], c2n(probe_data_fft_mean[2,startk:]), color = cm.rainbow(it/its_timesteps.shape[0]), alpha = .5)

        ## actual
        axs[0].plot(ks[startk:], c2n(true_data_fft_mean[0,startk:]), color = "black", linewidth = 3)
        axs[1].plot(ks[startk:], c2n(true_data_fft_mean[1,startk:]), color = "black", linewidth = 3)
        axs[-1].plot(ks[startk:], c2n(true_data_fft_mean[2,startk:]), color = "black", linewidth = 3)

        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis = 0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim = 1)
        axs[0].plot(ks[startk:], c2n(probe_data_fft_mean[0,startk:]), color = "black", linewidth = 3, linestyle = "--")
        axs[1].plot(ks[startk:], c2n(probe_data_fft_mean[1,startk:]), color = "black", linewidth = 3, linestyle = "--")
        axs[-1].plot(ks[startk:], c2n(probe_data_fft_mean[2,startk:]), color = "black", linewidth = 3, linestyle = "--")

        axs[0].legend()
        for ax in axs:
            ax.set_xlabel('k')
            ax.set_ylabel('FFT')
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3)

        plt.suptitle(f"Probe data FFT\nReverse timesteps from {ftausteps} to {stausteps}, repeated beta")
        plt.savefig(f"{path_outputs_model}/spectrum_revFinal-{revfinal}_yscale-{yscale}.png", dpi = 200)
"""