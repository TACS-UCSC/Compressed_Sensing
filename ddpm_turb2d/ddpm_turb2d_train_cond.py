from importlib import reload
import logging
import sys
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from datetime import datetime
from scipy.stats import norm
from pprint import pformat

# Load configuration
with open("./setup_turb2d.yaml", "r") as f:
    setup = yaml.safe_load(f)

# Setup paths and device
sys.path.append(setup["repo_dir"])
output_dir = setup["output_dir"]
models_dir = setup["models_dir"]
data_dir = setup["data_dir"]
logging_dir = setup["logging_dir"]
device = setup["torch_device"]

if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

# Import custom modules
import models
reload(models)
from models import simple_unet
from models import fno2d
reload(simple_unet)
reload(fno2d)
from models.simple_unet import SimpleUnet, SimpleUnetCond
from models.fno2d import FNO2D_grid, FNO2D_grid_tembedding, FNO2D_grid_tembedding_cond
from models import loss_functions
reload(loss_functions)
from models.loss_functions import LOSS_FUNCTIONS
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler
import metrics
reload(metrics)
from metrics import rfft_abs_mirror_torch

# Get current time for model naming
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Load hyperparameters
with open("./ddpm_turb2d_config.yml", 'r') as h:
    hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

# Extract hyperparameters
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

# Generate model name if not provided
if model_name is None:
    model_name = f"ddpm_arch-{ddpm_arch}_time-{current_time}_timesteps-{timesteps}_epochs-{epochs}"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
printlog = logging.info
printlog("-"*40)
printlog(f"Running ddpm_turb2d.py for {model_name}...")
printlog(f"loaded ddpm_turb2d_config: {pformat(hyperparam_dict)}")
printlog("-"*40)

# Create model directory
model_dir = f"{models_dir}/{model_name}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data dimensions
Nx = 256
Ny = 256
numchannels = 1
lead = 0  # same time predictions

# Load training data
train_sparse_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_input_sparse.pkl"
train_pred_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_pred_sparse.pkl"
truth_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/train_truth.pkl"

# Load data files
with open(train_sparse_loc, "rb") as f:
    train_input_sparse = pickle.load(f)

with open(train_pred_loc, "rb") as f:
    train_pred_sparse = pickle.load(f)

with open(truth_loc, "rb") as f:
    truth_train = pickle.load(f)

# Permute dimensions to [batch, channel, height, width]
train_input_sparse = train_input_sparse.permute((0, 3, 1, 2))
train_pred_sparse = train_pred_sparse.permute((0, 3, 1, 2))
truth_train = truth_train.permute((0, 3, 1, 2))

# Create grid coordinates
xs = torch.linspace(0, 1, Nx)
ys = torch.linspace(0, 1, Ny)

# Reload modules to ensure latest versions
reload(models)
reload(simple_unet)
reload(fno2d)

# Initialize model based on architecture
if ddpm_arch == "unet":
    model = SimpleUnet(**ddpm_params).to(device)
elif ddpm_arch == "unet_cond":
    model = SimpleUnetCond(**ddpm_params).to(device)
elif ddpm_arch == "fno2d":
    model = FNO2D_grid_tembedding_cond(**ddpm_params).to(device)
    model.gridx = xs.to(device)
    model.gridy = ys.to(device)
    
# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Test model with a small batch
t = torch.randint(0, timesteps, (batch_size,), device=device).long()
data = truth_train
cond_data = train_pred_sparse
_ = model(data[:batch_size], cond_data[:batch_size], t)

hyperparam_dict["epochs_run"] = 0

##############
## TRAINING ##
##############

if True:
    printlog(f"Training {model_name}...")
    
    # Initialize noise scheduler
    if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

    # Plot alphas_cumprod for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(timesteps), c2n(alphas_cumprod), label='alphas_cumprod')
    plt.xlabel('Timesteps')
    plt.ylabel('Alphas Cumulative Product')
    plt.title(f'Alphas Cumulative Product over Timesteps\nbeta_start: {beta_start}, beta_end: {beta_end}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{model_dir}/alphas_cumprod.png", dpi=200)
    plt.close()

    # Initialize tracking variables
    if "epochs_run" not in hyperparam_dict:
        hyperparam_dict["epochs_run"] = 0

    loss_batch = []
    loss_epoch = []
    ibatch = 0
    
    # Training loop
    for epoch in range(epochs):
        printbatch = 0
        for batch_num in range(0, data.shape[0], batch_size):
            # Get batch data
            data_batch = data[batch_num:batch_num+batch_size].to(device)
            cond_batch = cond_data[batch_num:batch_num+batch_size].to(device)
            batch_size_actual = data_batch.shape[0]

            if train_type == "noise":
                # Sample random timesteps
                t = torch.randint(0, timesteps, (batch_size_actual,), device=device)

                # Add noise to data
                noise = torch.randn_like(data_batch)
                noisy_data = torch.sqrt(alphas_cumprod[t].view(-1, 1, 1, 1)) * data_batch + \
                             torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1)) * noise

                # Train model to predict noise
                optimizer.zero_grad()
                predicted_noise = model(noisy_data, cond_batch, t)
                
                # Select loss function based on batch number
                if ibatch <= loss_function_start_batch or loss_function_start_batch == -1:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](predicted_noise.permute((0, 2, 3, 1)), 
                                                   noise.permute((0, 2, 3, 1)), 
                                                   **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_function](predicted_noise.permute((0, 2, 3, 1)), 
                                                        noise.permute((0, 2, 3, 1)), 
                                                        **loss_args_end)

                loss.backward()
                optimizer.step()

            elif train_type == "tauFull":
                # Full timestep sampling from timesteps to 0
                tau0 = torch.zeros(batch_size_actual, device=device).long()
                tau1 = torch.full((batch_size_actual,), timesteps-1, device=device).long()
                rev_steps = timesteps

                # Add noise for tau0 and tau1 timesteps
                noise = torch.randn_like(data_batch)
                data_noise_rev = torch.sqrt(alphas_cumprod[tau0].view(-1, 1, 1, 1)) * data_batch + \
                                 torch.sqrt(1 - alphas_cumprod[tau0].view(-1, 1, 1, 1)) * noise
                data_noise = torch.sqrt(alphas_cumprod[tau1].view(-1, 1, 1, 1)) * data_batch + \
                             torch.sqrt(1 - alphas_cumprod[tau1].view(-1, 1, 1, 1)) * noise

                # Train model through reverse process
                optimizer.zero_grad()
                x = data_noise

                for tr in range(rev_steps):
                    tau = tau1 - tr  # Reverse from last timestep
                    predicted_noise = model(x, tau)
                    alpha_t = alphas[tau].view(-1, 1, 1, 1)
                    alpha_bar_t = alphas_cumprod[tau].view(-1, 1, 1, 1)
                    x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)
                    
                    # Add noise only if tau > 0
                    teff = torch.max(torch.ones(tau.shape, device=device), tau).long()
                    beta_t = betas[teff].view(-1, 1, 1, 1)
                    step_add_noise = (tau.view(-1, 1, 1, 1) > 0).int()
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x, requires_grad=True) * step_add_noise

                # Select loss function based on batch number
                if ibatch < loss_function_start_batch:
                    loss_use = loss_function_start
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0, 2, 3, 1)), 
                                                   data_noise_rev.permute((0, 2, 3, 1)), 
                                                   **loss_args_start)
                else:
                    loss_use = loss_function
                    loss = LOSS_FUNCTIONS[loss_use](x.permute((0, 2, 3, 1)), 
                                                   data_noise_rev.permute((0, 2, 3, 1)), 
                                                   **loss_args_end)

                loss.backward()
                optimizer.step()

            # Track losses
            loss_batch.append([ibatch, loss.item()])

            # Print progress
            if ibatch >= printbatch:
                printlog(f"Epoch [{epoch+1}/{epochs}], ibatch {ibatch+1}, loss_use: {loss_use}, Loss: {loss.item():.8f}")
                printbatch = ibatch + 10
            
            ibatch += 1

        # Track epoch loss
        loss_epoch.append([ibatch, loss.item()])

        # Plot training progress
        loss_batch_arr = np.array(loss_batch)
        loss_epoch_arr = np.array(loss_epoch)
        plt.plot(loss_batch_arr[:,0], loss_batch_arr[:,1], color="blue", label="batch loss", alpha=0.5)
        plt.scatter(loss_epoch_arr[:,0], loss_epoch_arr[:,1], color="red", label="epoch loss")
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
        plt.savefig(f"{model_dir}/loss_batch_epoch.png", dpi=200)
        plt.close()

        # Save model and update config
        torch.save(model.state_dict(), f"{model_dir}/{model_name}_epoch-{epoch+1}.pt")
        hyperparam_dict["epochs_run"] += 1
        with open(f"{model_dir}/config.yml", 'w') as h:
            yaml.dump(hyperparam_dict, h, default_flow_style=False)

else:
    # Code for loading a pre-trained model (not used in current run)
    model_name = "ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79"
    model_loc = "/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80_epoch-79.pt"
    model.load_state_dict(torch.load(model_loc))

    # Load config for pre-trained model
    config_path = "/glade/derecho/scratch/llupinji/diffusion_qgm_outputs/models_ddpm_specLoss/ddpm_arch-unet_time-2025-02-21-01-25_timesteps-1000_epochs-80/config.yml"
    with open(config_path, 'r') as h:
        hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

    # Extract parameters from loaded config
    timesteps = hyperparam_dict["timesteps"]
    beta_start = hyperparam_dict["beta_start"]
    beta_end = hyperparam_dict["beta_end"]
    batch_size = hyperparam_dict["batch_size"]
    epochs = hyperparam_dict["epochs"]
    beta_scheduler = hyperparam_dict["beta_scheduler"]
    epochs_run = hyperparam_dict["epochs_run"]

    # Initialize noise scheduler
    if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)

#############
## TESTING ##
#############

# Setup for sampling and evaluation
seeds = 200  # Number of samples to generate
stausteps = 0
ftausteps = timesteps
timesteps_rev_list = np.arange(stausteps, ftausteps)
its = np.arange(len(timesteps_rev_list))
its_timesteps = np.array(list(zip(its, reversed(timesteps_rev_list))))
num_rev_steps = len(its_timesteps)

# Select timesteps for plotting
idx_skip = int(len(timesteps_rev_list)/40)
its_timesteps_plot_og = its_timesteps[::idx_skip]
its_timesteps_plot = np.concatenate((its_timesteps_plot_og, its_timesteps[-idx_skip:]), axis=0)
its_timesteps_plot2 = its_timesteps_plot_og
itrevfinal = np.where(its_timesteps[:,1] == 0)[0][0]

# Prepare for inference
model.eval()
probe_data = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny))
probe_noise = torch.empty((seeds, len(timesteps_rev_list), 1, numchannels, Nx, Ny))
seeds_idx = np.arange(seeds)

# Load test data
test_input_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_input_sparse.pkl"
test_pred_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_pred_sparse.pkl"
truth_test_loc = "/glade/derecho/scratch/llupinji/turb2d_data/MSEch_FNO99S_EP200/test_truth.pkl"

with open(test_input_loc, "rb") as f:
    test_input_sparse = pickle.load(f)

with open(test_pred_loc, "rb") as f:
    test_pred_sparse = pickle.load(f)

with open(truth_test_loc, "rb") as f:
    truth_test = pickle.load(f)

# Fix bug with code saving two channels
test_pred_sparse = test_pred_sparse[:,:,:,[0]]

# Permute dimensions
test_input_sparse = test_input_sparse.permute((0, 3, 1, 2))
test_pred_sparse = test_pred_sparse.permute((0, 3, 1, 2))
truth_test = truth_test.permute((0, 3, 1, 2))

data_test = test_pred_sparse
cond_data_test = test_pred_sparse

# Run inference
with torch.no_grad():
    for s in range(seeds):
        printlog(f"Seed: {s+1}/{seeds}")
        
        # Add noise to test data
        noise = torch.randn_like(data_test[[s]])
        noisy_data = torch.sqrt(alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * data_test[[s]] + \
                     torch.sqrt(1 - alphas_cumprod[timesteps-1].view(-1, 1, 1, 1)) * noise
        x = noisy_data

        # Reverse diffusion process
        for it, t in its_timesteps:
            timestep = torch.tensor([t], device=device)
            predicted_noise = model(x, cond_data_test[[s]], timestep)
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

            # Add noise if using linear scheduler and not at final step
            if beta_scheduler == "linear" and t > stausteps:
                teff = max(1, t)
                beta_t = betas[teff]
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)

            # Store results
            probe_data[s, it] = x
            probe_noise[s, it] = predicted_noise

# Prepare data for analysis
probe_pred_data = probe_data.permute(0, 1, 2, 3, 5, 4)
probe_pred_noise = probe_noise.permute(0, 1, 2, 3, 5, 4)

train_input_sparse_use = test_input_sparse[seeds_idx].permute(0, 1, 3, 2)
train_pred_sparse_use = test_pred_sparse[seeds_idx].permute(0, 1, 3, 2)
train_truth_use = truth_test[seeds_idx].permute(0, 1, 3, 2)

# Create output directories
path_outputs_model = f"{output_dir}/{model_name}"
if not os.path.exists(path_outputs_model):
    printlog(f"Creating directory: {path_outputs_model}")
    os.makedirs(path_outputs_model)
    
path_outputs_model_timesteps = f"{path_outputs_model}/fno_ddpm_timesteps"
if not os.path.exists(path_outputs_model_timesteps):
    printlog(f"Creating directory: {path_outputs_model_timesteps}")
    os.makedirs(path_outputs_model_timesteps)

chs = ["vorticity"]
yscales = ["linear", "log"]

true_sparse_pred_diff = f"{path_outputs_model_timesteps}/true_sparse_pred_diff"
if not os.path.exists(true_sparse_pred_diff):
    os.makedirs(true_sparse_pred_diff)

# Get timesteps for visualization
itau1, tau1 = its_timesteps[0]
itau2, tau2 = its_timesteps[-10]
itau3, tau3 = its_timesteps[-1]

# Generate comparison plots
for istep, s in enumerate(seeds_idx, 0):
    print(f"Plotting timestep: {istep}")
    str_step = f"{istep:06d}"

    fig, axs = plt.subplots(1, 6, figsize=(18, 5))
    axs[0].imshow(train_input_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[0].set_title("Sparse 99%")
    axs[1].imshow(train_pred_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[1].set_title("FNO on Sparse")
    axs[2].imshow(probe_pred_data[s, 0, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[2].set_title(r"FNO+DDPM $\tau$ = %i" % tau1)
    axs[3].imshow(probe_pred_data[s, -10, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[3].set_title(r"FNO+DDPM $\tau$ = %i" % tau2)
    axs[4].imshow(probe_pred_data[s, -1, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[4].set_title(r"FNO+DDPM $\tau$ = %i" % tau3)
    axs[5].imshow(train_truth_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[5].set_title("Truth")

    plt.suptitle(f"Comparison for Seed {s}\n full tau, conditioning/{timesteps}")
    plt.tight_layout()
    plt.savefig(f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}.png", dpi=300)
    plt.close()

# Create list of image paths for video creation
png_locs_txt = ""
for istep, s in enumerate(seeds_idx, 0):
    str_step = f"{istep:06d}"
    png_loc = f"{true_sparse_pred_diff}/true_sparse_pred_diff_timestep-{str_step}.png"
    png_locs_txt += f"{png_loc}\n"

true_sparse_pred_diff_txt_loc = f"{path_outputs_model_timesteps}/true_sparse_pred_diff_timestep.txt"    
with open(true_sparse_pred_diff_txt_loc, "w") as f:
    f.write(png_locs_txt)

# Create video from images
os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {path_outputs_model_timesteps}/true_sparse_pred_diff.mp4')

# Generate simplified comparison plots (FNO, DDPM, Truth)
for istep, s in enumerate(seeds_idx, 0):
    print(f"Plotting timestep: {istep}")
    str_step = f"{istep:06d}"

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(train_pred_sparse_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[0].set_title("FNO on Sparse")
    axs[1].imshow(probe_pred_data[s, -1, 0, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[1].set_title(r"FNO+DDPM $\tau$ = %i" % tau3)
    axs[2].imshow(train_truth_use[s, 0, :, :].cpu().numpy(), cmap='coolwarm')
    axs[2].set_title("Truth")

    plt.suptitle(f"Comparison for Seed {s}\n full tau, conditioning/{timesteps}")
    plt.tight_layout()
    plt.savefig(f"{true_sparse_pred_diff}/true_pred_timestep-{str_step}.png", dpi=300)
    plt.close()

# Create video from simplified comparison
os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {true_sparse_pred_diff}/true_sparse_pred_diff_timestep-%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {path_outputs_model_timesteps}/true_pred_timestep.mp4')

# Helper function for FFT analysis
def compute_y_avg_fft(data):
    """Compute FFT along y-axis and average the magnitudes"""
    data_fft = np.fft.rfft(data, axis=1)
    return np.mean(np.abs(data_fft), axis=0)

# Prepare data for FFT analysis
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

# Plot FFT comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual FFTs with transparency
for data_array, (label, color) in zip(arrays_to_plot, labels.items()):
    for s in range(data_array.shape[0]):
        y_avg_fft = compute_y_avg_fft(data_array[s])
        ks = np.arange(y_avg_fft.shape[0])
        ax.plot(ks[2:-1], y_avg_fft[2:-1], color=color, alpha=0.3)

# Add legend entries
for _, (label, color) in zip(arrays_to_plot, labels.items()):
    ax.plot([], [], label=label, color=color, alpha=1)

ax.set_xlabel('k')
ax.set_ylabel('Amplitude')
ax.set_title('Vertically Average Spectrum')
ax.legend()
ax.grid(alpha=0.3)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{path_outputs_model_timesteps}/y_avg_fft_comparison.png", dpi=300)
plt.close()

# Generate timestep visualization for selected seeds
for s in seeds_idx[:2]:  # Only use first two seeds for visualization
    plot_dir = f"{path_outputs_model_timesteps}/s-{s}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for it, t in its_timesteps_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        # Get data limits for colorbar
        data = probe_data[s, it, 0, 0]
        vmin = torch.min(data).cpu().numpy()
        vmax = torch.max(data).cpu().numpy()
        
        im = ax.imshow(data.cpu().numpy(), cmap='coolwarm')
        ax.set_title(f"{chs[0]}")
        cbar = fig.colorbar(im, orientation='vertical', ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)

        plt.suptitle(f"{model_name}\nrev timestep: {t}")
        plt.savefig(f"{plot_dir}/probe_revStep_s-{s}_t-{pthstr(t)}.png")
        plt.close()

# Spectrum plots of the probe_data
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    startk = 2

    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1)

    ks = np.arange(true_data_fft_mean.shape[1])
    ax.plot([], [], color="black", linewidth=2, label="actual")
    ax.plot([], [], color="black", linewidth=2, label="true final reverse", linestyle="--")
    
    # Plot FFT for each timestep
    for it, t in its_timesteps_plot:
        # Calculate FFT of probe data at this timestep
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim=1)

        # Add legend entry for selected timesteps
        if (it % (len(its_timesteps)//3)) == 0:
            ax.plot([], [], color=cm.rainbow(it/its_timesteps.shape[0]), label=f"reverse t: {t}")

        # Plot FFT
        ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')
    
    # Plot true data FFT
    ax.plot(ks[startk:], c2n(true_data_fft_mean[0, startk:]), color="black", linewidth=3)

    # Plot final reverse timestep FFT
    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis=0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim=1)
    ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:]), color="black", linewidth=3, linestyle="--")

    ax.set_title(f"{chs[0]}")
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.suptitle(f"Probe data FFT\nReverse timesteps from {ftausteps} to {stausteps}")
    plt.tight_layout()
    
    # Save with different y-scales
    for yscale in yscales:
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrum_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

    plt.close()

# Spectrum difference plots
yscales_diff = ["linear", "symlog"]
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    startk = 2

    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    
    # Plot FFT difference for each timestep
    for it, t in its_timesteps_plot:
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
        
        # Plot difference between probe and true data FFT
        ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:] - true_data_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)
    
    # Plot final timestep difference
    probe_data_fft = rfft_abs_mirror_torch(probe_data[:, itrevfinal, 0], axis=3).mean(axis=0)
    probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
    ax.plot(ks[startk:], c2n(probe_data_fft_mean[0, startk:] - true_data_fft_mean[0, startk:]), 
            color="black", linewidth=3, linestyle="--")

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    ax.set_title(f"{chs[0]}")
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)

    plt.suptitle(f"Probe data FFT, difference from true\nReverse timesteps from {ftausteps} to {stausteps}")

    # Save with different y-scales
    for yscale in yscales_diff:
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumDiff_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)
            
    plt.close()

# Wave number error analysis
for revfinal in [len(its_timesteps)]:
    # Calculate FFT of true data
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0).to(device)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1).to(device)

    ks = np.arange(true_data_fft_mean.shape[1])
    startk = 2
    ks_keep = ks[startk:]

    # Initialize array to store error for each channel, k, and timestep
    error_by_k_timestep = np.zeros((len(chs), len(ks), len(its_timesteps)))

    # Calculate error for each timestep
    for it, t in its_timesteps:
        probe_data_fft = rfft_abs_mirror_torch(probe_data[:, it, 0], axis=3).mean(axis=0)
        probe_data_fft_mean = torch.mean(probe_data_fft, dim=1).to(device)
        error_by_k_timestep[0, :, it] = c2n((probe_data_fft_mean[0] - true_data_fft_mean[0, :]).abs())
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot error for each k value across timesteps
    for ich, ch in enumerate(chs):
        for k in ks_keep:
            ax.plot(its, error_by_k_timestep[ich, k, its], 
                    color=cm.rainbow_r(k/ks[-1]), alpha=0.5)

    # Add colorbar for k values
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow_r'), ax=ax)
    numticks = 11
    kticks_use = np.linspace(ks[-1], ks[0], num=numticks).astype(int)
    kticks_use_norm = (kticks_use)/ks[-1]
    cbar.set_ticks(kticks_use_norm)
    cbar.set_ticklabels([f"{k}" for k in kticks_use])
    cbar.set_label('Wave number k')

    ax.set_title(f"{chs[0]}")
    ax.set_xlabel('1 - tau (inverse diffusion step)')
    ax.set_ylabel('Amplitude Error')
    ax.grid(alpha=0.3)

    plt.suptitle(f"Probe data FFT, difference from true vs timesteps, for each k")

    # Save with different y-scales
    for yscale in yscales_diff:
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/kDiff_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)
            
    plt.close()

# Noise spectrum analysis
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    startk = 2

    # Calculate FFT of true data for reference
    true_data_fft = rfft_abs_mirror_torch(train_truth_use, axis=3).mean(axis=0)
    true_data_fft_mean = torch.mean(true_data_fft, dim=1)
    ks = np.arange(true_data_fft_mean.shape[1])
    
    # Plot noise spectrum for each timestep
    for it, t in its_timesteps_plot:
        noise_fft = rfft_abs_mirror_torch(probe_noise[:, it, 0], axis=3).mean(axis=0)
        noise_fft_mean = torch.mean(noise_fft, dim=1)

        # Add legend entry for selected timesteps
        if (it % (len(its_timesteps)//3)) == 0:
            ax.plot([], [], color=cm.rainbow(it/its_timesteps.shape[0]), label=f"reverse t: {t}")

        # Plot noise spectrum
        ax.plot(ks[startk:], c2n(noise_fft_mean[0, startk:]), 
                color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    # Plot final timestep noise spectrum
    final_noise_fft = rfft_abs_mirror_torch(probe_noise[:, itrevfinal, 0], axis=3).mean(axis=0)
    final_noise_fft_mean = torch.mean(final_noise_fft, dim=1)

    ax.set_title(f"{chs[0]}")
    ax.legend()
    ax.set_xlabel('k')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"Probe data noise spectrum\nReverse timesteps from {ftausteps} to {stausteps}")
    plt.tight_layout()

    # Save with different y-scales
    for yscale in yscales:
        ax.set_yscale(yscale)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/spectrumNoise_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

    plt.close()

# Noise distribution analysis
for revfinal in [len(its_timesteps)]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    bins = 500
    pdf_range = (-4, 4)
    
    # Plot noise distribution for each timestep
    for it, t in its_timesteps_plot2:
        for ich, ch in enumerate(chs):
            # Flatten noise data and compute histogram
            noise_flat = c2n(probe_noise[:, it, 0, ich]).flatten()
            hist, bin_edges = np.histogram(noise_flat, bins=bins, range=pdf_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            hist = hist/(hist.sum()*bin_width)  # normalize
            ax.plot(bin_centers, hist, color=cm.rainbow(it/its_timesteps.shape[0]), alpha=0.5)

    # Plot standard normal distribution for comparison
    x = np.linspace(pdf_range[0], pdf_range[1], bins)
    pdf = norm.pdf(x, loc=0, scale=1)
    ax.plot(x, pdf, color="black", alpha=1.0, label="Gaussian")

    # Add colorbar for timesteps
    cbar = fig.colorbar(cm.ScalarMappable(cmap='rainbow'), ax=ax)
    numticks = 11
    cbar.set_ticks(np.linspace(0, 1, num=numticks))
    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(ftausteps, stausteps, num=numticks)])
    cbar.set_label('Reverse timestep')

    ax.set_title(f"{chs[0]}")
    ax.set_xlim(pdf_range)
    ax.legend()
    ax.set_xlabel('Noise')
    ax.set_ylabel('PDF')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"Noise distribution\nReverse timesteps from {ftausteps} to {stausteps}")
    
    # Save with different y-scales
    for yscale in ["linear", "log"]:
        ax.set_yscale(yscale)
        plt.tight_layout()
        plt.savefig(f"{path_outputs_model}/histNoise_revFinal-{revfinal}_yscale-{yscale}.png", dpi=200)

    plt.close()

## finding the pdf of the diffusion process noise 
printlog(f"Model: {model_name} pipeline finished.")