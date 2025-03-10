import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)

LossFunction= "MSE"  # Loss funtion is either MSE or Spectrum
EPOCH=100
# LAMBDA=0.7
# MODELNAME=LossFunction+'CNN90S_EP'+str(EPOCH)+'Lambda'+str(LAMBDA)+'WaveLat'+str(50)
MODELNAME=LossFunction+'ch_CNN99S_EP'+str(EPOCH)
print("MODELNAME:"+MODELNAME)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
from load_data_CNN import get_dynamics_data
u_train_m, u_train_o, u_test_m, u_test_o = get_dynamics_data()

u_train_m = u_train_m.to(device)
u_train_o = u_train_o.to(device)
u_test_m = u_test_m.to(device)
u_test_o = u_test_o.to(device)
print("Testcheck 99 Sparsity",u_train_m.shape, u_train_o.shape, u_test_m.shape, u_test_o.shape)


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

model = CNNModel().to(device)
criterion = nn.MSELoss()


# Path where the model is saved
model_path = "/glade/derecho/scratch/nasefi/compressed_sensing/MSEch_CNN99S_EP100.pth"

# Load the saved model's state dict
model.load_state_dict(torch.load(model_path))
model.eval()


# Test the model
test_dataset = TensorDataset(u_test_m, u_test_o)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
total_loss = 0
all_outputs = []
all_labels = []

# Loop over all test data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_outputs.append(outputs)
        all_labels.append(labels)

average_loss = total_loss / len(test_loader)
sqrt_of_average_loss = math.sqrt(average_loss)

print(f"Average MSE Test Loss: {average_loss:.6f}")
print(f"Average RMSE Test Loss: {sqrt_of_average_loss:.6f}")
# Concatenate all batch outputs and labels
all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)


def plot_fixed_index(data_loader, model, device, index):
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if images.size(0) > index:  # Ensure the batch is large enough
                image = images[index].unsqueeze(0).to(device)
                label = labels[index].unsqueeze(0).to(device)
                output = model(image)
                imageT = image.transpose(2,3)
                labelT = label.transpose(2,3)
                outputT = output.transpose(2,3)

                input_vis = imageT.cpu().squeeze()[0]  # Visualize the first channel

                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                axes[0].imshow(input_vis, cmap='gray')
                axes[0].set_title('Input')
                axes[1].imshow(outputT.cpu().squeeze(), cmap='gray')
                axes[1].set_title('Predicted Output')
                axes[2].imshow(labelT.cpu().squeeze(), cmap='gray')
                axes[2].set_title('Actual Output')
                plt.savefig(MODELNAME+str(index)+'.png')  # Dynamic filename based on index
                plt.show()
                plt.close(fig)  # Close the figure to free up memory
                break  # Stop after the first batch
            else:
                print(f"Batch size is smaller than the specified index {index}.")
                break
# Usage examples:
plot_fixed_index(test_loader, model, device, index=1)  # Plot the first index of a batch
plot_fixed_index(test_loader, model, device, index=5) 
plot_fixed_index(test_loader, model, device, index=10)  
plot_fixed_index(test_loader, model, device, index=15)  
plot_fixed_index(test_loader, model, device, index=20) 

#Moein Code
def compute_spectrum_torch(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    return torch.abs(torch.fft.rfft(data_torch, dim=-1)).numpy()

def compute_mean_std_spectrum_torch(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=2))
    return magnitude_spectrum.mean(1).mean(0).numpy(), magnitude_spectrum.mean(1).std(0).numpy()

#  [batch, latitude, long]

def spectrum_rmse(pred_data, target_data):
    pred_spectrum, _ = compute_mean_std_spectrum_torch(pred_data)
    target_spectrum, _ = compute_mean_std_spectrum_torch(target_data)
    return np.sqrt(np.mean((pred_spectrum - target_spectrum) ** 2))

def compute_spectrum_niloo(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
    return magnitude_spectrum.mean(2).mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()


def compute_spectrum_c(data):
    data_torch = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data.float()
    magnitude_spectrum = torch.abs(torch.fft.rfft(data_torch, dim=1))
    return magnitude_spectrum.mean(0).numpy(), magnitude_spectrum.mean(0).std(0).numpy()

#  [batch, latitude, long]


# Assuming `outputs` and `originals` are numpy arrays with shape [3000, 1, 256, 256]
# I removed the channed dimension here. 

#one time step. 

print("Aggregated Outputs Shape:", all_outputs.shape)

print("Aggregated Outputs Shape:", all_outputs.shape)
print("Aggregated Originals Shape:", all_labels.shape)
# Assuming you have functions to compute spectrum defined
outputs_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_outputs[:,0].float().cpu())
originals_mean_spectrum, _ = compute_mean_std_spectrum_torch(all_labels[:,0].float().cpu())

# Initialize lists to store outputs and labels for all batches


# Plot the spectrum FFT

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"FFT.png")
plt.show()

outputs_mean_spectrum, _ = compute_spectrum_niloo(all_outputs[:,0].float().cpu())
originals_mean_spectrum, _ = compute_spectrum_niloo(all_labels[:,0].float().cpu())

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"niloo"+"FFT.png")
plt.show()

#3000, 256, 256 

outputs_mean_spectrum, _ = compute_spectrum_c(all_outputs[20,0].float().cpu())
originals_mean_spectrum, _ = compute_spectrum_c(all_labels[20,0].float().cpu())

plt.figure(figsize=(10, 5))
plt.loglog(outputs_mean_spectrum, color='blue', label='Mean Output Spectrum')
plt.loglog(originals_mean_spectrum, color='black', label='Mean Original Spectrum')
plt.title('Power Spectrum for '+MODELNAME)
plt.xlabel('Wavenumber')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+"conradnew20"+"FFT.png")
plt.show()

