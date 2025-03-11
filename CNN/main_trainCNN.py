import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Loss_Spectrum import spectral_sqr_abs2

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
print("Startsparsity",u_train_m.shape, u_train_o.shape, u_test_m.shape, u_test_o.shape)

# Split training data into training and validation sets
total_train_samples = u_train_m.shape[0]
val_size = int(0.2 * total_train_samples)  # 20% for validation
train_size = total_train_samples - val_size

train_dataset, val_dataset = random_split(TensorDataset(u_train_m, u_train_o), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

if LossFunction == "MSE":
    criterion = nn.MSELoss()
elif LossFunction== "Spectrum_MSE":
    criterion= spectral_sqr_abs2

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping criteria
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
losslist = []
val_loss_list = []

# Training loop with early stopping
for epoch in range(EPOCH):  # High epoch count as formal; stops early if necessary
    model.train()
    epoch_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losslist.append(avg_epoch_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    print(f'Epoch: {epoch + 1}, Training Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODELNAME+'.pth')  # Save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

print("Finished Training or Stopped Early due to Non-Improvement")

loss_data = pd.DataFrame({
    'Epoch': range(1, len(losslist) + 1),
    'Training Loss': losslist,
    'Validation Loss': val_loss_list
})
loss_data.to_csv(MODELNAME+'.csv', index=False)


# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(losslist, label='Training Loss', marker='o')
plt.plot(val_loss_list, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss of '+ MODELNAME)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+'.png')
plt.show()

