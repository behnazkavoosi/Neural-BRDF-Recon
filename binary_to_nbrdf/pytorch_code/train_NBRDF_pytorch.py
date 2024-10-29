import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset

import common
import coords
import fastmerl

torch.set_default_dtype(torch.float32)

# Variable definitions
Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']
batch_size = 512
epochs = 100
learning_rate = 5e-4
np.random.seed(0)
torch.manual_seed(0)

# Use CUDA if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss function
def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log1p(y_true) - torch.log1p(y_pred)))

# Define the neural network architecture
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=6, out_features=21, bias=True)
        self.fc2 = torch.nn.Linear(in_features=21, out_features=21, bias=True)
        self.fc3 = torch.nn.Linear(in_features=21, out_features=3, bias=True)

        # Initialize weights
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

        self.fc1.weight = torch.nn.Parameter(torch.zeros((6, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True)
        self.fc2.weight = torch.nn.Parameter(torch.zeros((21, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True)
        self.fc3.weight = torch.nn.Parameter(torch.zeros((21, 3), dtype=torch.float32).uniform_(-0.05, 0.05).T, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(torch.exp(self.fc3(x)) - 1.0)  # Additional ReLU
        return x

# Dataset class for MERL BRDF data
class MerlDataset(Dataset):
    def __init__(self, merlPath, batchsize):
        super(MerlDataset, self).__init__()
        self.bs = batchsize
        self.BRDF = fastmerl.Merl(merlPath)

        self.reflectance_train = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.8)
        self.reflectance_test = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.2)

        self.train_samples = torch.tensor(self.reflectance_train[Xvars].values, dtype=torch.float32, device=device)
        self.train_gt = torch.tensor(self.reflectance_train[Yvars].values, dtype=torch.float32, device=device)

        self.test_samples = torch.tensor(self.reflectance_test[Xvars].values, dtype=torch.float32, device=device)
        self.test_gt = torch.tensor(self.reflectance_test[Yvars].values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.train_samples.shape[0]

    def get_trainbatch(self, idx):
        return self.train_samples[idx:idx + self.bs, :], self.train_gt[idx:idx + self.bs, :]

    def shuffle(self):
        r = torch.randperm(self.train_samples.shape[0])
        self.train_samples = self.train_samples[r, :]
        self.train_gt = self.train_gt[r, :]

# BRDF to RGB conversion
def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
          torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb

# BRDF value extraction
def brdf_values(rvectors, brdf=None):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    else:
        raise NotImplementedError("Something went really wrong.")

    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    return brdf_arr

def generate_nn_datasets(brdf, nsamples=800000, pct=0.8):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])

    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf)

    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    df = df.drop(df[df['brdf_r'] < 0].index)
    return df


def reconstruct_brdf(model, sampling_theta_h, sampling_theta_d, sampling_phi_d):

    theta_h_values = np.linspace(0, np.pi / 2, sampling_theta_h)
    theta_d_values = np.linspace(0, np.pi / 2, sampling_theta_d)
    phi_d_values = np.linspace(0, np.pi, sampling_phi_d)

    rvectors = []
    for theta_h in theta_h_values:
        theta_h = (theta_h * theta_h) / (np.pi / 2)
        for theta_d in theta_d_values:
            for phi_d in phi_d_values:

                phi_d = common.normalize_phid(phi_d)

                hx = np.sin(theta_h)
                hy = 0.0  
                hz = np.cos(theta_h)
                dx = np.sin(theta_d) * np.cos(phi_d)
                dy = np.sin(theta_d) * np.sin(phi_d)
                dz = np.cos(theta_d)
                rvectors.append([hx, hy, hz, dx, dy, dz])

    rvectors = np.array(rvectors)

    rvectors_tensor = torch.tensor(rvectors, dtype=torch.float32, device=device)

    predicted_brdf = model(rvectors_tensor).to(device)

    rgb_values = brdf_to_rgb(rvectors_tensor, predicted_brdf)

    merl_instance = fastmerl.Merl('./data/green-metallic-paint2.binary') 
    merl_instance.from_array(rgb_values.detach().cpu().numpy())
    merl_instance.write_merl_file('/mnt/d/Projects/Paper05/brdf-rendering/brdfs/reconstructed_brdf.binary') 

    return rgb_values.detach().cpu().numpy()

# Instantiate the model and optimizer
model = MLP().to(device)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-15, weight_decay=0.0, amsgrad=False)

# Read MERL BRDF
name = 'green-metallic-paint2'
merlpath = f'data/{name}.binary'
merl = MerlDataset(merlpath, batchsize=batch_size)
"""
# Training Loop
start = time.time()
train_losses = []  # helper, for plotting

for epoch in range(epochs):
    losses = []  # batch-losses per epoch
    merl.shuffle()
    epochStart = time.time()
    num_batches = int(merl.train_samples.shape[0] / batch_size)

    # Iterate over batches
    for k in range(num_batches):
        optim.zero_grad()

        # Get batch from MERL data, feed into model to get prediction
        mlp_input, groundTruth = merl.get_trainbatch(k * merl.bs)
        output = model(mlp_input).to(device)

        # Convert to RGB data
        rgb_pred = brdf_to_rgb(mlp_input, output)
        rgb_true = brdf_to_rgb(mlp_input, groundTruth)

        loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    epoch_loss = sum(losses) / len(losses)
    train_losses.append(epoch_loss)
    print("Epoch {}/{} - Loss {:.7f} - Time: {:.4f}s".format(epoch + 1, epochs, epoch_loss, time.time() - epochStart))

print("Trained for {} epochs in {} seconds".format(epochs, time.time() - start))

# Save loss plot
plt.plot(train_losses)
plt.savefig("loss.png")

# Save model weights
for el in model.named_parameters():
    param_name = el[0]   # either fc1.bias or fc1.weight
    weights = el[1]
    segs = param_name.split('.')
    if segs[-1] == 'weight':
        param_name = segs[0]
    else:
        param_name = segs[0].replace('fc', 'b')

    filename = '{}_{}.npy'.format(name, param_name)
    curr_weight = weights.detach().cpu().numpy().T  # transpose because Mitsuba code was developed for TF convention
    np.save(filename, curr_weight)
"""

weight_names = ['fc1', 'fc2', 'fc3']  
bias_names = ['b1', 'b2', 'b3']  

for weight_name in weight_names:
    
    filename = f"{name}_{weight_name}.npy"
    if os.path.exists(filename):
        weights = np.load(filename)
        model.state_dict()[f'{weight_name}.weight'].copy_(torch.tensor(weights.T, dtype=torch.float32))  
    else:
        print(f"Warning: Weight file {filename} not found!")

for i, bias_name in enumerate(bias_names):
    filename = f"{name}_{bias_name}.npy" 
    if os.path.exists(filename):
        bias_weights = np.load(filename)
        model.state_dict()[f'fc{i+1}.bias'].copy_(torch.tensor(bias_weights.T, dtype=torch.float32))  
    else:
        print(f"Warning: Bias weight file {filename} not found!")

sampling_theta_h = 90  
sampling_theta_d = 90  
sampling_phi_d = 180   
reconstructed_brdf = reconstruct_brdf(model, sampling_theta_h, sampling_theta_d, sampling_phi_d)

print("Reconstruction completed and saved to 'reconstructed_brdf.binary'.")
