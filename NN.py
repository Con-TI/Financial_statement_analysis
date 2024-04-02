# Works but slow and uses a lot of electricity so not gonna try running locally
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
df = pd.read_pickle('./data/MEGA_PICKLE/MEGAMEGA.pkl')
df_dropped = df.drop(columns=['next_1y_pct_change'])
print(df_dropped.shape)

# Check if any inf values exist in the DataFrame

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.5))
            input_size = hidden_size
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            absolute = torch.abs(x)
            max_value = torch.max(absolute)
            item = str(max_value.item())
            try:
                exponent_part = item.split('e+')[1]
                x = x / (10 ** float(exponent_part))
            except IndexError:
                pass
        x = self.output_layer(x)
        return x


model = Net(len(df_dropped.columns), [3000,1000,1000,50,100], 1).to(device)
for param in model.parameters():
    param.requires_grad = True

x = torch.tensor(df_dropped.values, dtype=torch.float32).to(device)
y = torch.tensor(df[['next_1y_pct_change']].values, dtype=torch.float32).to(device)
y[y > 0] = 1
y[y <= 0] = -1
# Create DataLoader for batching
dataset = TensorDataset(x, y)
batch_size = 64  # Adjust batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()
for epoch in range(1000):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(batch_x)
        # Compute Loss
        loss = criterion(outputs, batch_y)
        # Backward pass
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # Update params
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(dataset)
    print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss}')