# no relationship so isn't useful
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

df = pd.read_pickle('./data/MEGA_PICKLE/MEGA.pkl')

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self.hidden_layers.append(nn.Sigmoid())
            input_size = hidden_size
        self.output_layer = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

model = Net(660,[100,100,100,10],1)
# x = torch.randn(50,1)
# y = torch.sin(x)
x = torch.tensor(df.drop(columns=['next_1y_pct_change']).values, dtype=torch.float32)
y = torch.tensor(df[['next_1y_pct_change']].values, dtype=torch.float32)
y = torch.where(y > 0, torch.tensor(1.0), torch.tensor(0.0))

print(x)
print(y)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.11)

for epoch in range(318):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    # perform a backward pass (backpropagation)
    loss.backward()
    # Update the parameters
    optimizer.step()