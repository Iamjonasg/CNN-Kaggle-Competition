#Jonas Gabirot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 40
batch_size = 4
learning_rate = 0.001



df_result = pd.read_csv('train_result.csv')
df_result = df_result.drop(columns=["Index"])
df_result = df_result.to_numpy()
df = pd.read_csv('train.csv')
df = df.to_numpy()
df = np.delete(df,1568 ,axis=1)
test = pd.read_csv("test.csv")
test = test.to_numpy()
test = np.delete(test,1568 ,axis=1)
test = test.reshape(10000,1,56,28) 

df_result = df_result.flatten()

df = df.reshape(50000,1,56,28) 




tensor_test = torch.Tensor(test)
tensor_df = torch.Tensor(df)
tensor_result = torch.Tensor(df_result)

train_dataset = TensorDataset(tensor_df,tensor_result)
test_dataset = TensorDataset(tensor_df,tensor_result)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)





class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))  
        x = x.view(-1, 704)            # -> n, 704
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 19
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

model.eval()
pred=[]
for image in tensor_test:
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    pred.append(predicted.item())



pd.DataFrame(pred).to_csv("results.csv")

