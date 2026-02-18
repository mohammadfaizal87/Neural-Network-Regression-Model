# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement

Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships.
A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="1265" height="725" alt="image" src="https://github.com/user-attachments/assets/ef73cd9e-726a-48aa-ba36-c57481123895" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: MOHAMMAD FAIZAL SK
### Register Number: 212223240092
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("EX01_DATASET - Sheet1.csv")

X = data[['UNIT']].values
y = data[['BILL']].values

print("Dataset Information")
print(data)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,7)
        self.fc3 = nn.Linear(7,6)
        self.fc4 = nn.Linear(6,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

faizal_model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(faizal_model.parameters(), lr=0.01)

losses = []

for epoch in range(2000):
    optimizer.zero_grad()
    output = faizal_model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.title("Training Loss vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

sample_units = float(input("Enter Units to Predict Bill: "))
sample = [[sample_units]]

sample_scaled = scaler_X.transform(sample)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

with torch.no_grad():
    pred = faizal_model(sample_tensor)

predicted_bill = scaler_y.inverse_transform(pred.numpy())

print("Units:", sample_units)
print("Predicted Bill:", predicted_bill[0][0])
```
## Dataset Information
<img width="687" height="241" alt="image" src="https://github.com/user-attachments/assets/947aa0e9-251c-4fd5-aab0-e4803f0b3b56" />



## OUTPUT

### Training Loss Vs Iteration Plot

<img width="800" height="504" alt="image" src="https://github.com/user-attachments/assets/ca05e924-68b5-431d-94ef-45d419477e66" />



### New Sample Data Prediction
<img width="402" height="76" alt="image" src="https://github.com/user-attachments/assets/76b61d47-de08-4914-ad2a-99d5c2c79613" />



## RESULT

The neural network regression model was successfully developed and trained.
