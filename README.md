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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

name = "MOHAMMAD FAIZAL SK"
register_number = "212223240092"

dataset1 = pd.read_csv('/content/EX01_DATASET - Sheet1.csv')
X = dataset1[['UNIT']].values
y = dataset1[['BILL']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class FAIZAL_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 7)
        self.fc3 = nn.Linear(7, 6)
        self.fc4 = nn.Linear(6, 1)
        self.history = {'loss': []}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

ai_brain = FAIZAL_MODEL()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):

        optimizer.zero_grad()

        output = ai_brain(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()


print(f'Prediction: {prediction}')
```
## Dataset Information
<img width="687" height="241" alt="image" src="https://github.com/user-attachments/assets/947aa0e9-251c-4fd5-aab0-e4803f0b3b56" />



## OUTPUT
<img width="817" height="235" alt="image" src="https://github.com/user-attachments/assets/76dede64-0f4e-4338-bf2b-3dd94669b340" />

### Training Loss Vs Iteration Plot

<img width="832" height="499" alt="image" src="https://github.com/user-attachments/assets/ecfe2689-dfba-4406-9dbe-08ec09989ca0" />




### New Sample Data Prediction
<img width="835" height="51" alt="image" src="https://github.com/user-attachments/assets/48ad6516-ad40-4fc3-9ab5-c4ecc81226df" />




## RESULT

The neural network regression model was successfully developed and trained.
