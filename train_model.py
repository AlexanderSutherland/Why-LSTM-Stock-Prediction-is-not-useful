import torch
import torch.nn as nn
import torch.optim as optim
import Models/cnn_lstm

def train_model(model, train_loader, criterion = nn.MSELoss, optimizer = 'TO DO', epochs = 10000):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')
