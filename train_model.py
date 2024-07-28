import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import pandas as pd
from cnn_lstm import CNN_LSTM
from data_util import DataUtil
from torch.utils.data import DataLoader, TensorDataset


def generate_data_loaders(batch_size=32, 
                          start_date = dt.datetime(2014, 1, 1), 
                          end_date = dt.datetime(2016, 1, 1), 
                          split_ratio = 0.8):
    # Set date range
    date_range = pd.date_range(start=start_date, end = end_date)
    
    # Grab X_data and Y_data (Y is one day ahead of X)
    X_data = DataUtil.grab_data_combined(dates=date_range)[:-1]
    Y_data = DataUtil.grab_SMH_adj_close(dates=date_range)[1:]
    
    # Where to split X and Y
    split_index = int(split_ratio * len(X_data))
    
    # Create the train and test data sets
    x_train = X_data[:split_index]
    y_train = Y_data[:split_index]
    x_test = X_data[split_index:]
    y_test = Y_data[split_index:]
    

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader


def train_model(model_type = CNN_LSTM, criterion = nn.MSELoss(), optimizer_type=optim.Adam, epochs = 10000, load_model = None):
    
    # Initiate Model
    if model_type == CNN_LSTM:
        # Model initialization parameters
        cnn_out_channels = 1
        lstm_hidden_size = 1
        lstm_num_layers = 1
        output_size = 1
        model = model_type(cnn_out_channels, 
                           lstm_hidden_size, 
                           lstm_num_layers, 
                           output_size)
    else:
        # TO DO!!
        pass
    
    # Load in previous model
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    
    # Create Optimizer
    optimizer = optimizer_type(model.parameters(), lr=0.001)
    
    # Generate data
    train_loader, test_loader = generate_data_loaders()
    
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
    
    
    
    # Save off model updates
    save_model = input('Would you like to save the model? Type "YES" to confirm: ')
    if save_model.upper() == 'YES':
        torch.save(model.state_dict(), 'model.pth')
    
    return model, test_loader

def test_model(model, test_loader, criterion=nn.MSELoss()):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

def main():
    model, test_loader = train_model()
    test_model(model, test_loader)  

if __name__ == "__main__":
    # print("Executing as the main program")
    data_util = DataUtil()
    combined_data = data_util.grab_SMH_adj_close()
    print(combined_data.shape)
