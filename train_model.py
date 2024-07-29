import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import pandas as pd
from cnn_lstm import CNN_LSTM
from lstm import LSTM
from data_util import DataUtil
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    criterion = nn.MSELoss()
    optimizer_type = optim.Adam
    epochs = 200
    
    # Date Range
    start_date = dt.datetime(2014, 1, 1)
    end_date = dt.datetime(2016, 1, 1)
        
    # Split ratio for Train and Test
    split_ratio = 0.8
    
    # Past Days to include in current day:
    num_prev_days = 50
    
    # Generate the data sets for training and testing (Called loaders)
    train_loader, test_loader, x_train, y_train, x_test, y_test = generate_data_loaders(batch_size=batch_size,
                                                                                        start_date=start_date,
                                                                                        end_date=end_date,
                                                                                        split_ratio=split_ratio,
                                                                                        device=device,
                                                                                        num_prev_days = num_prev_days)
    
    # Train the model
    model = train_model(train_loader,
                        test_loader,
                        model_type=CNN_LSTM,
                        criterion=criterion,
                        optimizer_type=optimizer_type,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        load_model=None,
                        device=device,
                        num_prev_days = num_prev_days)
    
    # Test model
    test_model(test_loader, model, criterion, device) 
    
    # Plot Price Predictions:
    plot_results(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    

def generate_data_loaders(batch_size=32, 
                          start_date=dt.datetime(2014, 1, 1), 
                          end_date=dt.datetime(2016, 1, 1), 
                          split_ratio=0.8,
                          device='cpu',
                          num_prev_days = 0):
    """
    Grabs the data loaders given a date range.

    Args:
        batch_size (int, optional): Size of each data batch. Defaults to 32.
        start_date (datetime, optional): Start date of the data range. Defaults to January 1, 2014.
        end_date (datetime, optional): End date of the data range. Defaults to January 1, 2016.
        split_ratio (float, optional): Ratio to split the data into training and testing sets. Defaults to 0.8.

    Returns:
        tuple: A tuple containing training and test DataLoader instances.
    """
    
    # Grab X_data and Y_data
    data_util = DataUtil(start_date=start_date, end_date=end_date)
    X_data, _, _ = data_util.grab_data_combined(num_of_prev_days=num_prev_days)
    Y_data, _, _ = data_util.grab_SMH_adj_close()
    
    # Y is one day ahead of X
    X_data = X_data[:-1]
    Y_data = Y_data[1:]
    
    # Convert data to float32
    X_data = X_data.float()
    Y_data = Y_data.float()
    
    X_data = X_data.to(device=device)
    Y_data = Y_data.to(device=device)
    
    # Where to split X and Y
    split_index = int(split_ratio * len(X_data))
    
    # Create the train and test data sets
    x_train = X_data[:split_index]
    y_train = Y_data[:split_index]
    x_test = X_data[split_index:]
    y_test = Y_data[split_index:]
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, x_train, y_train, x_test, y_test


def train_model(train_loader, test_loader, model_type, criterion, optimizer_type, epochs, learning_rate, load_model, device, num_prev_days = 0):
    """
    Trains the given model for stock price prediction.

    Args:
        model_type (class): The model class to be used for training.
        criterion (class): The loss function to use.
        optimizer_type (class): The optimizer class to use.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        load_model (str, optional): Path to a pre-trained model to load. Defaults to None.
        device (str): Device to run the training on (cpu or cuda).

    Returns:
        torch.nn.Module: The trained model.
    """
    # Initiate Model
    if model_type == CNN_LSTM:
        # Model initialization parameters
        cnn_in_channels = 6 + 6 * num_prev_days
        cnn_out_channels = 100
        lstm_hidden_size = 128
        lstm_num_layers = 2
        output_size = 1
        model = model_type(cnn_in_channels,
                           cnn_out_channels, 
                           lstm_hidden_size, 
                           lstm_num_layers, 
                           output_size).to(device)
    elif model_type == LSTM:
        # TO DO FENG
        input_size = 6  # Number of features
        lstm_hidden_size = 128
        lstm_num_layers = 2
        output_size = 1
        model = LSTM(input_size,
                     lstm_hidden_size, 
                     lstm_num_layers, 
                     output_size).to(device)
    else:
        raise ValueError('No Model type given!')
    
    # Load previous model if provided
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    
    # Create Optimizer
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    test_loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss_history.append(total_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}')   
        
        _, test_avg_loss = test_model(test_loader, model, criterion, device)
        test_loss_history.append(test_avg_loss)
    
    plot_loss = input('Would you like to create a plot comparing the train and test datasets losses? Type "YES" to confirm: ')
    if plot_loss.upper() == 'YES':
        plt.plot(train_loss_history, label='Train')
        plt.plot(test_loss_history, label='Test')
        plt.legend()
        plt.title('Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.show()
    
    # Save model updates
    save_model = input('Would you like to save the model? Type "YES" to confirm: ')
    if save_model.upper() == 'YES':
        torch.save(model.state_dict(), 'model.pth')
    
    return model

# Test model on test data
def test_model(test_loader, model, criterion, device):
    """
    Tests the given model on the test data.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_loader (torch.utils.data.DataLoader): The DataLoader containing test data.
        criterion (class): The loss function to use.
        device (str): Device to run the testing on (cpu or cuda).

    Returns:
        tuple: The model and the average loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')
    
    return model, avg_loss

def plot_results(model, x_train, y_train, x_test, y_test):
    model.eval()
    with torch.no_grad():
        train_est = []
        test_est = []
        for idx in range(x_train.shape[0]):
            output = model(x_train[idx].unsqueeze(0))
            output = output.cpu()  # Move to CPU
            train_est.append(output[0][0].item())  # Convert to scalar and append
            print(f'[Train] Predicted: {output[0][0].item()}, Actual: {y_train[idx].item()}')
        plt.plot(train_est, label='Train Estimate')
        plt.plot(y_train.cpu(), label='Train Actual')  # Move y_train to CPU
        plt.legend()
        plt.title('Training Price Estimation')
        plt.ylabel('Price')
        plt.xlabel('Day')
        plt.show()
            
        for idx in range(x_test.shape[0]):
            output = model(x_test[idx].unsqueeze(0))
            output = output.cpu()  # Move to CPU
            test_est.append(output[0][0].item())  # Convert to scalar and append
            print(f'[Test] Predicted: {output[0][0].item()}, Actual: {y_test[idx].item()}')
        
        plt.plot(test_est, label='Test Estimate')
        plt.plot(y_test.cpu(), label='Test Actual')  # Move y_test to CPU
        plt.legend()
        plt.title('Test Price Estimation')
        plt.ylabel('Price')
        plt.xlabel('Day')
        plt.show()

if __name__ == "__main__":
    main()