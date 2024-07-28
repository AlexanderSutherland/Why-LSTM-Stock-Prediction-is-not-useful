import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import pandas as pd
from cnn_lstm import CNN_LSTM
from data_util import DataUtil
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

def main():
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    
    # Hyperparmeters
    batch_size = 32
    learning_rate = 0.0001
    criterion = nn.MSELoss()
    optimzer_type = optim.Adam
    epochs = 10000
    
    # Date Range
    start_date = dt.datetime(2014, 1, 1)
    end_date = dt.datetime(2016, 1, 1)
    
    # Split ratio for Train and Test
    split_ratio = 0.8
    
    # Generate the data sets for training and testing (Called loaders)
    train_loader, test_loader = generate_data_loaders(batch_size = batch_size,
                                                      start_date = start_date,
                                                      end_date = end_date,
                                                      split_ratio = split_ratio)
    
    # Train the model
    model = train_model(train_loader,
                        model_type = CNN_LSTM,
                        criterion = criterion,
                        optimizer_type = optimzer_type,
                        epochs = epochs,
                        learning_rate = learning_rate,
                        load_model = None)
    
    # Test model
    test_model(test_loader, model) 

def generate_data_loaders(batch_size=32, 
                          start_date = dt.datetime(2014, 1, 1), 
                          end_date = dt.datetime(2016, 1, 1), 
                          split_ratio = 0.8):
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
    
    
    # Set date range
    date_range = pd.date_range(start=start_date, end = end_date)
    
    # Grab X_data and Y_data (Y is one day ahead of X)
    data_util = DataUtil()
    X_data = data_util.grab_data_combined(dates=date_range)[:-1]
    Y_data = data_util.grab_SMH_adj_close(dates=date_range)[1:]
    
    # Convert data to float32
    X_data = X_data.float()
    Y_data = Y_data.float()
    
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
                              batch_size = batch_size, 
                              shuffle = False)
    test_loader = DataLoader(test_dataset, 
                             batch_size = batch_size, 
                             shuffle = False)
    
    return train_loader, test_loader


def train_model(train_loader, model_type = CNN_LSTM, criterion = nn.MSELoss(), optimizer_type=optim.Adam, epochs = 10000, learning_rate = 0.001, load_model = None):
    """
    Trains the given model for stock price prediction.

    Args:
        model_type (class): The model class to be used for training.
        criterion (class, optional): The loss function to use. Defaults to nn.MSELoss.
        optimizer_type (class, optional): The optimizer class to use. Defaults to optim.Adam.
        epochs (int, optional): Number of epochs for training. Defaults to 10000.
        load_model (str, optional): Path to a pre-trained model to load. Defaults to None.

    Returns:
        tuple: The trained model and the test DataLoader.
    """
    
    
    # Initiate Model
    if model_type == CNN_LSTM:
        # Model initialization parameters
        cnn_in_channels = 18
        cnn_out_channels = 100
        lstm_hidden_size = 200
        lstm_num_layers = 4
        output_size = 1
        model = model_type(cnn_in_channels,
                           cnn_out_channels, 
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
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            print('Epoch', epoch)
            print('x batch:', x_batch)
            print('y batch:', y_batch)
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
    
    return model

# Test model on test data
def test_model(test_loader, model, criterion=nn.MSELoss()):
    """
    Tests the given model on the test data.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_loader (torch.utils.data.DataLoader): The DataLoader containing test data.
        criterion (class, optional): The loss function to use. Defaults to nn.MSELoss.

    Returns:
        None
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')
    
    return model


if __name__ == "__main__":
    main()
