import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
from cnn_lstm import CNN_LSTM
from lstm import LSTM
from data_util import DataUtil
from train_model import train_model, test_model
from plotter import plot_price_predictions, plot_loss_epoch

def main():
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer_type = optim.Adam
    epochs = 200
    
    # Date Range
    start_date = dt.datetime(2014, 1, 1)
    end_date = dt.datetime(2016, 1, 1)
        
    # Split ratio for Train and Test
    split_ratio = 0.8
    
    # Generate the data sets for training and testing (Called loaders)
    data_util = DataUtil(start_date,end_date)
    train_loader, test_loader, x_train, y_train, x_test, y_test = data_util.generate_data_loaders(batch_size=batch_size,
                                                                split_ratio=split_ratio,
                                                                look_back=7,
                                                                device=device)
    

    # TO DO FENG
    look_back = 7
    model = LSTM(1, look_back, 4).to(device)

    
    # Train the model
    model = train_model(train_loader,
                        test_loader,
                        model=model,
                        criterion=criterion,
                        optimizer_type=optimizer_type,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        load_model=None,
                        device=device)
    
    # Test model
    test_model(test_loader, model, criterion, device) 
    
    # Plot Price Predictions:
    model = model.to('cpu')
    pred_train = model(x_train).detach().numpy()
    pred_test = model(x_test).detach().numpy()
    plot_price_predictions(pred_train,y_train,pred_test,y_test)
    
if __name__ == "__main__":
    main()