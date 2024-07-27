import torch
import torch.nn as nn
import torch.optim as optim

# INCOMPLETE!

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_out_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        
        # CNN Layer 1
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM Layer 2
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, batch_first=True)
        
        # Convert Data to Linear layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)
        
        # Flatten Function
        self.flatten = nn.Flatten()


    def forward(self, x):
        
        output = x
        return output