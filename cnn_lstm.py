import torch
import torch.nn as nn
import torch.optim as optim


# INCOMPLETE!

class CNN_LSTM(nn.Module):
    def __init__(self, 
                 cnn_in_channels, 
                 cnn_out_channels, 
                 lstm_hidden_size, 
                 lstm_num_layers, 
                 output_size):
        super(CNN_LSTM, self).__init__()
        
        # CNN Layer 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_in_channels, 
                      out_channels=cnn_out_channels, 
                      kernel_size=2, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # CNN Layer 1
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_in_channels, 
                      out_channels=cnn_out_channels, 
                      kernel_size=2, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM Layer 2
        self.lstm = nn.LSTM(input_size=cnn_out_channels, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Convert Data to Linear layer
        self.fc = nn.Linear(lstm_hidden_size, 
                            output_size)
        
        # Flatten Function
        self.flatten = nn.Flatten()


    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        
        # Permute to fit LSTM input requirements (batch, seq_len, feature)
        out = out.permute(0, 2, 1)
        
        # Pass through LSTM layers
        out, hidden_x = self.lstm(out)
        
        # Only take the output of the last time step
        out = out[:, -1, :]
        
        # Pass through Fully Connected layer
        out = self.fc(out)
        
        return out