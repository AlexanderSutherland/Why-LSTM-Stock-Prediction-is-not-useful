import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first = True):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTM, self).__init__()

        # Save off inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Define the LSTM layer
        if batch_first:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            
        # Define the fully connected layers
        self.fc = nn.Linear(hidden_size, 10)  # Output of LSTM to hidden layer
        self.fc2 = nn.Linear(10, 50)  # Hidden layer to final output
        self.fc3 = nn.Linear(50, 25)  # Hidden layer to final output
        self.fc4 = nn.Linear(25, 1)  # Hidden layer to final output

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        
        batch_size = x.size(0)
        
        if self.batch_first:
            # Initialize hidden state and cell state with zeros
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            
        
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        
        # Pass the output of the last time step through fully connected layers
        if self.batch_first:
            out = self.fc(out[:, -1, :])  # Take output from the last time step
        else:
            out = self.fc(out)  # Take output from the last time step
        out = F.sigmoid(out)  # Apply sigmoid activation using functional API
        out = self.fc2(out)  # Final fully connected layer
        out = F.relu(out)  # Apply tanh activation using functional API
        out = self.fc3(out)  # Final fully connected layer
        out = F.tanh(out)  # Apply tanh activation using functional API
        out = self.fc4(out)  # Final fully connected layer
        
        return out
