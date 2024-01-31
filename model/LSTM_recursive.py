import torch
import torch.nn as nn
import numpy as np


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=53, num_layers=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])
        return output

# Example usage for recursive forecasting
def recursive_forecast(model, initial_sequence, steps):
    predictions = []
    input_sequence = initial_sequence.clone()
    
    for _ in range(steps):
        # Get the next step prediction
        next_step = model(input_sequence)
        predictions.append(next_step.unsqueeze(1))
        # Append prediction to sequence and remove oldest value
        new_input = torch.cat((input_sequence[:, 1:, :], next_step.unsqueeze(1)), axis=1)
        input_sequence = new_input
    return torch.cat(predictions,dim=1)