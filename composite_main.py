# Import necessary libraries
import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import KFold

sys.path.append('/data/users3/ygao11/data/Moder_Severe_AD/A-loss_function')

# Import models and utility functions
from model.LSTM_recursive import recursive_forecast, LSTMForecaster
from model.TA_LSTM import TALSTM
from model.dataloader import TimeSeriesDataset, load_data, sliding_window_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup argument parser
parser = argparse.ArgumentParser(description='Train a neural network on ICN time series data with joint loss')
parser.add_argument('--augmentation_model', choices=['LSTM_stateless', 'LSTM_recursive'], default='LSTM_recursive', help='the selection of data argumentation model')
parser.add_argument('--step', type=int, default=4, help='step for augmentation of LSTM recursive model')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer')
parser.add_argument('--regression_model', choices=['RegressionCNN', 'TimeSeriesCNN', 'TimeSeriesTransformer', 'TimeSeriesCNN_att', 'TA_LSTM'], default='TA_LSTM', help='Which model to use')
parser.add_argument('--eval_mode', choices=['ten_fold_cv', 'test', 'valid'], default='ten_fold_cv', help='Evaluation mode: 10-fold cross-validation or a simple train-test split (test) or simple re-evaluate the validation in ten-fold cv')
parser.add_argument('--num_of_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='default batch size during training')
parser.add_argument('--lambda_c', type=float, default=0.5, help='weight between forecasting loss and regression loss')
args = parser.parse_args()

for arg in vars(args):
    value = getattr(args, arg)
    print(f"{arg}: {value} (Type: {type(value).__name__})")


# Setup experiment logging directory
base_path = "/data/users3/ygao11/data/Moder_Severe_AD/A-loss_function/experiment_joint_log/E{:03d}_log"
base_number = 0  # Initial log number
while os.path.exists(base_path.format(base_number)):
    base_number += 1
save_dir = base_path.format(base_number)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('model saved in:', save_dir)
    

# Load data
X, y = load_data.load_data()  # Assuming load_data function exists within the load_data module
print('X, y shapes:', X.shape, y.shape)
print('Age mean and std:', np.mean(y), np.std(y))


# K-Fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store losses
train_fore_losses, train_regress_losses, train_joint_losses = [], [], []
test_fore_losses, test_regress_losses = [], []
associate_fore_losses, best_regress_losses = [], []
flag_print_statement = True


# Main training loop with K-Fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    best_regress_loss, associate_fore_loss = float('inf'), float('inf')
    save_path_fore = os.path.join(save_dir, f"forecasting_{fold}.pth")
    save_path_regres = os.path.join(save_dir, f"regression_{fold}.pth")
    print(f"======== Start Fold {fold} ========")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = data.DataLoader(test_dataset)

    # Initialize models and loss functions
    if args.augmentation_model == 'LSTM_recursive':
        fore_model = LSTMForecaster(input_dim=53, hidden_dim=53, output_dim=53).to(device)  # Assuming these dimensions are correct


    if args.regression_model == 'TA_LSTM':
        seq_len = 122 + args.step  # Ensure this matches the expected input sequence length for TALSTM
        regres_model = TALSTM(seqlen=seq_len).to(device)

    # Combine parameters from both models for the optimizer
    optimizer = optim.Adam(list(fore_model.parameters()) + list(regres_model.parameters()), lr=args.lr)  # Added learning rate to optimizer

    # Loss functions
    fore_criterion_mse = nn.MSELoss()
    regress_criterion_mae = nn.L1Loss()

    # Training loop for each epoch
    for epoch in range(args.num_of_epoch):
        # Variables to accumulate losses
        total_joint_loss, total_fore_loss, total_regress_loss = 0.0, 0.0, 0.0
        prev_fore_loss, prev_regress_loss = float('inf'), float('inf')  # Initialize with high values
        batch_count = 0

        for inputs, labels in train_loader:
            labels = labels.unsqueeze(1).to(device)

            # Generate sliding window overlaps
            sli_inputs, sli_labels = sliding_window_overlap.sliding_windows_overlap(inputs)
            sli_inputs, sli_labels = torch.tensor(sli_inputs, dtype=torch.float32).to(device), torch.tensor(sli_labels, dtype=torch.float32).to(device)

            # Forecasting with recursive model
            fore_output = recursive_forecast(fore_model, sli_inputs, args.step)
            fore_loss = fore_criterion_mse(fore_output, sli_labels)
            
            if flag_print_statement: print(fore_model, sli_inputs.shape)

            # Regression task
            regres_inputs = inputs[:, -20:, :].to(device)  # Assuming last 20 timesteps are used for regression
            last_fore = recursive_forecast(fore_model, regres_inputs, args.step)
            concatenated_inputs = torch.cat((inputs.to(device), last_fore), dim=1)
            regress_output = regres_model(concatenated_inputs)
            regress_loss = regress_criterion_mae(regress_output, labels)

            # Dynamic weighting based on the reduction of losses
            reduction_fore = max(prev_fore_loss - fore_loss.item(), 0)  # Ensure non-negative
            reduction_regress = max(prev_regress_loss - regress_loss.item(), 0)
            total_reduction = reduction_fore + reduction_regress + 1e-9  # Avoid division by zero
            # lambda_c = reduction_regress / total_reduction  # Update lambda based on reductions
            lambda_c = args.lambda_c

            # Calculate joint loss with dynamic lambda
            joint_loss = lambda_c * fore_loss + (1 - lambda_c) * regress_loss

            # Backpropagation
            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()

            # Update previous losses
            prev_fore_loss, prev_regress_loss = fore_loss.item(), regress_loss.item()

            # Accumulate losses
            total_fore_loss += fore_loss.item()
            total_regress_loss += regress_loss.item()
            total_joint_loss += joint_loss.item()
            batch_count += 1
            
            # Debugging prints for the first batch
            if flag_print_statement:
                print(f'The generative data is prepared in a sliding window overlap approach')
                print(f'In the current batch, forecasting input {sli_inputs.shape}, forecasting output {fore_output.shape}')
                print(f'Regression input {concatenated_inputs.shape}, regression output {regress_output.shape}')
                flag_print_statement = False

        # Average losses for the epoch
        avg_joint_loss = total_joint_loss / batch_count
        avg_fore_loss = total_fore_loss / batch_count
        avg_regress_loss = total_regress_loss / batch_count

        # Test phase
        with torch.no_grad():
            total_test_fore_loss, total_test_regress_loss = 0.0, 0.0
            test_batch_count = 0
            for inputs, labels in test_loader:
                labels = labels.unsqueeze(1).to(device)

                # Sliding window for test data
                sli_inputs, sli_labels = sliding_window_overlap.sliding_windows_overlap(inputs)
                sli_inputs, sli_labels = torch.tensor(sli_inputs, dtype=torch.float32).to(device), torch.tensor(sli_labels, dtype=torch.float32).to(device)

                # Forecasting and regression for test data
                test_fore_output = recursive_forecast(fore_model, sli_inputs, args.step)
                test_fore_loss = fore_criterion_mse(test_fore_output, sli_labels)

                regres_inputs = inputs[:, -20:, :].to(device)
                last_fore = recursive_forecast(fore_model, regres_inputs, args.step)
                concatenated_inputs = torch.cat((inputs.to(device), last_fore), dim=1)
                test_regress_output = regres_model(concatenated_inputs)
                test_regress_loss = regress_criterion_mae(test_regress_output, labels)

                # Accumulate test losses
                total_test_fore_loss += test_fore_loss.item()
                total_test_regress_loss += test_regress_loss.item()
                test_batch_count += 1

            # Calculate average test losses
            avg_test_fore_loss = total_test_fore_loss / test_batch_count
            avg_test_regress_loss = total_test_regress_loss / test_batch_count
            print('test batch count for fold', fold, 'is: ',test_batch_count)

            # Save best model
            if avg_test_regress_loss < best_regress_loss:
                best_regress_loss = avg_test_regress_loss
                associate_fore_loss = avg_test_fore_loss
                torch.save(regres_model.state_dict(), save_path_regres)
                torch.save(fore_model.state_dict(), save_path_fore)
        # Logging for the epoch
        print(f'Epoch: {epoch}, Joint Loss: {avg_joint_loss}, Train Forecast Loss: {avg_fore_loss}, Train Regression Loss: {avg_regress_loss}, lambda: {lambda_c}')
        print(f'Test Forecast Loss: {avg_test_fore_loss}, Test Regression Loss: {avg_test_regress_loss}')
    
    print('best testing regression loss for fold', fold, 'is', best_regress_loss)
    print('testing forecasting loss for fold', fold, 'is', associate_fore_loss)

    associate_fore_losses.append(associate_fore_loss)
    best_regress_losses.append(best_regress_loss)

print('avg forecasting loss for ten fold', np.mean(np.array(associate_fore_losses)))
print('avg regression loss for ten fold', np.mean(np.array(best_regress_losses)))


