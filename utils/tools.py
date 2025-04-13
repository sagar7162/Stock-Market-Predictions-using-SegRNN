import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf', dates=None, scaler=None):
    """
    Results visualization with dates on x-axis
    """
    plt.figure(figsize=(10, 6))
    
    # Apply inverse transformation if scaler is provided to get real values
    if scaler is not None:
        try:
            # Create properly shaped arrays for inverse transformation
            # The input data has shape (96,), but the scaler expects (n_samples, n_features)
            # We need to create a dummy array with zeros for all other features
            num_features = scaler.mean_.shape[0]
            target_feature_idx = -1  # Assuming the target is the last feature, adjust if needed
            
            # Handle true values
            true_2d = np.zeros((len(true), num_features))
            true_2d[:, target_feature_idx] = true
            true_transformed = scaler.inverse_transform(true_2d)[:, target_feature_idx]
            
            # Handle predictions if present
            if preds is not None:
                preds_2d = np.zeros((len(preds), num_features))
                preds_2d[:, target_feature_idx] = preds
                preds_transformed = scaler.inverse_transform(preds_2d)[:, target_feature_idx]
            else:
                preds_transformed = None
        except Exception as e:
            print(f"Error during inverse transformation: {e}")
            print("Falling back to original values")
            true_transformed = true
            preds_transformed = preds
    else:
        true_transformed = true
        preds_transformed = preds
    
    # If dates are provided, use them for x-axis
    if dates is not None and len(dates) == len(true):
        # Convert pandas Series to list if needed
        if hasattr(dates, 'values'):
            dates = dates.values
            
        # Convert dates to matplotlib format if they're not already
        if isinstance(dates[0], (str, pd.Timestamp, np.datetime64)):
            dates = pd.to_datetime(dates)
            
        # Plot with dates
        plt.plot(dates, true_transformed, label='GroundTruth', linewidth=2)
        if preds is not None:
            plt.plot(dates, preds_transformed, label='Prediction', linewidth=2)
            
        # Format the x-axis to show dates properly
        plt.gcf().autofmt_xdate()  # Auto-format date labels
        ax = plt.gca()
        
        # Use appropriate date formatter based on date range
        days = (dates[-1] - dates[0]).total_seconds() / (24 * 3600)
        if days > 365:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xlabel('Date (Year-Month)')
        elif days > 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.xlabel('Date (Month-Day)')
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xlabel('Date (Month-Day Hour)')
    else:
        # Fallback to numeric indices if dates are not available
        plt.plot(true_transformed, label='GroundTruth', linewidth=2)
        if preds is not None:
            plt.plot(preds_transformed, label='Prediction', linewidth=2)
    
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    # model_params = 0
    # for parameter in model.parameters():
    #     model_params += parameter.numel()
    #     print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
