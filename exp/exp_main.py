from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, VanillaRNN, SegRNN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
import pandas as pd

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'VanillaRNN': VanillaRNN,
            'SegRNN': SegRNN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("train steps:", train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Extract date information from the dataset for visualization
        dates = None
        df_raw = None
        try:
            # Directly read the dates from the original CSV file
            df_raw = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
            
            # Get the date column from the original data
            if 'date' in df_raw.columns:
                # Calculate test start index based on the dataset split
                num_train = int(len(df_raw) * 0.7)
                num_test = int(len(df_raw) * 0.2)
                test_start_idx = len(df_raw) - num_test
                
                # Get dates for test data - convert to list to avoid index issues
                date_series = df_raw['date'].iloc[test_start_idx:].reset_index(drop=True)
                dates = pd.to_datetime(date_series, format="%d-%m-%Y", errors='coerce')
                print(f"Successfully extracted {len(dates)} dates for visualization")
        except Exception as e:
            print(f"Could not extract dates from original data: {e}")
            dates = None

        # Create a list to store prediction results with dates
        prediction_results = []

        begin_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                # Save prediction results with dates for each batch
                if i % 20 == 0:
                    # Get the scaler for inverse transformation
                    scaler = test_data.scaler if hasattr(test_data, 'scaler') else None
                    
                    # Only use the prediction part (not concatenating with input)
                    gt = true[0, :, -1]  # Ground truth (prediction horizon only)
                    pd_values = pred[0, :, -1]  # Predictions
                    
                    # Extract the corresponding dates for this batch
                    batch_dates = None
                    if dates is not None:
                        try:
                            # Calculate the proper dates for the prediction period only
                            batch_start_idx = i * test_loader.batch_size + self.args.seq_len
                            pred_end_idx = batch_start_idx + self.args.pred_len
                            if batch_start_idx < len(dates):
                                # Use only prediction period dates
                                batch_dates = dates[batch_start_idx:pred_end_idx]
                                
                                # Get real values if scaler is available
                                if scaler is not None:
                                    try:
                                        # Create properly shaped arrays for inverse transformation
                                        num_features = scaler.mean_.shape[0]
                                        target_feature_idx = -1
                                        
                                        # Handle true values
                                        true_2d = np.zeros((len(gt), num_features))
                                        true_2d[:, target_feature_idx] = gt
                                        true_transformed = scaler.inverse_transform(true_2d)[:, target_feature_idx]
                                        
                                        # Handle predictions
                                        preds_2d = np.zeros((len(pd_values), num_features))
                                        preds_2d[:, target_feature_idx] = pd_values
                                        preds_transformed = scaler.inverse_transform(preds_2d)[:, target_feature_idx]
                                        
                                        # Store the transformed values
                                        batch_results = []
                                        for j in range(len(batch_dates)):
                                            if j < len(true_transformed) and j < len(preds_transformed):
                                                batch_results.append({
                                                    'Date': batch_dates.iloc[j] if hasattr(batch_dates, 'iloc') else batch_dates[j],
                                                    'Actual': true_transformed[j],
                                                    'Predicted': preds_transformed[j]
                                                })
                                        prediction_results.extend(batch_results)
                                    except Exception as e:
                                        print(f"Error transforming values: {e}")
                        except Exception as e:
                            print(f"Error extracting dates for batch {i}: {e}")
                    
                    visual(gt, pd_values, os.path.join(folder_path, str(i) + '.pdf'), 
                           dates=batch_dates, scaler=scaler)

        ms = (time.time() - begin_time) * 1000 / len(test_data)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f.write('\n')
        f.write('\n')
        f.close()

        # Fix the metrics array to ensure homogeneous shape
        metrics_array = []
        for m in [mae, mse, rmse, mape, mspe, rse, corr]:
            # Convert any non-scalar metrics to their mean or first element
            if hasattr(m, '__iter__'):
                metrics_array.append(float(np.mean(m)))
            else:
                metrics_array.append(float(m))

        # Save the prediction results to a CSV file
        if prediction_results:
            # Fix for all entries in the prediction_results list
            if hasattr(test_data, 'scaler'):
                # First pass: Get original values from raw data
                for result in prediction_results:
                    date_str = result['Date'].strftime("%d-%m-%Y") if hasattr(result['Date'], 'strftime') else result['Date']
                    if df_raw is not None and 'date' in df_raw.columns:
                        original_row = df_raw[df_raw['date'] == date_str]
                        if not original_row.empty and 'Close' in original_row.columns:
                            # If we find a match in the original data, use that value for Actual
                            result['Actual'] = original_row['Close'].values[0]
                
                # Second pass: Calculate global scaling factor based on all available data points
                valid_pairs = [(result['Actual'], result['Predicted']) 
                              for result in prediction_results 
                              if not isinstance(result['Actual'], str)]  # Filter out any string values
                
                if valid_pairs:
                    actuals = np.array([a for a, _ in valid_pairs])
                    predictions = np.array([p for _, p in valid_pairs])
                    
                    # Calculate the median scaling ratio to be robust to outliers
                    scaling_ratios = actuals / predictions
                    # Use median ratio to avoid influence of extreme values
                    global_scaling_factor = np.median(scaling_ratios)
                    
                    print(f"Global scaling factor: {global_scaling_factor:.4f}")
                    
                    # Apply the global scaling factor to all predictions
                    for result in prediction_results:
                        result['Predicted'] = result['Predicted'] * global_scaling_factor

            df_results = pd.DataFrame(prediction_results)
            csv_path = os.path.join(folder_path, 'stock_predictions.csv')
            df_results.to_csv(csv_path, index=False)
            print(f"Saved stock price predictions to {csv_path}")

        np.save(folder_path + 'metrics.npy', np.array(metrics_array))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save as numpy array
        np.save(folder_path + 'real_prediction.npy', preds)
        
        # Also save predictions in CSV format with dates
        try:
            # Get the future dates from pred_data
            if hasattr(pred_data, 'future_dates'):
                # Extract the future dates (last pred_len entries)
                future_dates = pred_data.future_dates[-self.args.pred_len:]
                
                # Get the predictions for the target feature
                future_predictions = preds[0, :, -1]  # Using the last feature dimension (assumed to be the target)
                
                # Apply inverse transform to get real stock values
                if hasattr(pred_data, 'scaler'):
                    scaler = pred_data.scaler
                    num_features = scaler.mean_.shape[0]
                    target_feature_idx = -1
                    
                    # Create a properly shaped array for inverse transform
                    preds_2d = np.zeros((len(future_predictions), num_features))
                    preds_2d[:, target_feature_idx] = future_predictions
                    
                    # Apply inverse transform
                    real_preds = scaler.inverse_transform(preds_2d)[:, target_feature_idx]
                else:
                    real_preds = future_predictions
                    
                # Create DataFrame and save to CSV
                df_future = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted': real_preds
                })
                
                csv_path = os.path.join(folder_path, 'future_predictions.csv')
                df_future.to_csv(csv_path, index=False)
                print(f"Saved future predictions to {csv_path}")
                
                # Create visualization of future predictions
                plt.figure(figsize=(10, 6))
                plt.plot(df_future['Date'], df_future['Predicted'], label='Future Prediction', linewidth=2)
                plt.title('Stock Price Forecast')
                plt.xlabel('Date')
                plt.ylabel('Stock Price')
                plt.legend()
                plt.gcf().autofmt_xdate()
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, 'future_prediction_plot.pdf'), bbox_inches='tight')
                plt.close()
                print(f"Saved future prediction plot to {folder_path + 'future_prediction_plot.pdf'}")
            else:
                print("Warning: Could not find future_dates in pred_data")
        except Exception as e:
            print(f"Error creating future predictions CSV: {e}")
            
        return
