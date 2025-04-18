import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format="%d-%m-%Y", dayfirst=True, errors='coerce')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format="%d-%m-%Y", dayfirst=True, errors='coerce')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Apply preprocessing: Rename 'Date' column to 'date' if needed
        if 'Date' in df_raw.columns and 'date' not in df_raw.columns:
            df_raw.rename(columns={'Date': 'date'}, inplace=True)
            
        # For stock data, check if this is a stock dataset
        stock_columns = ['Prev Close', 'Open', 'High', 'Low', 'Close']
        is_stock_data = all(col in df_raw.columns for col in stock_columns)
        
        # Print debugging info
        # print(f"Detected columns: {list(df_raw.columns)}")
        # print(f"Is stock data: {is_stock_data}")
        
        if is_stock_data:
            # Very specific handling for stock data - only keep these exact columns
            stock_input_columns = ['Prev Close', 'Open', 'High', 'Low']
            
            # Ensure target is properly set
            if self.target not in df_raw.columns or self.target == 'OT':
                self.target = 'Close'
                
            # Keep only date, the 4 input columns, and target
            keep_columns = ['date'] + stock_input_columns + [self.target]
            # print(f"Keeping only these columns: {keep_columns}")
            df_raw = df_raw[keep_columns]
        else:
            # Original handling for non-stock data
            cols = list(df_raw.columns)
            if self.target in cols:
                cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            
        # Print columns after filtering
        # print(f"Final columns: {list(df_raw.columns)}")
        
        # Continue with the rest of the code
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Print shape of data after processing
        # print(f"Data shape after processing: {data.shape}")

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format="%d-%m-%Y", dayfirst=True, errors='coerce')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
                                          
        # Apply preprocessing: Rename 'Date' column to 'date' if needed
        if 'Date' in df_raw.columns and 'date' not in df_raw.columns:
            df_raw.rename(columns={'Date': 'date'}, inplace=True)
            
        # For stock data, identify if we have the essential stock columns
        # More flexible check to detect stock data format
        essential_stock_columns = ['Prev Close', 'Open', 'High', 'Low', 'Close']
        is_stock_data = all(col in df_raw.columns for col in essential_stock_columns)
        
        # print(f"[Pred] Detected columns: {list(df_raw.columns)}")
        # print(f"[Pred] Is stock data: {is_stock_data}")
        
        if is_stock_data and not self.cols:
            # Specific handling for stock data
            input_columns = ['Prev Close', 'Open', 'High', 'Low']
            
            # Ensure target is properly set for stock data
            if self.target not in df_raw.columns or self.target == 'OT':
                self.target = 'Close'  # Default target for stock data
                
            # Set cols for stock data
            self.cols = input_columns.copy()
            if self.target not in self.cols:
                self.cols.append(self.target)
                
            # Keep only date, the 4 input columns, and target
            keep_columns = ['date'] + input_columns + [self.target]
            # print(f"[Pred] Keeping only these columns: {keep_columns}")
            df_raw = df_raw[keep_columns]
        
        # Original handling when cols are provided or for non-stock data
        if self.cols:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            if self.target in cols:
                cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')
                
        # Ensure proper column order
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(f"[Pred] Final columns: {list(df_raw.columns)}")
        
        # Calculate prediction boundaries
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Print shape of data after processing
        # print(f"[Pred] Data shape after processing: {data.shape}")
        
        # More robust date parsing
        tmp_stamp = df_raw[['date']][border1:border2]
        
        # Print sample dates to help diagnose parsing issues
        sample_dates = tmp_stamp['date'].iloc[:5].tolist()
        # print(f"[Pred] Sample date strings: {sample_dates}")
        
        # Try different date formats in order of likelihood
        date_formats = [
            "%d-%m-%Y",   # 31-12-2020
            "%Y-%m-%d",   # 2020-12-31
            "%m/%d/%Y",   # 12/31/2020
            "%d/%m/%Y",   # 31/12/2020
            "%Y/%m/%d"    # 2020/12/31
        ]
        
        # Try each format until one works
        for date_format in date_formats:
            try:
                # print(f"[Pred] Trying date format: {date_format}")
                tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'], format=date_format, errors='raise')
                # print(f"[Pred] Date format {date_format} worked!")
                break
            except ValueError:
                continue
        
        # If none of the specific formats worked, try with errors='coerce' as fallback
        if pd.isna(tmp_stamp['date']).all():
            # print("[Pred] All specific formats failed, using pandas default parser with coerce")
            tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'], errors='coerce')
        
        # Check if we have valid dates to work with
        valid_dates = tmp_stamp['date'].dropna()
        if len(valid_dates) == 0:
            # print("[Pred] WARNING: Could not parse any dates! Using current date as fallback")
            last_date = pd.Timestamp.now()
        else:
            # Use the last valid date
            last_date = valid_dates.iloc[-1]
            # print(f"[Pred] Using last valid date: {last_date}")
            
        # Generate prediction dates
        pred_dates = pd.date_range(start=last_date, periods=self.pred_len + 1, freq=self.freq)
        # print(f"[Pred] Generated {len(pred_dates)} future dates starting from {pred_dates[0]}")

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp['date'].values) + list(pred_dates[1:])
        
        # Store the original dates (including future predictions) for later use
        self.future_dates = df_stamp.date.copy()
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month if pd.notnull(row) else 1, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day if pd.notnull(row) else 1, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday() if pd.notnull(row) else 0, 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour if pd.notnull(row) else 0, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute if pd.notnull(row) else 0, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # Fill NaT values with the last valid date to avoid errors in time_features
            df_stamp['date'] = df_stamp['date'].fillna(method='ffill')
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.df_stamp = df_stamp  # Store the full DataFrame with dates for later reference

    def __getitem__(self, index):
        # For prediction, we only need the last sequence from the data
        # Ignore the index parameter since we always return the same last sequence
        s_begin = 0  # Always use the first (and only) sequence
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:self.seq_len]  # Last sequence_length items
        if self.inverse:
            seq_y = self.data_x[:self.label_len]
        else:
            seq_y = self.data_y[:self.label_len]
        seq_x_mark = self.data_stamp[:self.seq_len]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Original implementation is causing errors in the predict function
        # return len(self.data_x) - self.seq_len + 1
        
        # Simply return 1 for prediction mode since we're only predicting one sequence
        return 1
