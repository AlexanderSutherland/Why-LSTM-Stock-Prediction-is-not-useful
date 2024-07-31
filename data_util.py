import os  
import pandas as pd
import torch
import datetime as dt  
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class DataUtil:
    """
    _summary_
    This class is use to grab and manipulate the data given in the /Data folder 
    """
    def __init__(self, start_date = dt.datetime(2014, 1, 1), end_date=dt.datetime(2016, 1, 1)) -> None:
        """
        Initializes the DataUtil class with relevant directories, default dates, and stock file information.
        """
        
        # Relevant Directories:
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.current_dir, 'Data')
        self.all_stock_files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

        # Default dates:
        self.start_date = start_date
        self.end_date = end_date

    def update_dates(self, start_date = None, end_date=None):
        """
        Updates the set start and end dates in the class.

        Args:
            start_date (dt.datetime): The new start date.
            end_date (dt.datetime): The new end date.
        """
        if isinstance(start_date, dt.datetime):
            print('DataUtil: Updating Start Date')
            self.start_date = start_date
        
        if isinstance(end_date, dt.datetime):
            print('DataUtil: Updating End Date')
            self.end_date = end_date

    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> None:
        """
        Fills missing values in the DataFrame using forward fill and backward fill.

        Args:
            df (pd.DataFrame): The DataFrame with missing values.

        Returns:
            None: The DataFrame is modified in place.
        """
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
    
    
    def rename_columns(self, df, num_of_prev_days):
        return df.rename(columns=lambda x: f"{x}_{str(num_of_prev_days)}")
    
    def create_shifted_dates(self, df, num_of_prev_days):
        start_index = df.index.get_indexer([self.start_date], method='nearest')[0]
        end_index = df.index.get_indexer([self.end_date], method='nearest')[0]
        
        if start_index < num_of_prev_days:
            raise ValueError ('Not enough days prior for start date given number of days to look past')
        
        df_new = df.copy()[start_index:end_index+1]

        for idx in range(1, num_of_prev_days + 1):
            df_temp = df.shift(idx).iloc[start_index:end_index+1]
            df_temp = self.rename_columns(df_temp, idx)
            df_new = pd.concat([df_new, df_temp], axis=1)
        return df_new
    
    def grab_data_combined(self, ignore_SMH=True, add_none_trading_days=False, num_of_prev_days=0) -> torch.Tensor:
        """
        Combines all stock data into a single DataFrame for the specified date range.

        Args:
            ignore_SMH (bool): Whether to ignore SMH files.
            add_none_trading_days (bool): Whether to add non-trading days.
            num_of_prev_days (int): Number of previous days to include in the data.

        Returns:
            tuple: A tuple containing the combined stock data as a torch.Tensor, numpy array, and list of DataFrames.
        """
        list_stocks_df = []

        for symbol_file in self.all_stock_files:
            file_path = os.path.join(self.data_dir, symbol_file)
            print(f'Processing file: {file_path}')

            if ignore_SMH and 'SMH' in symbol_file:
                print(f'Skipping {symbol_file}')
                continue

            df = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date))
            df_temp = pd.read_csv(file_path, index_col="Date", parse_dates=True, na_values=["nan"])

            if num_of_prev_days > 0:
                df_temp = self.create_shifted_dates(df_temp, num_of_prev_days=num_of_prev_days)

            if add_none_trading_days:
                df = df.join(df_temp)
                self.fill_missing_values(df)
            else:
                df = df_temp.loc[self.start_date:self.end_date]

            list_stocks_df.append(df)
        
        np_stocks_df = np.array(list_stocks_df)
        np_stocks_df = np_stocks_df.transpose(1, 0, 2)
        tensor_stocks_df = torch.tensor(np_stocks_df)
        return tensor_stocks_df, np_stocks_df, list_stocks_df
               
        
    
    def grab_single_stock_data(self, stock_name: str = None, add_none_trading_days = False, add_date_buffer = True) -> pd.DataFrame:
        """
        Grabs the specific Stock/ETF data given a date range.

        Args:
            dates (pd.DatetimeIndex, optional): The date range to use. Defaults to self.date_range.

        Returns:
            pd.DataFrame of the stock: stock data.
        """
        file_path_set = False
        if stock_name is None:
            raise ValueError ('No Stock Name given')
        
        # Find the stock name (can be given in name or file format)
        for idx, symbol_file in enumerate(self.all_stock_files):
            if stock_name in symbol_file and '.csv' in symbol_file:
                file_path = os.path.join(self.data_dir, symbol_file)
                file_path_set = True
                print(f'File found: {file_path}')
                break
        
        if not file_path_set:
            raise ValueError ('No Stock Name not found')
        
        # Read the CSV file into a temporary DataFrame
        df = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date))
        df_temp = pd.read_csv(
            file_path,
            index_col="Date",
            parse_dates=True,
            na_values=["nan"],
        )
        # Join the temporary DataFrame with the main DataFrame
        if add_none_trading_days:
            df = df.join(df_temp)
            self.fill_missing_values(df)
        else:
            df = df_temp.loc[self.start_date:self.end_date]
        
        return df
    
    def grab_SMH_data(self, add_date_buffer = True) -> pd.DataFrame:
        """
        Grabs the SMH data given a date range.

        Args:
            dates (pd.DatetimeIndex, optional): The date range to use. Defaults to self.date_range.

        Returns:
            pd.DataFrame of SMH: SMH data.
        """
        return self.grab_single_stock_data(stock_name = 'SMH.csv')
    
    
    def grab_SMH_adj_close(self) -> torch.Tensor:
        """
        Grabs the adjusted close prices of SMH.

        Returns:
            tuple: A tuple containing the adjusted close prices as a torch.Tensor, numpy array, and DataFrame.
        """
        df = self.grab_SMH_data().loc[:, "Adj Close"]
        np_array = np.array(df)
        return torch.tensor(np_array), np_array, df
    
    def grab_SMH_adj_close_minmax(self, min_scal = -1, max_scal = 1, look_back = 7):
        """
        Grabs and scales the daily returns of SMH using MinMaxScaler.

        Args:
            min_scal (int): The minimum scale value.
            max_scal (int): The maximum scale value.
            look_back (int): The number of previous days to include in the data.

        Returns:
            np.ndarray: The scaled data.
        """
        
        df_smh = pd.read_csv("./Data/SMH.csv") #, parse_dates =['date'])    
        df_smh["Date"] = pd.to_datetime(df_smh["Date"]) 
        df_smh = df_smh[(df_smh['Date'] >= self.start_date) & (df_smh['Date'] <= self.end_date)]
        
        df_target = df_smh[["Adj Close"]]
        
        df_shifted = self.add_previous_dates_adj_closing(df_target, look_back)   # add price data of seven (lookback) previous day to each entry of the df. Drop the first six entries as they don't have enough previous date data 
        np_shifted = df_shifted.to_numpy()
        
        #scaling df value to be between 01
        scaler = MinMaxScaler(feature_range=(min_scal, max_scal))
        np_shifted = scaler.fit_transform(np_shifted)
        return np_shifted
        
    def grab_SMH_daily_return_minmax(self, min_scal=-1, max_scal=1, look_back=7):
        """
        Grabs and scales the daily returns of SMH using MinMaxScaler.

        Args:
            min_scal (int): The minimum scale value.
            max_scal (int): The maximum scale value.
            look_back (int): The number of previous days to include in the data.

        Returns:
            np.ndarray: The scaled data.
        """
        df_smh = pd.read_csv("./Data/SMH.csv")
        df_smh["Date"] = pd.to_datetime(df_smh["Date"])
        df_smh['daily_return'] = df_smh['Adj Close'].pct_change()
        df_smh = df_smh[(df_smh['Date'] >= self.start_date) & (df_smh['Date'] <= self.end_date)]
        df_target = df_smh[["daily_return"]]
        df_shifted = self.add_previous_dates_pct_change(df_target, look_back)
        np_shifted = df_shifted.to_numpy()
        np_shifted = np_shifted*100
        return np_shifted
    
    def grab_SMH_daily_return_minmax_all_features(self, min_scal=-1, max_scal=1, look_back=7):
        """
        Grabs and scales the daily returns of SMH using MinMaxScaler.

        Args:
            min_scal (int): The minimum scale value.
            max_scal (int): The maximum scale value.
            look_back (int): The number of previous days to include in the data.

        Returns:
            np.ndarray: The scaled data.
        """
        df_smh = pd.read_csv("./Data/SMH.csv")
        df_smh["Date"] = pd.to_datetime(df_smh["Date"])
        df_smh['daily_return'] = df_smh['Adj Close'].pct_change()
        df_smh = df_smh[(df_smh['Date'] >= self.start_date) & (df_smh['Date'] <= self.end_date)]
        df_target = df_smh
        df_shifted = self.add_previous_dates_pct_change_all_features(df_target, look_back)
        df_shifted.drop(df_shifted.iloc[:, 0:6], inplace=True, axis=1)
        np_shifted = df_shifted.to_numpy()
        scaler = MinMaxScaler(feature_range=(min_scal, max_scal))
        np_shifted = scaler.fit_transform(np_shifted)
        return np_shifted
    
    
    def add_previous_dates_pct_change(self, df, lookback):
        """
        Adds previous dates' data to the DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            lookback (int): The number of previous days to include.

        Returns:
            pd.DataFrame: The DataFrame with added previous dates' data.
        """
        new_df = df.copy()
        if "Date" in new_df.columns:
            new_df.set_index("Date", inplace=True)

        for i in range(1, lookback + 1):
            new_df[f'daily_return(t-{i})'] = new_df['daily_return'].shift(i)

        new_df.dropna(inplace=True)
        return new_df
    
    def add_previous_dates_adj_closing(self, df, lookback):
        """
        Adds previous dates' data to the DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            lookback (int): The number of previous days to include.

        Returns:
            pd.DataFrame: The DataFrame with added previous dates' data.
        """
        new_df = df.copy()
        if "Date" in new_df.columns:
            new_df.set_index("Date", inplace=True)

        for i in range(1, lookback + 1):
            new_df[f'Adj Close(t-{i})'] = new_df['Adj Close'].shift(i)

        new_df.dropna(inplace=True)
        return new_df

    def add_previous_dates_pct_change_all_features(self, df, lookback):
        """
        Adds previous dates' data to the DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            lookback (int): The number of previous days to include.

        Returns:
            pd.DataFrame: The DataFrame with added previous dates' data.
        """
        new_df = df.copy()
        if "Date" in new_df.columns:
            new_df.set_index("Date", inplace=True)

        for i in range(1, lookback + 1):
            for column in df:
                if column != 'Date':
                    new_df[column+'(t-'+str(i)+')'] = new_df[column].shift(i)

        new_df.dropna(inplace=True)
        return new_df
    
    
    def generate_data_loaders_close_price(self,
                            batch_size=32, 
                            split_ratio=0.8,
                            look_back = 7,
                            device='cpu'):
        """
        Grabs the data loaders given a date range.

        Args:
            batch_size (int, optional): Size of each data batch. Defaults to 32.
            split_ratio (float, optional): Ratio to split the data into training and testing sets. Defaults to 0.8.

        Returns:
            tuple: A tuple containing training and test DataLoader instances.
        """
            
        data_shifted = self.grab_SMH_adj_close_minmax(look_back=look_back)
        x = data_shifted[:, 1:]
        y = data_shifted[:, 0]
        

        split = int(len(x) * split_ratio)

        # reshaping training data to be able to feed into LSTM
        x_train = x[:split].reshape(-1, look_back, 1)
        x_test = x[split:].reshape(-1, look_back, 1)
        y_train = y[:split].reshape(-1, 1)
        y_test = y[split:].reshape(-1, 1)

        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        x_test = torch.tensor(x_test).float()
        y_test = torch.tensor(y_test).float()

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False )
        
        return train_loader, test_loader, x_train, y_train, x_test, y_test
    
    def generate_data_loaders_daily_returns(self,
                                            batch_size=32, 
                                            split_ratio=0.8,
                                            look_back = 7,
                                            device='cpu'):
        """
        Grabs the data loaders given a date range.

        Args:
            batch_size (int, optional): Size of each data batch. Defaults to 32.
            split_ratio (float, optional): Ratio to split the data into training and testing sets. Defaults to 0.8.

        Returns:
            tuple: A tuple containing training and test DataLoader instances.
        """
            
        data_shifted = self.grab_SMH_daily_return_minmax(look_back=look_back)

        x = data_shifted[:, 1:]
        y = data_shifted[:, 0]

        split = int(len(x) * split_ratio)

        # reshaping training data to be able to feed into LSTM
        x_train = x[:split].reshape(-1, look_back, 1)
        x_test = x[split:].reshape(-1, look_back, 1)
        y_train = y[:split].reshape(-1, 1)
        y_test = y[split:].reshape(-1, 1)

        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        x_test = torch.tensor(x_test).float()
        y_test = torch.tensor(y_test).float()

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False )
        
        return train_loader, test_loader, x_train, y_train, x_test, y_test
    
    def generate_data_loaders_daily_returns_all_features(self,
                                                        batch_size=32, 
                                                        split_ratio=0.8,
                                                        look_back = 7,
                                                        device='cpu'):
        """
        Grabs the data loaders given a date range.

        Args:
            batch_size (int, optional): Size of each data batch. Defaults to 32.
            split_ratio (float, optional): Ratio to split the data into training and testing sets. Defaults to 0.8.

        Returns:
            tuple: A tuple containing training and test DataLoader instances.
        """
            
        data_shifted = self.grab_SMH_daily_return_minmax_all_features(look_back=look_back)

        x = data_shifted[:, 1:]
        y = data_shifted[:, 0]


        split = int(len(x) * split_ratio)

        # reshaping training data to be able to feed into LSTM
        x_train = x[:split]
        x_test = x[split:]
        y_train = y[:split].reshape(-1, 1)
        y_test = y[split:].reshape(-1, 1)

        x_train = torch.tensor(x_train).float()
        x_train = x_train.unsqueeze(1)
        y_train = torch.tensor(y_train).float()
        x_test = torch.tensor(x_test).float()
        x_test = x_test.unsqueeze(1)
        y_test = torch.tensor(y_test).float()

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False )
        
        return train_loader, test_loader, x_train, y_train, x_test, y_test
    
    def convert_daily_returns_to_price(self, daily_returns):
        pass

if __name__ == "__main__":
    # print("Executing as the main program")
    data_util = DataUtil()
    # combined_data = data_util.grab_data_combined()
    data_info = combined_data = data_util.generate_data_loaders_daily_returns()


