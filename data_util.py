import os  
import pandas as pd
import torch
import datetime as dt  
import numpy as np


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
        self.date_range = pd.date_range(self.start_date, self.end_date)


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
    
    def grab_data_combined(self, ignore_SMH = True, add_none_trading_days = False, num_of_prev_days = 0) -> torch.Tensor:
        """
        Combines all stock data into a single DataFrame for the specified date range.

        Args:
            dates (pd.DatetimeIndex, optional): The date range to use. Defaults to self.date_range.

        Returns:
            list of pd.DataFrames: The combined stocks data.
        """
        
            
        list_stocks_df = []
        # Initialize an empty DataFrame with the specified dates
        for idx, symbol_file in enumerate(self.all_stock_files):
            file_path = os.path.join(self.data_dir, symbol_file)
            print(f'Processing file: {file_path}')

            # Skip files related to 'SMH'
            if ignore_SMH and 'SMH' in symbol_file:
                print(f'Skipping {symbol_file}')
                continue

            # Read the CSV file into a temporary DataFrame
            df = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date))
            df_temp = pd.read_csv(
                file_path,
                index_col="Date",
                parse_dates=True,
                na_values=["nan"],
            )
            # Only add previous day data to current date if num of previous days is greater than 0
            if num_of_prev_days > 0:
                df_temp = self.create_shifted_dates(df_temp, 
                                                    num_of_prev_days= num_of_prev_days)

            # Join the temporary DataFrame with the main DataFrame
            if add_none_trading_days:
                df = df.join(df_temp)
                self.fill_missing_values(df)
            else:
                df = df_temp.loc[self.start_date:self.end_date]
            list_stocks_df.append(df)
        
        print()
        np_stocks_df = np.array(list_stocks_df) # [samples, dates, data]
        np_stocks_df = np_stocks_df.transpose(1, 0, 2) # [dates, samples, data]
        tensor_stocks_df = torch.tensor(np_stocks_df)
        return tensor_stocks_df
               
        
    
    def grab_a_stock_data(self, stock_name: str = None, add_none_trading_days = False, add_date_buffer = True) -> pd.DataFrame:
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
        
        return self.grab_a_stock_data(stock_name = 'SMH.csv')
    
    def grab_SMH_adj_close(self, add_date_buffer = True) -> torch.Tensor:
        df = self.grab_SMH_data().loc[:,"Adj Close"]
        np_array = np.array(df)
        return torch.tensor(np_array)
        



if __name__ == "__main__":
    # print("Executing as the main program")
    data_util = DataUtil()
    # combined_data = data_util.grab_data_combined()
    combined_data = data_util.grab_data_combined(num_of_prev_days=10)
    print(combined_data.shape)
    print(data_util.grab_data_combined().shape)


