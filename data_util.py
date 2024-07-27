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
    def __init__(self) -> None:
        """
        Initializes the DataUtil class with relevant directories, default dates, and stock file information.
        """
        
        # Relevant Directories:
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.current_dir, 'Data')
        self.all_stock_files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

        # Default dates:
        self.start_date = dt.datetime(2014, 1, 1)
        self.end_date = dt.datetime(2016, 1, 1)
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

    def grab_data_combined(self, dates: pd.DatetimeIndex = None) -> pd.DataFrame:
        """
        Combines all stock data into a single DataFrame for the specified date range.

        Args:
            dates (pd.DatetimeIndex, optional): The date range to use. Defaults to self.date_range.

        Returns:
            list of pd.DataFrames: The combined stocks data.
        """
        
        if type(dates) != pd.DatetimeIndex:
            print('[WARNING] No date range has been given, using default date range')
            dates = self.date_range
        list_stocks_df = []
        # Initialize an empty DataFrame with the specified dates
        for idx, symbol_file in enumerate(self.all_stock_files):
            file_path = os.path.join(self.data_dir, symbol_file)
            print(f'Processing file: {file_path}')

            # Skip files related to 'SMH'
            if 'SMH' in symbol_file:
                print(f'Skipping {symbol_file}')
                continue

            # Read the CSV file into a temporary DataFrame
            df = pd.DataFrame(index=dates)
            df_temp = pd.read_csv(
                file_path,
                index_col="Date",
                parse_dates=True,
                na_values=["nan"],
            )

            # Join the temporary DataFrame with the main DataFrame
            df = df.join(df_temp)
            self.fill_missing_values(df)
            list_stocks_df.append(df)
        
        np_stocks_df = np.array(list_stocks_df)
        tensor_stocks_df = torch.tensor(np_stocks_df)
        return tensor_stocks_df



if __name__ == "__main__":
    print("Executing as the main program")
    data_util = DataUtil()
    combined_data = data_util.grab_data_combined()
    print(combined_data.shape)

