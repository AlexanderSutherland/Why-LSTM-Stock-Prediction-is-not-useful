
def lstm_dataprep(df, lookback):
    new_df = df.copy()
    if "Date" in new_df.columns:
        new_df.set_index("Date", inplace=True)

    for i in range(1, lookback + 1):
        new_df[f'Adj Close(t-{i})'] = new_df['Adj Close'].shift(i)

    new_df.dropna(inplace=True)
    return new_df