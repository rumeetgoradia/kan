import os

from data import provide as data_provider
import torch

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # spy = yf.Tickers('AAPL MSFT')
    # print(spy.tickers['AAPL'].info)
    market_data = data_provider.get_market_data('data/processed/market.csv')

    stock_data = data_provider.get_stock_data('data/processed/sp500.csv')

    merged_data = data_provider.merge_market_and_stock_data(market_data, stock_data)

    X, y = data_provider.sequence_data(merged_data)
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print(X.shape, y.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
