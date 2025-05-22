# OLS_Stock_Predictor
# Author: Austin Morris
# Date: 21 May, 2025

Simple OLS(Ordinary Least Squares) regression model to predict next closing price of a stock using yfinance, numpy, matlibplot, and scikit. 
# Features or independent variables
This model uses technical indicators: moving average, price momentum, volume change/moving average,  and volatility measure.
# Target or dependent variable
End of day price of the provided stock Stock.

# To Use

insert a string containing the ticker name of the stock of interest into the fetch_stock_data(ticker('Name of Stock Ticker' ), period('period to gather data from' e.g. max, 5y(5 years), etc.)) in the main function to predict end of day price for that stock using OLS.

Or use this command line from terminal:

python your_script.py --ticker [Name of Stock Ticker] --period [Desired Period for Data Collection]

Default: SOFI, max
