import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# The ultimate answer for reproducibility 
np.random.seed(42)

# gathering stock information using the yfinance labrary

def gather_stock_data(ticker, period) :
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period = period)
        if hist_data.empty:
            print(f"No data found for  {ticker}")
            return None
        return hist_data
        
    except Exception as e:
        print(f"Error gathering data: {e}")
        return None
    
# This model will use technical indicators as features
def create_features(hist_data) :

    """Add basic technical indicators"""

    # 10 Day Moving Average
    hist_data['MA10'] = hist_data['Close'].rolling(window = 10).mean()
    

    #Price Momentum
    hist_data['Price_Change'] = hist_data['Close'].pct_change()
   

    #Volume Indicator
    hist_data['Vol_Change'] = hist_data['Volume'].pct_change()
   

    #Volitility Measure
    hist_data['Volatility'] = hist_data['Close'].rolling(window = 10).std()

    #Target Variable: Next Closing price.
    hist_data['Closing_Price'] = hist_data['Close']

    # Drop NaN values
    hist_data.dropna(inplace = True)

    return hist_data

def build_model(hist_data, test_size = .1) :
    
    # Define Dependent and Independent Variables
    features  = ['MA10', 'Price_Change', 'Vol_Change', 'Volatility']
    
    X = hist_data[features]
    y = hist_data['Closing_Price']

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
                                                        shuffle = False)
    
    train = sm.add_constant(X_train)
    test = sm.add_constant(X_test)

    
    model = sm.OLS(y_train, train).fit()
    
    # Predictions
    y_pred_train = model.predict(train)
    y_pred_test = model.predict(test)

    #Evaluations
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    #print Evaluation Results: Precision of 4 Decimal Places.
    print("\nModel Evaluation:")
    print(f"Training RMSE: ${train_rmse:.4f}")
    print(f"Test RMSE: ${test_rmse:.4f}")
    print(f"Test MAE: ${test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return model, X_test, y_test, y_pred_test

# Visualize Results: Actual vs Predicted Price

def visualize_results(ticker, y_test, y_pred) :

    plt.figure(figsize = (12,6))

    #Plot Actual Values
    plt.plot(y_test.index, y_test.values, label = 'Actual Price', color = 'blue')

    #Plot Predicted Values
    plt.plot(y_test.index, y_pred, label = 'Predicted Price', color = 'red', linestyle = '--')

    plt.title(f'{ticker} Stock Price : Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha = .3)

    plt.tight_layout()
    plt.savefig('OLS_model_vs_Actual.png')
    plt.show()
    

def predict_price(model, latest_data) :

    
    prediction_data = sm.add_constant(latest_data)

    # Prediction
    price_prediction = model.predict(prediction_data).iloc[-1]

    return price_prediction

import argparse

def main(ticker, period) :

    df = gather_stock_data(ticker = ticker, period = period)

    if df is None :
        print('Failed to fetch data.')
        return
    
    

    # Define Features
    featured_data = create_features(df)
    
    #Print Preview of data with features added
    print("\nFeatures created. Preview:")
    print(featured_data[['Close', 'MA10', 'Closing_Price']].head())

    #X_test is not accessed but is a necessary part of building the model. 
    model, X_test, y_test, y_pred = build_model(featured_data)

    #Print Model Summary
    print(model.summary())


    #Create Plot
    visualize_results(ticker, y_test, y_pred)

    # Predict Price
    latest_data = featured_data[['Close', 'MA10', 'Price_Change', 'Vol_Change', 'Volatility']].iloc[-1:]
    closing_price = predict_price(model, latest_data)

    latest_close = df['Close'].iloc[-1]

    print(f"\nLatest closing price: ${latest_close:.2f}")
    print(f"Predicted next closing price: ${closing_price:.2f}")
    print(f"Predicted change: {(closing_price - latest_close) / latest_close * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock price prediction using OLS model")
    parser.add_argument("--ticker", type=str, default="SOFI", help="Stock ticker symbol (e.g., AAPL, TSLA)")
    parser.add_argument("--period", type=str, default="max", help="Time period for stock data (e.g., 1y, 6mo, 3mo)")
    args = parser.parse_args()

main(ticker=args.ticker, period=args.period)