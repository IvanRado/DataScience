import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    sp500 = pd.read_csv('sphist.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    mask = sp500['Date'] > datetime(year = 2015, month = 4, day = 1)
    sp500 = sp500.sort_values('Date', ascending = True)
    print(sp500.shape)
    print(sp500.head())
    
    # Want to make a few indicators for stock price prediction
    # We will make the average price from the past 5 days - Done
    # The average price from the past 30 days - Done
    # The average price from the past 365 days - Done
    # The ratio between the average price for the past 5 days and past 365 days
    # The standard deviation of the price over the past 5 days
    # The standard deviation of the price over the last 365 days
    # The ratio between these standard deviations
    
    # Loop through every entry in the data set and calculate the requisite indicators
    
    # Add two indicators that aren't just based on the price
    # Ratio of the average volume of the last five days to the last year
    # The standard deviation of the volume over the past year
    
    past_5 = []
    past_30 = []
    past_365 = []
    volume_5 = []
    volume_365 = []
    volume_365_std = []
    volume_ratio = []
    for i in range(len(sp500['Close'])):
        
        # Average of the past five days
        if i < 5:
            past_5.append(0)
            past_30.append(0)
            past_365.append(0)
            volume_5.append(0)
            volume_365.append(0)
            volume_365_std.append(0)
            volume_ratio.append(0)
        elif i >= 5 and i < 30: 
            past_5.append(sp500['Close'][i-5:i-1].mean())
            past_30.append(0)
            past_365.append(0)
            volume_5.append(sp500['Volume'][i-5:i-1].mean())
            volume_365.append(0)
            volume_365_std.append(0)
            volume_ratio.append(0)
        elif i >= 30 and i < 365:
            past_5.append(sp500['Close'][i-5:i-1].mean())
            past_30.append(sp500['Close'][i-30:i-1].mean())
            past_365.append(0)
            volume_5.append(sp500['Volume'][i-5:i-1].mean())
            volume_365.append(0)
            volume_365_std.append(0)
            volume_ratio.append(0)
        else: 
            past_5.append(sp500['Close'][i-5:i-1].mean())
            past_30.append(sp500['Close'][i-30:i-1].mean())
            past_365.append(sp500['Close'][i-365:i-1].mean())
            volume_5.append(sp500['Volume'][i-5:i-1].mean())
            volume_365.append(sp500['Volume'][i-365:i-1].mean())
            volume_365_std.append(sp500['Volume'][i-365:i-1].std())
            volume_ratio.append(volume_5[i]/volume_365[i])
            
    
    sp500['past_5'] = past_5
    sp500['past_30'] = past_30
    sp500['past_365'] = past_365
    sp500['volume_ratio'] = volume_ratio
    sp500['volume_365_std'] = volume_365_std
    
    # Drop any rows before Jan 2nd, 1951 since we don't have enough data to compute
    # a yearly metric for these entires
    cleaned = sp500[sp500['Date'] > datetime(year = 1951, month = 1, day = 2)]
    
    # Drop any rows that have missing values
    cleaned = cleaned.dropna(axis = 0)
    
    # Separate the train and test sets
    train = cleaned[cleaned['Date'] < datetime(year = 2013, month = 1, day = 1)]
    test = cleaned[cleaned['Date'] >= datetime(year = 2013, month = 1, day = 1)]
    
    # We will use mean squared error as our error metric as we are using a linear
    # regression model to fit our data
    lr = LinearRegression()
    features = ['past_5', 'past_30', 'past_365', 'volume_ratio', 'volume_365_std']
    target_train = train['Close']
    target_test = test['Close']
    lr.fit(train[features], target_train)
    predictions = lr.predict(test[features])
    error = mean_squared_error(target_test, predictions)
    print(error)
          