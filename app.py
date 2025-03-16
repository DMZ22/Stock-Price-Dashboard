import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import warnings
import os

# Suppress warnings and set plotting styles
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# App configuration
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")
st.title('Stock Price Prediction App')
st.markdown("""
This app predicts stock prices 2 hours ahead and provides buy/sell signals for both long and short trades.
* **Predictions and signals are valid for 1 hour only**.
* Data is fetched from Finnhub API with yfinance as fallback.
* LSTM model is used for prediction.
""")

# Sidebar setup
st.sidebar.header('User Input')

# API Key Management - Fixed implementation
@st.cache_data(ttl=3600)
def get_finnhub_api_key():
    """
    Get Finnhub API key from environment variables, Streamlit secrets, or default value.
    """
    # Use the provided API key
    return "cva4re1r01qshflg3900cva4re1r01qshflg390g"

@st.cache_data(ttl=3600)
def get_finnhub_secret():
    """
    Get Finnhub secret from environment variables, Streamlit secrets, or default value.
    """
    # Use the provided secret
    return "cva41ehr01qshflfupg0"

finnhub_api_key = get_finnhub_api_key()
finnhub_secret = get_finnhub_secret()

# Stock symbol search with proper error handling
@st.cache_data(ttl=3600)
def search_company(query):
    """Search for company symbols using Finnhub API"""
    try:
        url = f"https://finnhub.io/api/v1/search?q={query}&token={finnhub_api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('result', [])
        else:
            st.sidebar.warning(f"API Error: {response.status_code}. Using direct ticker input.")
            return []
    except Exception as e:
        st.sidebar.warning(f"Search error: {str(e)}. Using direct ticker input.")
        return []

# User inputs
stock_query = st.sidebar.text_input("Enter stock name or ticker:", "AAPL")
stock_symbol = "AAPL"  # Default

if stock_query:
    search_results = search_company(stock_query)
    if search_results:
        stock_options = [f"{result['symbol']}: {result['description']}" for result in search_results]
        selected_stock = st.sidebar.selectbox("Select a stock:", stock_options)
        stock_symbol = selected_stock.split(':')[0].strip() if selected_stock else stock_query
    else:
        stock_symbol = stock_query
        st.sidebar.info(f"Using '{stock_symbol}' as the ticker symbol.")

# Timeframe selection
timeframe_options = {'1 hour': '60', '15 minutes': '15', '5 minutes': '5'}  # Updated to match Finnhub API requirements
timeframe_selection = st.sidebar.radio("Select data resolution:", list(timeframe_options.keys()))
timeframe = timeframe_options[timeframe_selection]

# Set time period for historical data (fixed at 30 days)
days_to_fetch = 30
end_date = datetime.now()
start_date = end_date - timedelta(days=days_to_fetch)

# Improved data fetching functions with better fallback mechanism

@st.cache_data(ttl=1800)
def fetch_finnhub_data(symbol, resolution, from_time, to_time):
    """Fetch stock data from Finnhub API with improved error handling"""
    try:
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={resolution}&from={int(from_time.timestamp())}&to={int(to_time.timestamp())}&token={finnhub_api_key}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('s') == 'no_data':
                st.sidebar.warning(f"No data available from Finnhub for {symbol} with selected parameters.")
                return None
                
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime([datetime.datetime.fromtimestamp(x) for x in data['t']]))
            
            if df.empty:
                st.sidebar.warning("Finnhub returned empty dataset.")
                return None
                
            return df
        else:
            st.sidebar.warning(f"Finnhub API error: {response.status_code}")
            return None
    except Exception as e:
        st.sidebar.warning(f"Finnhub error: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def fetch_yfinance_data(ticker, period='1mo', interval='5m'):
    """Fetch stock data from Yahoo Finance with improved error handling"""
    try:
        st.sidebar.info(f"Attempting to fetch data from Yahoo Finance...")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            st.sidebar.warning(f"Yahoo Finance returned empty dataset for {ticker}.")
            return None
            
        # Check if we have enough data points
        if len(data) < 10:  # Arbitrary minimum threshold
            st.sidebar.warning(f"Insufficient data points from Yahoo Finance: {len(data)}")
            return None
            
        return data
    except Exception as e:
        st.sidebar.warning(f"Yahoo Finance error: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def fetch_alpha_vantage_data(symbol, interval='5min', outputsize='full'):
    """Fetch stock data from Alpha Vantage as a third option"""
    try:
        # You'll need to get your own API key from https://www.alphavantage.co/
        alpha_vantage_api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        
        # For demo purposes, we'll use a function that could work with Alpha Vantage
        if interval == '5':
            av_interval = '5min'
        elif interval == '15':
            av_interval = '15min'
        elif interval == '60':
            av_interval = '60min'
        else:
            av_interval = '5min'
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={av_interval}&outputsize={outputsize}&apikey={alpha_vantage_api_key}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            time_series_key = f"Time Series ({av_interval})"
            
            if time_series_key not in data:
                st.sidebar.warning(f"Alpha Vantage returned no valid data structure for {symbol}")
                return None
                
            time_series = data[time_series_key]
            
            # Process the data into a DataFrame
            records = []
            for timestamp, values in time_series.items():
                records.append({
                    'timestamp': timestamp,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['5. volume'])
                })
            
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
        else:
            st.sidebar.warning(f"Alpha Vantage API error: {response.status_code}")
            return None
    except Exception as e:
        st.sidebar.warning(f"Alpha Vantage error: {str(e)}")
        return None

def fetch_data(symbol, resolution, start_date, end_date):
    """Enhanced fetch data function with multiple fallbacks and retry mechanism"""
    data = None
    source = None
    retry_count = 0
    max_retries = 3
    
    # Create an empty placeholder for status messages
    status_message = st.empty()
    
    while data is None and retry_count < max_retries:
        retry_count += 1
        
        # Try Finnhub first
        if data is None:
            status_message.info(f"Attempt {retry_count}: Fetching data from Finnhub...")
            data = fetch_finnhub_data(symbol, resolution, start_date, end_date)
            if data is not None and not data.empty:
                source = "Finnhub"
                status_message.success(f"Successfully retrieved {len(data)} data points from Finnhub")
        
        # Fall back to yfinance if Finnhub fails
        if data is None:
            status_message.info(f"Attempt {retry_count}: Fetching data from Yahoo Finance...")
            # Map Finnhub resolution to yfinance interval format
            if resolution == "5":
                interval = "5m"
            elif resolution == "15":
                interval = "15m"
            elif resolution == "60":
                interval = "1h"
            else:
                interval = "5m"  # Default
            
            period = f"{days_to_fetch + 5}d"  # Add buffer days
            data = fetch_yfinance_data(symbol, period=period, interval=interval)
            if data is not None and not data.empty:
                source = "Yahoo Finance"
                status_message.success(f"Successfully retrieved {len(data)} data points from Yahoo Finance")
        
        # Fall back to Alpha Vantage as third option
        if data is None and retry_count == max_retries - 1:
            status_message.info(f"Final attempt: Fetching data from Alpha Vantage...")
            data = fetch_alpha_vantage_data(symbol, interval=resolution)
            if data is not None and not data.empty:
                source = "Alpha Vantage"
                status_message.success(f"Successfully retrieved {len(data)} data points from Alpha Vantage")
                
        # If all APIs failed, try with a slight modification to the ticker symbol
        if data is None and retry_count == max_retries:
            # Sometimes adding exchange prefix helps
            modified_symbols = [f"{symbol}.US", symbol.replace(".", "-")]
            
            for mod_symbol in modified_symbols:
                status_message.info(f"Trying modified symbol: {mod_symbol}...")
                data = fetch_finnhub_data(mod_symbol, resolution, start_date, end_date)
                if data is not None and not data.empty:
                    source = "Finnhub (modified symbol)"
                    status_message.success(f"Successfully retrieved {len(data)} data points using {mod_symbol}")
                    break
    
    # Clear the status message
    status_message.empty()
    
    # If still no data, show comprehensive error
    if data is None or data.empty:
        st.error(f"""
        Failed to retrieve data for {symbol} after {retry_count} attempts.
        
        Troubleshooting tips:
        1. Check if the symbol is correct
        2. Try a different timeframe
        3. Ensure you have internet connectivity
        4. The stock might not have data for the selected period
        """)
        
        # Offer to use demo data
        if st.button("Use demo data instead?"):
            data = generate_demo_data()
            source = "Demo data"
            st.info("Using generated demo data for demonstration purposes")
    else:
        st.success(f"Data successfully retrieved from {source}: {len(data)} data points")
        
    return data

def generate_demo_data():
    """Generate synthetic stock data for demonstration when all APIs fail"""
    # Create date range for the past 30 days with 5-minute intervals
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=30)
    date_range = pd.date_range(start=start, end=end, freq='5T')
    
    # Create random walk for price data
    np.random.seed(42)  # For reproducibility
    price = 100  # Starting price
    n = len(date_range)
    changes = np.random.normal(0, 1, n) * 0.5  # Random price changes
    prices = np.exp(np.cumsum(changes))  # Create a random walk
    prices = prices / prices[0] * price  # Scale to start at the initial price
    
    # Create dataframe
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.998, 1.002, n),
        'High': prices * np.random.uniform(1.001, 1.015, n),
        'Low': prices * np.random.uniform(0.985, 0.999, n),
        'Close': prices,
        'Volume': np.random.randint(1000, 100000, n)
    }, index=date_range)
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df
# LSTM model functions
def prepare_data(df, look_back=12, forecast_ahead=24, test_size=0.2):
    """Prepare data for LSTM model with better error checking"""
    try:
        # Focus on closing prices for prediction
        if len(df) < look_back + forecast_ahead + 10:  # Ensure minimum data points
            st.error(f"Insufficient data points ({len(df)}). Need at least {look_back + forecast_ahead + 10}.")
            return None, None, None, None, None
        
        data = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Split data into training and testing
        train_size = int(len(scaled_data) * (1 - test_size))
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - look_back:]
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, look_back, forecast_ahead)
        X_test, y_test = create_sequences(test_data, look_back, forecast_ahead)
        
        # Verify we have enough sequences
        if len(X_train) < 10 or len(X_test) < 5:
            st.error("Not enough training sequences generated. Try a different stock or timeframe.")
            return None, None, None, None, None
            
        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        st.error(f"Data preparation error: {str(e)}")
        return None, None, None, None, None

def create_sequences(data, look_back, forecast_ahead):
    """Create sequences for LSTM model with validation"""
    X, y = [], []
    if len(data) <= look_back + forecast_ahead:
        return np.array(X), np.array(y)
        
    for i in range(len(data) - look_back - forecast_ahead):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back + forecast_ahead - 1, 0])
    return np.array(X), np.array(y)

def build_lstm_model(look_back, units=50):
    """Build LSTM model architecture with try/except"""
    try:
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=units))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        st.error(f"Model building error: {str(e)}")
        return None

def train_model(model, X_train, y_train, epochs=30, batch_size=22, validation_split=0.1):
    """Train the LSTM model with early stopping and progress indication"""
    try:
        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Create a progress bar for training
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback to update progress
        class ProgressCallback(EarlyStopping):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Training epoch {epoch+1}/{epochs}")
                super().on_epoch_end(epoch, logs)
        
        # Display a progress bar during training
        with st.spinner("Training LSTM model..."):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, ProgressCallback(monitor='val_loss', patience=10)],
                verbose=0
            )
        
        # Clear the progress display
        status_text.empty()
        progress_bar.empty()
        
        # Show training results
        st.success(f"Model trained successfully over {len(history.history['loss'])} epochs")
        
        return model, history
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None
def make_predictions(model, X_test, scaler):
    """Make price predictions with the trained model"""
    try:
        # Reshape input
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict
        predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        predictions = scaler.inverse_transform(predictions)
        
        # Make sure predictions is a flat array of floats
        predictions = np.ravel(predictions).astype(np.float64)
        
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def generate_signals(df, predictions, threshold_pct=0.5):
    """Generate trading signals based on predictions"""
    try:
        # Ensure we're working with scalar values
        current_price = float(df['Close'].iloc[-1])
        
        # Get the last prediction value as a scalar
        if isinstance(predictions, np.ndarray) and predictions.size > 0:
            predicted_price = float(predictions[-1])
        else:
            st.error("Invalid predictions array")
            return None
        
        # Calculate percentage change
        future_change_pct = ((predicted_price / current_price) - 1) * 100
        
        # Rest of the function remains the same
        # ...
        signal_strength = min(abs(float(future_change_pct)) / (threshold_pct * 3), 1.0)
        
        # Get recent data for support/resistance
        recent_df = df.iloc[-30:] if len(df) >= 30 else df
        resistance = float(recent_df['High'].max()) if not recent_df.empty else current_price
        support = float(recent_df['Low'].min()) if not recent_df.empty else current_price
        
        # Calculate recent volatility
        if len(recent_df) < 2:
            recent_volatility = 0
        else:
            recent_volatility = float(recent_df['Close'].pct_change().std() * 100)
        
        # Calculate adjusted threshold
        adjusted_threshold = max(threshold_pct, recent_volatility * 0.5)
        
        # Calculate buy/sell ranges
        buy_range = (
            float(current_price * (1 - adjusted_threshold / 100)),
            float(current_price * (1 - adjusted_threshold / 200))
        )
        sell_range = (
            float(current_price * (1 + adjusted_threshold / 200)),
            float(current_price * (1 + adjusted_threshold / 100))
        )
        
        # Set default signals
        long_signal = "HOLD"
        short_signal = "HOLD"
        
        # Compare as floats, not Series
        future_change_pct = float(future_change_pct)
        adjusted_threshold = float(adjusted_threshold)
        
        if future_change_pct > adjusted_threshold:
            long_signal = "BUY"
            short_signal = "EXIT SHORT"
        elif future_change_pct < -adjusted_threshold:
            long_signal = "SELL"
            short_signal = "SHORT"
            
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_percent': future_change_pct,
            'signal_strength': signal_strength,
            'long_signal': long_signal,
            'short_signal': short_signal,
            'buy_range': buy_range,
            'sell_range': sell_range,
            'support': support,
            'resistance': resistance,
            'prediction_time': datetime.datetime.now(),
            'valid_until': datetime.datetime.now() + datetime.timedelta(hours=1)
        }

    except Exception as e:
        st.error(f"Signal generation error: {str(e)}")
        return None


# Main analysis function
def analyze_stock(symbol, resolution):
    """Main analysis function to orchestrate the prediction process"""
    try:
        # Fetch data
        data = fetch_data(symbol, resolution, start_date, end_date)
        
        if data is None or data.empty:
            st.error(f"Could not fetch data for {symbol}")
            return
         # Determine parameters based on resolution
        look_back = 12  # Default for all timeframes
        if resolution == '5':
            forecast_ahead = 24  # 2 hours ahead for 5 min data
        elif resolution == '15':
            forecast_ahead = 8   # 2 hours ahead for 15 min data
        else:  # 1 hour
            forecast_ahead = 2   # 2 hours ahead for 1 hour data
        
        # Prepare data
        X_train, y_train, X_test, y_test, scaler = prepare_data(
            data, look_back=look_back, forecast_ahead=forecast_ahead
        )
        
        if X_train is None:
            return
        
        # Build and train model
        model = build_lstm_model(look_back)
        model, history = train_model(model, X_train, y_train, epochs=50)
        
        if model is None:
            return
        
        # Get predictions
        with st.spinner("Generating predictions and signals..."):
            predictions = make_predictions(model, X_test, scaler)
            
            if predictions is None:
                return
                
            # Generate signals
            signals = generate_signals(data, predictions)
            
            if signals is None:
                return
        
        # Display results
        display_results(data, predictions, signals, symbol)
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return

def display_results(data, predictions, signals, symbol):
    """Display analysis results, charts and trading signals"""
    # Ensure predictions is a proper array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array([])
    
    # Display prediction summary
    st.header(f"Analysis Results for {symbol}")
    
    # Define columns before using them
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Prediction")
        st.metric(
            label="Current Price", 
            value=f"${signals['current_price']:.2f}",
            delta=f"{signals['change_percent']:.2f}%" if signals['change_percent'] else None
        )
        
        st.metric(
            label="Predicted Price (2hrs ahead)", 
            value=f"${signals['predicted_price']:.2f}"
        )
        
        # Show validity period
        st.caption(f"Prediction valid until: {signals['valid_until'].strftime('%H:%M:%S')}")
    
    with col2:
        st.subheader("Trading Signals")
        
        # Color-code the signals
        long_color = "green" if signals['long_signal'] == "BUY" else "red" if signals['long_signal'] == "SELL" else "gray"
        short_color = "green" if signals['short_signal'] == "SHORT" else "red" if signals['short_signal'] == "EXIT SHORT" else "gray"
        
        st.markdown(f"**Long Position:** <span style='color:{long_color};font-weight:bold'>{signals['long_signal']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Short Position:** <span style='color:{short_color};font-weight:bold'>{signals['short_signal']}</span>", unsafe_allow_html=True)
        
        # Signal strength indicator
        st.progress(signals['signal_strength'])
        st.caption(f"Signal Strength: {signals['signal_strength']*100:.1f}%")
    
    # Show price chart with predictions
    st.subheader("Price Chart & Prediction")
    
    # Prepare chart data
    chart_data = data.copy()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual prices
    ax.plot(chart_data.index[-100:], chart_data['Close'].values[-100:], 
            label='Actual Price', color='blue')
    
    # Add predicted price point
    last_date = chart_data.index[-1]
    prediction_date = last_date + pd.Timedelta(hours=2)
    ax.scatter(prediction_date, signals['predicted_price'], 
               color='red' if signals['change_percent'] < 0 else 'green',
               s=100, marker='*', label='Predicted Price')
    
    # Connect last actual price to prediction with dotted line
    ax.plot([last_date, prediction_date], 
            [chart_data['Close'].iloc[-1], signals['predicted_price']], 
            'k--', alpha=0.5)
    
    # Add buy/sell ranges
    if signals['long_signal'] == "BUY" or signals['short_signal'] == "EXIT SHORT":
        ax.axhspan(signals['buy_range'][0], signals['buy_range'][1], 
                  alpha=0.2, color='green', label='Buy Range')
    elif signals['long_signal'] == "SELL" or signals['short_signal'] == "SHORT":
        ax.axhspan(signals['sell_range'][0], signals['sell_range'][1], 
                  alpha=0.2, color='red', label='Sell Range')
    
    # Add support and resistance lines
    ax.axhline(y=signals['support'], linestyle='--', alpha=0.7, color='blue', label='Support')
    ax.axhline(y=signals['resistance'], linestyle='--', alpha=0.7, color='red', label='Resistance')
    
    # Format chart
    ax.set_title(f"{symbol} Price Prediction", fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the chart in Streamlit
    st.pyplot(fig)
    
  # Show additional technical metrics 
st.subheader("Technical Analysis")  

# Make sure chart_data is defined before this point
# For example: chart_data = yf.download(symbol, start=start_date, end=end_date)

# Make sure chart_data is defined before this point
# Define required variables
symbol = "AAPL"  # Default symbol or get from user input
start_date = datetime.now() - timedelta(days=365)  # Default to 1 year of data
end_date = datetime.now()  # Current date

# Then download the data
chart_data = yf.download(symbol, start=start_date, end=end_date)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Calculate some basic technical indicators 
chart_data['SMA_5'] = chart_data['Close'].rolling(window=5).mean() 
chart_data['SMA_20'] = chart_data['Close'].rolling(window=20).mean() 
chart_data['RSI'] = calculate_rsi(chart_data['Close'])  

chart_data = yf.download(symbol, start=start_date, end=end_date)
print("Downloaded data:", type(chart_data), "with shape:", chart_data.shape)
# Display indicators
tech_col1, tech_col2, tech_col3 = st.columns(3)  
def calculate_indicators(chart_data):
    # Calculate some basic technical indicators 
    chart_data['SMA_5'] = chart_data['Close'].rolling(window=5).mean() 
    chart_data['SMA_20'] = chart_data['Close'].rolling(window=20).mean() 
    chart_data['RSI'] = calculate_rsi(chart_data['Close'])
    return chart_data

# Download data
chart_data = yf.download(symbol, start=start_date, end=end_date)
# Calculate indicators
chart_data = calculate_indicators(chart_data)

# Check if we have valid data before displaying metrics
if not chart_data.empty and len(chart_data) > 20:  # Ensure we have enough data
    with tech_col1:     
        if not pd.isna(chart_data['SMA_5'].iloc[-1]):
            st.metric(label="5-Period SMA", value=f"${chart_data['SMA_5'].iloc[-1]:.2f}")
        else:
            st.metric(label="5-Period SMA", value="N/A")

    with tech_col2:     
        if not pd.isna(chart_data['SMA_20'].iloc[-1]):
            st.metric(label="20-Period SMA", value=f"${chart_data['SMA_20'].iloc[-1]:.2f}")
        else:
            st.metric(label="20-Period SMA", value="N/A")

    with tech_col3:
        if not pd.isna(chart_data['RSI'].iloc[-1]):
            rsi_value = chart_data['RSI'].iloc[-1]
            rsi_color = "red" if rsi_value > 70 else "green" if rsi_value < 30 else "black"
            
            st.metric(
                label="RSI (14)",          
                value=f"{rsi_value:.1f}",
                delta="Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
            )
        else:
            st.metric(label="RSI (14)", value="N/A")
else:
    st.warning("Insufficient data to calculate technical indicators")

# Show buy/sell suggestions     
st.subheader("Trading Recommendations")

# Define chart_data before using it
chart_data = {}  # Initialize empty dictionary or load data from appropriate source
# Then use it
chart_data['SMA_5'] = chart_data['close'].rolling(window=5).mean()

# Define signals dictionary if it doesn't exist
# First, check if 'signals' exists without trying to access it in isinstance()
# REMOVE THIS LINE: def signals

# Define signals dictionary
signals = {}  # Initialize signals as a dictionary directly

# No need for the try/except block anymore since we're directly initializing it
# Now we can check if it needs to be reset
if not isinstance(signals, dict):
    signals = {}

# Now fill in the required fields
if 'long_signal' not in signals:
    signals['long_signal'] = ""
if 'short_signal' not in signals:
    signals['short_signal'] = ""
if 'current_price' not in signals:
    signals['current_price'] = 0.0
# Default symbol if not defined
try:
    symbol
except NameError:
    symbol = "Unknown"
   
    
# Define required keys with default values
if 'long_signal' not in signals:
    signals['long_signal'] = ""
if 'short_signal' not in signals:
    signals['short_signal'] = ""
if 'current_price' not in signals:
    signals['current_price'] = 0.0
    
# Default symbol if not defined
if 'symbol' not in locals() or not symbol:
    symbol = "Unknown"
        
    if signals['long_signal'] == "BUY":
        st.success(f"💰 **BUY Opportunity**: Consider buying {symbol} at current price (${signals['current_price']:.2f}). Target price: ${signals['predicted_price']:.2f}")
    elif signals['long_signal'] == "SELL":
        st.error(f"💸 **SELL Signal**: Consider selling {symbol} at current price (${signals['current_price']:.2f}). Expected drop to: ${signals['predicted_price']:.2f}")
    elif 'short_signal' in signals and signals['short_signal'] == "SHORT":
        st.error(f"📉 **SHORT Opportunity**: Consider shorting {symbol} at current price (${signals['current_price']:.2f}). Target price: ${signals['predicted_price']:.2f}")
    else:
        st.info(f"⏳ **HOLD Position**: No strong signals detected for {symbol} at this time.")
else:
    st.error("Signal data is missing or invalid")
    # Disclaimer
    st.caption("**Disclaimer**: These predictions are for educational purposes only. Always do your own research before trading.")
# Helper function to calculate RSI
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) <= period:
        # Return a Series of NaN with the same index as prices
        return pd.Series(np.nan, index=prices.index)
        
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Run the analysis when the user submits
if st.sidebar.button('Analyze Stock'):
    if stock_symbol:
        with st.spinner(f"Analyzing {stock_symbol}..."):
            analyze_stock(stock_symbol, timeframe)
    else:
        st.sidebar.error("Please select a valid stock symbol")

# Add some additional information at the bottom
st.markdown("---")
st.markdown("""
**About this app:**
- The model uses historical price patterns to predict future prices
- Predictions are most accurate for short timeframes (2 hours)
- Trading signals are generated based on predicted price movements
- Always combine these signals with other analysis before making trading decisions
""")

# Add footer with credits
st.markdown("---")
st.caption("Developed with Streamlit, TensorFlow and Finnhub API")

