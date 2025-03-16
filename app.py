import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Function to get stock data
def get_stock_data(ticker, period='1y', interval='30m'):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock symbol
    - period: Time period to fetch data for (default: 1 year)
    - interval: Data interval (default: 30 minutes)
    
    Returns:
    - DataFrame with stock data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Function to prepare data for LSTM
def prepare_data(data, target_col='Close', lookback=12):
    """
    Prepare data for LSTM model
    
    Parameters:
    - data: DataFrame with stock data
    - target_col: Column to predict (default: Close price)
    - lookback: Number of time steps to look back (default: 12)
    
    Returns:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - scaler: Fitted scaler for inverse transformation
    """
    # Select only the target column
    dataset = data[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-lookback:]
    
    # Function to create sequences
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler, train_size

# Function to build LSTM model
def build_lstm_model(input_shape, dropout_rate=0.2):
    """
    Build LSTM model
    
    Parameters:
    - input_shape: Shape of input data
    - dropout_rate: Dropout rate to prevent overfitting
    
    Returns:
    - Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train model
def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
    """
    Train LSTM model
    
    Parameters:
    - model: LSTM model
    - X_train, y_train: Training data
    - epochs: Number of epochs
    - batch_size: Batch size
    - validation_split: Validation split
    
    Returns:
    - Trained model and history
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history

# Function to make predictions
def make_predictions(model, X_test, scaler, lookback, original_data, train_size):
    """
    Make predictions using trained model
    
    Parameters:
    - model: Trained LSTM model
    - X_test: Test data
    - scaler: Fitted scaler
    - lookback: Number of time steps looked back
    - original_data: Original dataset
    - train_size: Size of training data
    
    Returns:
    - DataFrame with original and predicted values
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'Actual': original_data['Close'].values[train_size:],
        'Predicted': np.append(np.repeat(np.nan, lookback), predictions.flatten())[:len(original_data)-train_size]
    }, index=original_data.index[train_size:])
    
    return pred_df

# Function to generate buy/sell signals
def generate_signals(predictions, window=5, threshold=0.01):
    """
    Generate buy/sell signals based on predicted trends
    
    Parameters:
    - predictions: DataFrame with predictions
    - window: Window size for trend calculation
    - threshold: Threshold for buy/sell decision
    
    Returns:
    - DataFrame with signals
    """
    signals = predictions.copy()
    
    # Calculate short-term momentum
    signals['Momentum'] = signals['Predicted'].pct_change(periods=window).fillna(0)
    
    # Generate signals
    signals['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
    signals.loc[signals['Momentum'] > threshold, 'Signal'] = 1  # Buy signal
    signals.loc[signals['Momentum'] < -threshold, 'Signal'] = -1  # Sell signal
    
    # Clean up signals (avoid repeated signals)
    signals['PrevSignal'] = signals['Signal'].shift(1).fillna(0)
    signals.loc[(signals['Signal'] == signals['PrevSignal']) & (signals['Signal'] != 0), 'Signal'] = 0
    
    return signals.drop('PrevSignal', axis=1)

# Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Stock Price Prediction with LSTM")
    
    # App title and description
    st.title("Stock Price Prediction with LSTM")
    st.markdown("""
    This app uses an LSTM neural network to predict stock prices and generate buy/sell signals.
    The model is trained on historical data and provides predictions with a focus on 30-minute accuracy.
    """)
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
    period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.sidebar.selectbox("Data Interval", ["15m", "30m", "1h"], index=1)
    lookback = st.sidebar.slider("Lookback Window (Time Steps)", 6, 24, 12)
    
    # Advanced options (hidden by default)
    with st.sidebar.expander("Advanced Model Options"):
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
        epochs = st.slider("Number of Epochs", 10, 100, 50, 5)
        batch_size = st.slider("Batch Size", 16, 64, 32, 8)
        signal_window = st.slider("Signal Window", 3, 10, 5, 1)
        signal_threshold = st.slider("Signal Threshold", 0.005, 0.02, 0.01, 0.001)
    
    # Button to run model
    if st.sidebar.button("Run Model"):
        # Show loading message
        with st.spinner("Fetching stock data and training model..."):
            try:
                # Get stock data
                st.session_state.data = get_stock_data(ticker, period, interval)
                
                if len(st.session_state.data) == 0:
                    st.error(f"No data found for ticker: {ticker}. Please check the symbol and try again.")
                    return
                
                # Prepare data
                X_train, y_train, X_test, y_test, scaler, train_size = prepare_data(
                    st.session_state.data, lookback=lookback
                )
                
                # Build and train model
                model = build_lstm_model((X_train.shape[1], 1), dropout_rate)
                model, history = train_model(
                    model, X_train, y_train, 
                    epochs=epochs, batch_size=batch_size
                )
                
                # Make predictions
                predictions = make_predictions(
                    model, X_test, scaler, lookback, 
                    st.session_state.data, train_size
                )
                
                # Generate signals
                st.session_state.signals = generate_signals(
                    predictions, window=signal_window, 
                    threshold=signal_threshold
                )
                
                st.session_state.model_run = True
                st.session_state.ticker = ticker
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
    
    # Display results if model has been run
    if 'model_run' in st.session_state and st.session_state.model_run:
        # Display stock info
        stock_info = yf.Ticker(st.session_state.ticker).info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${stock_info.get('currentPrice', 'N/A')}")
        
        with col2:
            st.metric("Market Cap", f"${stock_info.get('marketCap', 'N/A'):,}")
        
        with col3:
            st.metric("52 Week Range", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')} - ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
        
        # Create plots
        signals = st.session_state.signals
        
        # Main chart with predictions and signals
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          row_heights=[0.7, 0.3],
                          subplot_titles=("Price Prediction", "Buy/Sell Signals"))
        
        # Price predictions
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['Actual'], name="Actual Price",
                      line=dict(color='royalblue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['Predicted'], name="Predicted Price",
                      line=dict(color='orange', dash='dot')),
            row=1, col=1
        )
        
        # Add buy signals
        buy_signals = signals[signals['Signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Actual'],
                          mode='markers', name="Buy Signal",
                          marker=dict(color='green', size=10, symbol='triangle-up')),
                row=1, col=1
            )
        
        # Add sell signals
        sell_signals = signals[signals['Signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Actual'],
                          mode='markers', name="Sell Signal",
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=1, col=1
            )
        
        # Add momentum indicator
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['Momentum'],
                      name="Momentum", line=dict(color='purple')),
            row=2, col=1
        )
        
        # Add threshold lines
        threshold = signal_threshold
        fig.add_trace(
            go.Scatter(x=signals.index, y=[threshold] * len(signals),
                      name="Buy Threshold", line=dict(color='green', dash='dash')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=signals.index, y=[-threshold] * len(signals),
                      name="Sell Threshold", line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{st.session_state.ticker} Stock Price Prediction",
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal statistics
        st.subheader("Signal Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Buy Signals", len(buy_signals))
        
        with col2:
            st.metric("Sell Signals", len(sell_signals))
        
        with col3:
            # Calculate prediction accuracy (RMSE)
            from sklearn.metrics import mean_squared_error
            valid_idx = ~np.isnan(signals['Predicted'])
            rmse = np.sqrt(mean_squared_error(signals.loc[valid_idx, 'Actual'], 
                                             signals.loc[valid_idx, 'Predicted']))
            st.metric("Prediction RMSE", f"{rmse:.4f}")
        
        # Recent signals
        st.subheader("Recent Signals")
        recent_signals = signals.tail(10).copy()
        recent_signals['Signal'] = recent_signals['Signal'].map({0: "Hold", 1: "Buy", -1: "Sell"})
        st.dataframe(recent_signals[['Actual', 'Predicted', 'Momentum', 'Signal']])
        
        # Add current prediction for the next 30 minutes
        st.subheader("Current Prediction (Next 30 minutes)")
        
        last_signal = signals['Signal'].iloc[-1]
        signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
        signal_color = "green" if last_signal == 1 else "red" if last_signal == -1 else "gray"
        
        st.markdown(f"<h3 style='color:{signal_color}'>Current Signal: {signal_text}</h3>", unsafe_allow_html=True)
        
        # Calculate expected price change
        expected_change = signals['Momentum'].iloc[-1] * signals['Actual'].iloc[-1]
        change_direction = "increase" if expected_change > 0 else "decrease"
        
        st.markdown(f"""
        Based on the current momentum of {signals['Momentum'].iloc[-1]:.4f}, we expect the price to {change_direction}
        by approximately ${abs(expected_change):.2f} in the next time interval.
        
        **Current Price:** ${signals['Actual'].iloc[-1]:.2f}  
        **Predicted Price:** ${signals['Actual'].iloc[-1] + expected_change:.2f}
        """)
        
        # Disclaimer
        st.warning("""
        **Disclaimer:** This prediction is based on historical patterns and machine learning models.
        Financial markets are inherently unpredictable and these predictions should not be used as
        financial advice. Always conduct your own research before making investment decisions.
        """)

# Run the app
if __name__ == "__main__":
    main()