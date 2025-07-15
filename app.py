import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

def get_stock_data(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def monte_carlo_simulation(S0, mu, sigma, T=1, dt=1/252, num_simulations=1000):
    num_steps = int(T / dt)
    simulations = np.zeros((num_simulations, num_steps))
    for i in range(num_simulations):
        prices = [S0]
        for _ in range(num_steps-1):
            shock = np.random.normal(loc=(mu - 0.5 * sigma**2)*dt, scale=sigma * np.sqrt(dt))
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        simulations[i] = prices
    return simulations

def calculate_confidence(simulations, S0):
    # Final prices after simulation period
    final_prices = simulations[:, -1]
    # Calculate percentiles for confidence intervals
    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)
    
    expected_return = (p50 - S0) / S0 * 100
    conf_interval = ((p95 - p5) / S0) * 100  # Width of 90% CI in %
    
    # Confidence score: higher if interval is narrow relative to expected return magnitude
    if expected_return != 0:
        confidence = max(0, 100 - (conf_interval / abs(expected_return)) * 100)
    else:
        confidence = 0
    
    return expected_return, confidence, p5, p50, p95

def main():
    st.title("Monte Carlo Stock Price Predictor with Confidence")
    ticker = st.text_input("Enter stock ticker symbol", "AAPL").upper()
    
    data = get_stock_data(ticker)
    if data is None or data.empty:
        st.warning("No data available or failed to fetch.")
        return
    
    # Calculate log returns mean and std dev as drift and volatility
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    mu = log_returns.mean() * 252  # annualized drift
    sigma = log_returns.std() * np.sqrt(252)  # annualized volatility
    S0 = data['Close'][-1]

    simulations = monte_carlo_simulation(S0, mu, sigma)
    expected_return, confidence, p5, p50, p95 = calculate_confidence(simulations, S0)
    
    st.subheader(f"Stock: {ticker}")
    st.write(f"Current Price: ${S0:.2f}")
    
    st.write(f"Estimated expected price change over 1 year: {expected_return:.2f}%")
    st.write(f"Confidence in this estimate: {confidence:.1f}%")
    st.write(f"90% Confidence Interval for price after 1 year:")
    st.write(f"- 5th percentile: ${p5:.2f} ({(p5-S0)/S0*100:.2f}%)")
    st.write(f"- Median (50th percentile): ${p50:.2f} ({(p50-S0)/S0*100:.2f}%)")
    st.write(f"- 95th percentile: ${p95:.2f} ({(p95-S0)/S0*100:.2f}%)")

    # Summary decision helper
    if confidence < 30:
        st.warning("The prediction confidence is low — results are uncertain.")
    elif expected_return > 0:
        st.success(f"Model suggests the stock price is likely to rise by about {expected_return:.2f}%.")
    else:
        st.error(f"Model suggests the stock price may fall by about {abs(expected_return):.2f}%.")

    # Plot simulation results
    st.line_chart(pd.DataFrame(simulations.T))

if __name__ == "__main__":
    main()
