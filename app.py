import yfinance as yf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def get_latest_apple_data():
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")
    returns = hist['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    S0 = hist['Close'][-1]
    return S0, mu, sigma

def monte_carlo_simulation(S0, mu, sigma, T=1, steps=252, num_simulations=10000):
    dt = T / steps
    price_paths = np.zeros((steps + 1, num_simulations))
    price_paths[0] = S0
    
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z)
    
    return price_paths

def main():
    st.title("Monte Carlo Stock Price Simulation")
    
    S0, mu, sigma = get_latest_apple_data()
    st.write(f"Latest price: ${S0:.2f}")
    st.write(f"Expected daily return (mu): {mu:.5f}")
    st.write(f"Daily volatility (sigma): {sigma:.5f}")
    
    T = st.slider("Time horizon (years)", 0.1, 2.0, 1.0)
    steps = int(T * 252)
    num_simulations = st.slider("Number of simulations", 1000, 20000, 10000)
    
    if st.button("Run simulation"):
        with st.spinner("Simulating..."):
            price_paths = monte_carlo_simulation(S0, mu, sigma, T, steps, num_simulations)
            final_prices = price_paths[-1]
            
            lower_pct = np.percentile(final_prices, 5)
            upper_pct = np.percentile(final_prices, 95)
            median_price = np.percentile(final_prices, 50)
            
            downside_pct = (lower_pct - S0) / S0 * 100
            upside_pct = (upper_pct - S0) / S0 * 100
            median_pct = (median_price - S0) / S0 * 100
            
            st.write(f"Estimated 5th percentile price: ${lower_pct:.2f} ({downside_pct:.2f}% fall)")
            st.write(f"Estimated median price: ${median_price:.2f} ({median_pct:.2f}% change)")
            st.write(f"Estimated 95th percentile price: ${upper_pct:.2f} (+{upside_pct:.2f}% rise)")
            
            fig, ax = plt.subplots()
            ax.hist(final_prices, bins=50, alpha=0.7, color='blue')
            ax.axvline(lower_pct, color='red', linestyle='dashed', label='5th percentile')
            ax.axvline(median_price, color='orange', linestyle='dashed', label='Median')
            ax.axvline(upper_pct, color='green', linestyle='dashed', label='95th percentile')
            ax.legend()
            ax.set_xlabel('Price at time T')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
