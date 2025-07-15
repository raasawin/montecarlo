import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Cache the data for 24 hours to avoid excessive API calls and rate limiting
@st.cache_data(ttl=24*3600)
def get_latest_apple_data():
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")
    S0 = hist["Close"][-1]
    returns = hist["Close"].pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return S0, mu, sigma

def monte_carlo_stock_simulation(
    S0,                   # Initial stock price
    mu,                   # Expected return (annualized drift)
    sigma,                # Volatility (annualized standard deviation)
    T,                    # Time horizon in years
    num_simulations,
    num_steps_per_year=252,
    show_histogram=True,
    show_sample_paths=True,
    num_sample_paths=10
):
    dt = 1 / num_steps_per_year
    num_steps = int(T * num_steps_per_year)

    # Generate random normal values for all simulations and steps
    dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, num_steps))

    # Log return increments under GBM
    increments = (mu - 0.5 * sigma**2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=1)

    # Convert log returns to price paths
    S_t = S0 * np.exp(log_returns)
    S_t = np.hstack((np.full((num_simulations, 1), S0), S_t))  # Add initial price

    final_prices = S_t[:, -1]
    rise_probability = np.mean(final_prices > S0)
    falls = num_simulations - np.sum(final_prices > S0)

    ci_lower = np.percentile(final_prices, 2.5)
    ci_upper = np.percentile(final_prices, 97.5)

    # Display results
    st.write(f"Out of {num_simulations} simulations:")
    st.write(f"  Price rose in {int(rise_probability * num_simulations)} cases ({rise_probability*100:.2f}%)")
    st.write(f"  Price fell in {int(falls)} cases ({(1 - rise_probability)*100:.2f}%)")
    st.write(f"  95% confidence interval for final price: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Plot histogram of final prices
    if show_histogram:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.hist(final_prices, bins=50, alpha=0.6, color='skyblue', edgecolor='black')
        ax_hist.axvline(S0, color='red', linestyle='--', label='Initial Price')
        ax_hist.axvline(ci_lower, color='green', linestyle=':', label='95% CI Lower')
        ax_hist.axvline(ci_upper, color='green', linestyle=':', label='95% CI Upper')
        ax_hist.set_title("Distribution of Final Stock Prices")
        ax_hist.set_xlabel("Stock Price")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        ax_hist.grid(True)
        st.pyplot(fig_hist)

    # Plot sample price paths
    if show_sample_paths:
        fig_paths, ax_paths = plt.subplots(figsize=(10, 5))
        for i in range(min(num_sample_paths, num_simulations)):
            ax_paths.plot(S_t[i], linewidth=1, alpha=0.7)
        ax_paths.set_title("Sample Stock Price Paths")
        ax_paths.set_xlabel("Time Steps")
        ax_paths.set_ylabel("Stock Price")
        ax_paths.grid(True)
        st.pyplot(fig_paths)

    return rise_probability, final_prices, S_t


def main():
    st.title("Monte Carlo Simulation of Apple Stock Price")
    st.write("This app simulates Apple (AAPL) stock price paths using Geometric Brownian Motion based on real historical data fetched from Yahoo Finance (cached to avoid rate limits).")

    # Fetch cached data
    S0, mu, sigma = get_latest_apple_data()

    st.write(f"Latest Apple stock price: ${S0:.2f}")
    st.write(f"Estimated annual return (mu): {mu:.4f}")
    st.write(f"Estimated annual volatility (sigma): {sigma:.4f}")

    # User inputs
    T = st.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    num_simulations = st.number_input("Number of simulations", min_value=1000, max_value=50000, value=10000, step=1000)
    num_sample_paths = st.number_input("Number of sample paths to plot", min_value=1, max_value=50, value=10)

    if st.button("Run Simulation"):
        monte_carlo_stock_simulation(S0, mu, sigma, T, num_simulations, num_sample_paths=num_sample_paths)

if __name__ == "__main__":
    main()

