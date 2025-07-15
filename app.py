import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch live Apple stock data
def get_latest_apple_data():
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")
    current_price = hist["Close"][-1]
    returns = hist["Close"].pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return current_price, mu, sigma

# Monte Carlo Simulation
def monte_carlo_stock_simulation(S0, mu, sigma, T, num_simulations, steps_per_year=252):
    dt = 1 / steps_per_year
    steps = int(T * steps_per_year)
    dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=1)
    S_t = S0 * np.exp(log_returns)
    S_t = np.hstack((np.full((num_simulations, 1), S0), S_t))
    final_prices = S_t[:, -1]

    rise_prob = np.mean(final_prices > S0)
    ci_lower = np.percentile(final_prices, 2.5)
    ci_upper = np.percentile(final_prices, 97.5)

    return S_t, final_prices, rise_prob, ci_lower, ci_upper

# Streamlit UI
st.title("📈 Apple Stock Monte Carlo Simulation")
st.markdown("This app simulates future AAPL stock prices using Monte Carlo simulation based on historical volatility.")

with st.spinner("Fetching real-time AAPL data..."):
    S0, mu, sigma = get_latest_apple_data()

st.sidebar.header("Simulation Settings")
T = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0, 0.1)
num_simulations = st.sidebar.slider("Number of Simulations", 100, 20000, 10000, 1000)
steps_per_year = st.sidebar.slider("Steps Per Year", 50, 500, 252, 50)
num_sample_paths = st.sidebar.slider("Sample Paths to Plot", 1, 20, 10)

st.write(f"**Initial AAPL Price:** ${S0:.2f}")
st.write(f"**Annualized Return (μ):** {mu*100:.2f}%")
st.write(f"**Annualized Volatility (σ):** {sigma*100:.2f}%")

# Run simulation
S_t, final_prices, rise_prob, ci_lower, ci_upper = monte_carlo_stock_simulation(
    S0, mu, sigma, T, num_simulations, steps_per_year
)

# Show results
st.subheader("📊 Final Price Distribution")
st.write(f"**Rise Probability:** {rise_prob*100:.2f}%")
st.write(f"**95% Confidence Interval:** [{ci_lower:.2f}, {ci_upper:.2f}]")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(final_prices, bins=50, color='skyblue', edgecolor='black')
ax1.axvline(S0, color='red', linestyle='--', label='Initial Price')
ax1.axvline(ci_lower, color='green', linestyle=':', label='95% CI Lower')
ax1.axvline(ci_upper, color='green', linestyle=':', label='95% CI Upper')
ax1.set_title("Distribution of Final Simulated AAPL Prices")
ax1.set_xlabel("Price")
ax1.set_ylabel("Frequency")
ax1.legend()
st.pyplot(fig1)

st.subheader("📈 Sample Simulated Paths")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for i in range(min(num_sample_paths, num_simulations)):
    ax2.plot(S_t[i], linewidth=1, alpha=0.7)
ax2.set_title("Sample Simulated AAPL Price Paths")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Price")
st.pyplot(fig2)
