import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Cache the data fetching functions to improve performance
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_sp500_tickers():
    """Scrapes S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'id': 'constituents'})
        if table:
            df = pd.read_html(str(table))[0]
            # Clean ticker symbols (some might have suffixes like .B or . on Wiki)
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            return tickers
        else:
            st.error("Could not find the S&P 500 constituents table on Wikipedia.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        return []
    except Exception as e:
        st.error(f"An error occurred while parsing Wikipedia data: {e}")
        return []

@st.cache_data(ttl=3600)
def get_market_caps(tickers):
    """Fetches market caps for a list of tickers."""
    if not tickers:
        return pd.DataFrame()

    market_caps = {}
    batch_size = 100 # Process in batches to avoid overwhelming yfinance or getting rate-limited
    tickers_to_process = tickers[:] # Create a copy

    progress_bar = st.progress(0, text="Fetching market data...")
    total_tickers = len(tickers_to_process)
    processed_count = 0

    while tickers_to_process:
        batch = tickers_to_process[:batch_size]
        tickers_to_process = tickers_to_process[batch_size:]
        try:
            yf_tickers = yf.Tickers(batch)
            # Using history metadata as a proxy sometimes works better than info
            for ticker_symbol, ticker_obj in yf_tickers.tickers.items():
                 try:
                     # Attempt to get market cap from info first
                     info = ticker_obj.info
                     mc = info.get('marketCap')
                     if mc:
                         market_caps[ticker_symbol] = mc
                     else:
                         # Fallback if marketCap not in info (less common now, but good practice)
                         hist = ticker_obj.history(period="1d")
                         if not hist.empty:
                             # Calculate from price and sharesOutstanding if needed/available
                             price = hist['Close'].iloc[-1]
                             shares = info.get('sharesOutstanding')
                             if shares:
                                market_caps[ticker_symbol] = price * shares
                             else:
                                market_caps[ticker_symbol] = None # Mark as unavailable
                         else:
                            market_caps[ticker_symbol] = None # Mark as unavailable
                 except Exception:
                     # Handle cases where a specific ticker fails
                     market_caps[ticker_symbol] = None # Mark as unavailable

        except Exception as e:
            st.warning(f"Could not fetch data for batch: {batch}. Error: {e}")
            for t in batch:
                 if t not in market_caps:
                     market_caps[t] = None # Ensure failed tickers are marked

        processed_count += len(batch)
        progress = min(1.0, processed_count / total_tickers)
        progress_bar.progress(progress, text=f"Fetching market data... ({processed_count}/{total_tickers})")


    progress_bar.empty() # Remove progress bar when done
    mc_df = pd.DataFrame.from_dict(market_caps, orient='index', columns=['MarketCap'])
    mc_df = mc_df.dropna().sort_values('MarketCap', ascending=False)
    return mc_df

@st.cache_data(ttl=600) # Cache individual stock details for 10 minutes
def get_stock_details(ticker_symbol):
    """Fetches detailed information for a single stock ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y") # Get 1 year history for charts/performance
        return info, hist
    except Exception as e:
        st.error(f"Could not fetch details for {ticker_symbol}: {e}")
        return None, None

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("📈 StockEdex - Top 50 S&P 500 Stocks by Market Cap")

# --- Data Loading ---
sp500_tickers = get_sp500_tickers()

if not sp500_tickers:
    st.stop() # Stop execution if we can't get tickers

market_cap_df = get_market_caps(sp500_tickers)

if market_cap_df.empty:
    st.warning("Could not retrieve market cap data for S&P 500 stocks. Unable to determine top 50.")
    st.stop()

top_50_tickers = market_cap_df.head(50).index.tolist()

# --- Sidebar for Stock Selection ---
st.sidebar.header("Select Stock")
selected_ticker = st.sidebar.selectbox(
    "Choose a stock from the Top 50 (by Market Cap):",
    top_50_tickers,
    index=0 # Default to the first stock (largest market cap)
)

# --- Main Area for Stock Details ---
st.header(f"Details for {selected_ticker}")

info, history = get_stock_details(selected_ticker)

if info:
    # Display basic info using columns
    st.subheader("Company Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="**Company Name**", value=info.get('longName', 'N/A'))
        st.metric(label="**Sector**", value=info.get('sector', 'N/A'))
        st.metric(label="**Industry**", value=info.get('industry', 'N/A'))
        st.metric(label="**Website**", value=info.get('website', 'N/A'))

    with col2:
        st.metric(label="**Market Cap**", value=f"${info.get('marketCap', 0):,}")
        st.metric(label="**Current Price**", value=f"${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
        st.metric(label="**Day High**", value=f"${info.get('dayHigh', 0):.2f}")
        st.metric(label="**Day Low**", value=f"${info.get('dayLow', 0):.2f}")


    # Display Summary
    st.subheader("Business Summary")
    st.write(info.get('longBusinessSummary', 'No summary available.'))

    # Display Key Stats & Valuation
    st.subheader("Key Statistics & Valuation")
    stats_col1, stats_col2, stats_col3 = st.columns(3)

    with stats_col1:
        st.metric("P/E Ratio (TTM)", f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A')
        st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A')
        st.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A'):.2f}" if isinstance(info.get('pegRatio'), (int, float)) else 'N/A')
        st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else 'N/A')

    with stats_col2:
        st.metric("Price to Sales (TTM)", f"{info.get('priceToSalesTrailing12Months', 'N/A'):.2f}" if isinstance(info.get('priceToSalesTrailing12Months'), (int, float)) else 'N/A')
        st.metric("Price to Book", f"{info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else 'N/A')
        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if isinstance(info.get('dividendYield'), (int, float)) else 'N/A')
        st.metric("Enterprise Value", f"${info.get('enterpriseValue', 0):,}" if info.get('enterpriseValue') else 'N/A')

    with stats_col3:
        st.metric("EV/Revenue", f"{info.get('enterpriseToRevenue', 'N/A'):.2f}" if isinstance(info.get('enterpriseToRevenue'), (int, float)) else 'N/A')
        st.metric("EV/EBITDA", f"{info.get('enterpriseToEbitda', 'N/A'):.2f}" if isinstance(info.get('enterpriseToEbitda'), (int, float)) else 'N/A')
        st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
        st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

    # Display Chart
    if history is not None and not history.empty:
        st.subheader("1-Year Stock Price History")
        st.line_chart(history['Close'])
    else:
        st.warning("Could not retrieve price history.")

else:
    st.error(f"Could not retrieve data for {selected_ticker}.")

st.sidebar.markdown("---")
st.sidebar.info("Data sourced from Yahoo Finance via yfinance & Wikipedia.") 