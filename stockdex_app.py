import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pyarrow as pa # Import pyarrow

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
        hist = ticker.history(period="max") # Get max history for charts/performance
        # Resample if history is very long
        hist = resample_history_if_needed(hist)
        return info, hist
    except Exception as e:
        st.error(f"Could not fetch details for {ticker_symbol}: {e}")
        return None, None

# --- Helper Functions ---

# Function to safely get numeric value, returning None if not numeric
def get_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None

# Function to highlight max/min values in a row (assuming higher is better for simplicity)
# Handles non-numeric types and NaNs gracefully
def highlight_max_min(s):
    numeric_vals = pd.to_numeric(s, errors='coerce') # Convert to numeric, non-convertibles become NaN
    if numeric_vals.isna().all() or len(numeric_vals.dropna()) < 2: # Don't highlight if all NaN or fewer than 2 numbers
        return [''] * len(s)

    max_val = numeric_vals.max()
    min_val = numeric_vals.min()

    # Create default styling (no background)
    styles = [''] * len(s)

    for i, val in enumerate(numeric_vals):
        if pd.notna(val):
            if val == max_val:
                styles[i] = 'background-color: lightgreen'
            elif val == min_val:
                 styles[i] = 'background-color: lightcoral' # Use lightcoral instead of red for readability
    return styles

# Function to resample history if it spans too many years
def resample_history_if_needed(history_df, threshold_years=10):
    if history_df.empty:
        return history_df
    # Calculate the time span
    time_span = history_df.index.max() - history_df.index.min()
    # Check if resampling is needed (e.g., > 10 years)
    if time_span > pd.Timedelta(days=threshold_years * 365.25):
        # Resample to weekly frequency, taking the mean. Use 'W-MON' for Monday-based weeks.
        # Keep only 'Close' price for simplicity in plotting
        history_df = history_df[['Close']].resample('W-MON').mean()
        history_df = history_df.dropna() # Drop weeks with no data
    return history_df

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="StockEdex", page_icon="📈") # Added page icon
st.title("📈 StockEdex - Top 50 S&P 500 Stocks") # Removed "by Market Cap" for brevity

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
st.sidebar.header("Select Stock(s) for Details or Comparison")
# Use multiselect instead of selectbox
selected_tickers = st.sidebar.multiselect( # Changed variable name
    "Choose stocks from the Top 50 (by Market Cap):",
    top_50_tickers,
    default=top_50_tickers[0:1] # Default to the first stock initially
)

# --- Main Area ---

if not selected_tickers:
    st.warning("👈 Please select at least one stock from the sidebar.") # Added emoji
    st.stop()

# --- Display Single Stock Details ---
if len(selected_tickers) == 1:
    selected_ticker = selected_tickers[0]
    st.header(f"📊 Details for {selected_ticker}") # Added icon
    info, history = get_stock_details(selected_ticker)

    if info:
        # --- Company Overview Section ---
        with st.container(border=True): # Use border for visual separation
            st.subheader(":briefcase: Company Overview") # Added icon
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="**Company Name**", value=info.get('longName', 'N/A'))
                st.metric(label="**Sector**", value=info.get('sector', 'N/A'))
                st.metric(label="**Industry**", value=info.get('industry', 'N/A'))
                website = info.get('website', 'N/A')
                if website and website != 'N/A':
                     st.markdown(f"**Website:** [{website}]({website})") # Make website clickable
                else:
                    st.metric(label="**Website**", value='N/A')


            with col2:
                st.metric(label="**Market Cap**", value=f"${info.get('marketCap', 0):,}")
                st.metric(label="**Current Price**", value=f"${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
                st.metric(label="**Day High**", value=f"${info.get('dayHigh', 0):.2f}")
                st.metric(label="**Day Low**", value=f"${info.get('dayLow', 0):.2f}")

        st.divider() # Add visual separation

        # --- Business Summary Section ---
        with st.container(border=True):
             st.subheader(":page_facing_up: Business Summary") # Added icon
             st.write(info.get('longBusinessSummary', 'No summary available.'))

        st.divider() # Add visual separation

        # --- Key Statistics & Valuation Section ---
        with st.container(border=True):
            st.subheader(":calculator: Key Statistics & Valuation") # Added icon
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            # Helper function to format metrics
            def format_metric(value, format_str="{:.2f}"):
                if isinstance(value, (int, float)):
                    try:
                        return format_str.format(value)
                    except (ValueError, TypeError):
                        return 'N/A'
                return 'N/A'

            def format_currency(value):
                 if isinstance(value, (int, float)):
                     return f"${value:,.0f}"
                 return 'N/A'

            def format_percent(value):
                 if isinstance(value, (int, float)):
                     return f"{value*100:.2f}%"
                 return 'N/A'


            with stats_col1:
                st.metric("P/E Ratio (TTM)", format_metric(info.get('trailingPE')))
                st.metric("Forward P/E", format_metric(info.get('forwardPE')))
                st.metric("PEG Ratio", format_metric(info.get('pegRatio')))
                st.metric("Beta", format_metric(info.get('beta')))

            with stats_col2:
                st.metric("Price to Sales (TTM)", format_metric(info.get('priceToSalesTrailing12Months')))
                st.metric("Price to Book", format_metric(info.get('priceToBook')))
                st.metric("Dividend Yield", format_percent(info.get('dividendYield')))
                st.metric("Enterprise Value", format_currency(info.get('enterpriseValue')))

            with stats_col3:
                st.metric("EV/Revenue", format_metric(info.get('enterpriseToRevenue')))
                st.metric("EV/EBITDA", format_metric(info.get('enterpriseToEbitda')))
                st.metric("52 Week High", format_currency(info.get('fiftyTwoWeekHigh')))
                st.metric("52 Week Low", format_currency(info.get('fiftyTwoWeekLow')))

        st.divider() # Add visual separation

        # --- Chart Section ---
        if history is not None and not history.empty:
             with st.container(border=True):
                st.subheader(":chart_with_upwards_trend: Stock Price History (Max)") # Added icon
                st.line_chart(history['Close'])
        else:
            st.warning("Could not retrieve price history.")

    else:
        st.error(f"Could not retrieve data for {selected_ticker}.")

# --- Display Stock Comparison ---
elif len(selected_tickers) > 1:
    st.header(f"🆚 Comparison for: {', '.join(selected_tickers)}") # Added icon

    comparison_data = {}
    history_data = {}
    metrics_to_compare = [
        'longName', 'sector', 'industry', 'marketCap', 'currentPrice',
        'trailingPE', 'forwardPE', 'pegRatio', 'priceToSalesTrailing12Months',
        'priceToBook', 'dividendYield', 'beta', 'enterpriseValue',
        'enterpriseToRevenue', 'enterpriseToEbitda', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'
    ]
    metrics_display_names = {
        'longName': 'Company Name', 'sector': 'Sector', 'industry': 'Industry',
        'marketCap': 'Market Cap ($)', 'currentPrice': 'Current Price ($)',
        'trailingPE': 'P/E (TTM)', 'forwardPE': 'Forward P/E', 'pegRatio': 'PEG Ratio',
        'priceToSalesTrailing12Months': 'P/S (TTM)', 'priceToBook': 'P/B',
        'dividendYield': 'Dividend Yield (%)', 'beta': 'Beta', 'enterpriseValue': 'Enterprise Value ($)',
        'enterpriseToRevenue': 'EV/Revenue', 'enterpriseToEbitda': 'EV/EBITDA',
        'fiftyTwoWeekHigh': '52 Week High ($)', 'fiftyTwoWeekLow': '52 Week Low ($)'
    }

    has_data = False
    for ticker in selected_tickers:
        info, history = get_stock_details(ticker)
        if info:
            has_data = True
            stock_metrics = {}
            for metric in metrics_to_compare:
                 stock_metrics[metric] = info.get(metric, 'N/A')
            comparison_data[ticker] = stock_metrics

            # Format specific metrics
            comparison_data[ticker]['marketCap'] = info.get('marketCap', 0)
            comparison_data[ticker]['enterpriseValue'] = info.get('enterpriseValue', 0)
            comparison_data[ticker]['dividendYield'] = info.get('dividendYield', 0) * 100 # Store as percentage value

        if history is not None and not history.empty:
            # Resample individual history if needed before adding to comparison dict
            resampled_hist = resample_history_if_needed(history)
            if not resampled_hist.empty:
                 history_data[ticker] = resampled_hist['Close']

    if not has_data:
        st.error("Could not retrieve data for any of the selected stocks.")
        st.stop()

    # --- Comparison Table ---
    with st.container(border=True): # Add border to comparison table container
        st.subheader(":bar_chart: Key Metrics Comparison") # Added icon
        # Create DataFrame and transpose for better readability (tickers as columns)
        compare_df = pd.DataFrame.from_dict(comparison_data, orient='index')

        # Select only the metrics we want to display and rename columns
        compare_df = compare_df[metrics_to_compare] # Ensure order
        compare_df = compare_df.rename(columns=metrics_display_names)
        compare_df = compare_df.transpose() # Metrics as rows, tickers as columns

        # Define formats first
        formats = {
            'Market Cap ($)': "{:,.0f}",
            'Current Price ($)': "{:,.2f}",
            'P/E (TTM)': "{:.2f}",
            'Forward P/E': "{:.2f}",
            'PEG Ratio': "{:.2f}",
            'P/S (TTM)': "{:.2f}",
            'P/B': "{:.2f}",
            'Dividend Yield (%)': "{:.2f}%",
            'Beta': "{:.2f}",
            'Enterprise Value ($)': "{:,.0f}",
            'EV/Revenue': "{:.2f}",
            'EV/EBITDA': "{:.2f}",
            '52 Week High ($)': "{:,.2f}",
            '52 Week Low ($)': "{:,.2f}",
        }

        # Ensure columns intended for numeric formatting/styling are numeric
        # This helps prevent Arrow serialization errors with styled dataframes
        for metric_display_name in formats:
            if metric_display_name in compare_df.index: # Check if the metric exists as an index (row name)
                compare_df.loc[metric_display_name] = pd.to_numeric(compare_df.loc[metric_display_name], errors='coerce')


        # Apply styling and formatting
        styled_df = compare_df.style.apply(highlight_max_min, axis=1).format(formats, na_rep="N/A")

        st.dataframe(styled_df, use_container_width=True) # Use container width

    st.divider() # Add visual separation

    # --- Comparison Chart ---
    with st.container(border=True): # Add border to comparison chart container
         st.subheader(":chart_with_upwards_trend: Price History Comparison (Max)") # Added icon and updated text
         if history_data:
             history_df = pd.DataFrame(history_data)
             # Removed normalization step
             # normalized_df = (history_df / history_df.iloc[0]) * 100

             # Plot individual charts for each stock
             for ticker in history_df.columns:
                 st.subheader(f"{ticker} Price History")
                 st.line_chart(history_df[ticker].dropna()) # Plot each stock's history, dropna for safety

         else:
             st.warning("Could not retrieve sufficient price history for comparison charts.")


# --- Footer Info ---
st.sidebar.markdown("---")
st.sidebar.info("Data sourced from Yahoo Finance via yfinance & Wikipedia.") 