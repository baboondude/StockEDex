import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pyarrow as pa # Import pyarrow
from datetime import datetime, timedelta # Import datetime

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

    # Use spinner instead of progress bar for this initial fetch
    # progress_bar = st.progress(0, text="Fetching market data...")
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

        # Optional: Add progress update logic here if spinner isn't enough
        # processed_count += len(batch)
        # progress = min(1.0, processed_count / total_tickers)
        # Consider updating a status text if needed

    # progress_bar.empty() # Remove progress bar when done
    mc_df = pd.DataFrame.from_dict(market_caps, orient='index', columns=['MarketCap'])
    mc_df = mc_df.dropna().sort_values('MarketCap', ascending=False)
    return mc_df

@st.cache_data(ttl=600) # Cache individual stock details for 10 minutes
def get_stock_details(ticker_symbol):
    """Fetches detailed information and MAX price history for a single stock ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch info first
        info = ticker.info
        # Always fetch max history for filtering later
        hist = ticker.history(period="max")
        # Resample if history is very long (only affects 'Max' view display)
        hist_resampled = resample_history_if_needed(hist.copy()) # Pass copy to avoid modifying original max hist
        return info, hist # Return original max history
    except Exception as e:
        st.error(f"Could not fetch details for {ticker_symbol}: {e}")
        return None, None

# --- Helper Functions ---

# Function to safely get numeric value, returning None if not numeric
def get_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None

# Function to highlight max/min values in a row
def highlight_max_min(s):
    numeric_vals = pd.to_numeric(s, errors='coerce')
    if numeric_vals.isna().all() or len(numeric_vals.dropna()) < 2:
        return [''] * len(s)

    max_val = numeric_vals.max()
    min_val = numeric_vals.min()

    styles = [''] * len(s)
    for i, val in enumerate(numeric_vals):
        if pd.notna(val):
            # Use more subtle, theme-consistent colors
            if val == max_val:
                styles[i] = 'background-color: #d1ecf1' # Light Blueish
            elif val == min_val:
                 styles[i] = 'background-color: #f8d7da' # Light Pinkish
    return styles

# Function to resample history if it spans too many years
def resample_history_if_needed(history_df, threshold_years=10):
    if history_df.empty:
        return history_df
    time_span = history_df.index.max() - history_df.index.min()
    if time_span > pd.Timedelta(days=threshold_years * 365.25):
        history_df = history_df[['Close']].resample('W-MON').mean()
        history_df = history_df.dropna()
    return history_df

# Function to filter history based on selected period
def filter_history(history_df, period):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    end_date = history_df.index.max()
    start_date = None

    if period == "1m":
        start_date = end_date - pd.DateOffset(months=1)
    elif period == "6m":
        start_date = end_date - pd.DateOffset(months=6)
    elif period == "1y":
        start_date = end_date - pd.DateOffset(years=1)
    elif period == "5y":
        start_date = end_date - pd.DateOffset(years=5)
    elif period == "Max":
        # For 'Max', apply resampling if needed for display
        return resample_history_if_needed(history_df.copy())
    else: # Default to Max if period is unrecognized
        return resample_history_if_needed(history_df.copy())

    # Filter data >= start_date
    filtered_df = history_df[history_df.index >= start_date]
    return filtered_df

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="StockEdex", page_icon="📈")
st.title("📈 StockEdex - Top 50 S&P 500 Stocks")

# --- Data Loading with Spinners ---
with st.spinner('Fetching S&P 500 tickers...'):
    sp500_tickers = get_sp500_tickers()

if not sp500_tickers:
    st.error("Failed to load S&P 500 tickers. Please try refreshing.")
    st.stop()

with st.spinner('Fetching market cap data...'):
    market_cap_df = get_market_caps(sp500_tickers)

if market_cap_df.empty:
    st.warning("Could not retrieve market cap data. Some features might be limited.")
    # Don't stop entirely, maybe user wants to search other stocks later?
    top_50_tickers = [] # Set to empty if no market cap data
else:
    top_50_tickers = market_cap_df.head(50).index.tolist()

# --- Sidebar for Stock Selection ---
st.sidebar.header("Select Stock(s) for Details or Comparison")
selected_tickers = st.sidebar.multiselect(
    "Choose stocks from the Top 50 (by Market Cap):",
    options=top_50_tickers, # Use options parameter
    default=top_50_tickers[0:1] if top_50_tickers else [] # Handle empty case
)

# --- Time Period Selection (Moved to Sidebar) ---
time_periods = ["1m", "6m", "1y", "5y", "Max"]
selected_period = st.sidebar.radio( # Changed from st.radio to st.sidebar.radio
    "Select Chart Time Period:",
    options=time_periods,
    index=len(time_periods) - 1, # Default to 'Max'
    horizontal=False, # Better layout in sidebar
    key='time_period_selector'
)

# --- Main Area ---

if not selected_tickers:
    st.warning("👈 Please select at least one stock from the sidebar.")
    st.stop()

# Remove the divider that was after the time selector in main area
# st.divider() # Separate selector from content

# --- Display Single Stock Details ---
if len(selected_tickers) == 1:
    selected_ticker = selected_tickers[0]
    st.header(f"📊 Details for {selected_ticker}")

    # Fetch data using spinner
    with st.spinner(f'Fetching details for {selected_ticker}...'):
        ticker_obj = yf.Ticker(selected_ticker)
        info = None
        history_max = None
        news = []
        financials = pd.DataFrame()
        earnings = pd.DataFrame()
        try:
            info = ticker_obj.info
            history_max = ticker_obj.history(period="max")
            news = ticker_obj.news
            financials = ticker_obj.financials # Annual financials
            earnings = ticker_obj.earnings # Annual earnings
        except Exception as e:
            st.error(f"An error occurred while fetching data for {selected_ticker}: {e}")
            # Allow partial data display if some parts failed

    if info: # Proceed if at least basic info was fetched
        # --- Define Tabs (Add News & Fundamentals) ---
        tab_overview, tab_stats, tab_chart, tab_fundamentals, tab_news = st.tabs([
            "📄 Overview",
            "🔢 Key Stats",
            "📈 Chart",
            "💰 Fundamentals",
            "📰 News"
        ])

        # --- Overview Tab --- 
        with tab_overview:
            col1, col2 = st.columns([1, 3]) # Adjust column ratio for logo
            with col1:
                # --- Logo --- 
                logo_url = info.get('logo_url')
                if logo_url:
                    st.image(logo_url, width=100) # Display logo
                else:
                    st.markdown("_(No Logo Available)_")

                # Display website below logo if available
                website = info.get('website', 'N/A')
                if website and website != 'N/A':
                    st.link_button("Visit Website", website)
                else:
                    st.metric(label="**Website**", value='N/A')

            with col2:
                # Use subcolumns for better alignment of metrics
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric(label="**Company Name**", value=info.get('longName', 'N/A'))
                    st.metric(label="**Sector**", value=info.get('sector', 'N/A'))
                    st.metric(label="**Industry**", value=info.get('industry', 'N/A'))
                with subcol2:
                    st.metric(label="**Market Cap**", value=f"${info.get('marketCap', 0):,}")
                    st.metric(label="**Current Price**", value=f"${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
                    st.metric(label="**Day High**", value=f"${info.get('dayHigh', 0):.2f}")
                    st.metric(label="**Day Low**", value=f"${info.get('dayLow', 0):.2f}")

            st.divider()
            st.subheader(":page_facing_up: Business Summary")
            st.write(info.get('longBusinessSummary', 'No summary available.'))

        # --- Key Stats Tab ---
        with tab_stats:
            # --- Key Statistics & Valuation Section (Inside Tab) ---
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

        # --- Chart Tab ---
        with tab_chart:
            if history_max is not None: # Check if history was fetched
                history_filtered = filter_history(history_max, selected_period)
                if not history_filtered.empty:
                    st.subheader(f"Stock Price History ({selected_period})")
                    st.line_chart(history_filtered['Close'])
                else:
                    st.warning(f"No historical data available for the selected '{selected_period}' period.")
            else:
                st.warning("Could not retrieve price history.")

        # --- Fundamentals Tab ---
        with tab_fundamentals:
            st.subheader("Annual Financial Trends")
            # --- Revenue Chart ---
            if isinstance(financials, pd.DataFrame) and not financials.empty and 'Total Revenue' in financials.index:
                try:
                    # Transpose for plotting (years as columns -> index)
                    revenue_df = financials.loc[['Total Revenue']].transpose().sort_index()
                    revenue_df = revenue_df.dropna()
                    revenue_df.index = revenue_df.index.strftime('%Y') # Format index to year
                    if not revenue_df.empty:
                        st.markdown("**Total Revenue**")
                        st.bar_chart(revenue_df)
                    else:
                         st.info("Revenue data not available or incomplete.")
                except Exception as e:
                    st.warning(f"Could not process revenue data: {e}")
            else:
                st.info("Revenue data not available.")

            st.divider()

            # --- Earnings Chart ---
            if isinstance(earnings, pd.DataFrame) and not earnings.empty and 'Earnings' in earnings.columns:
                try:
                    earnings_df = earnings[['Earnings']].sort_index()
                    earnings_df = earnings_df.dropna()
                    if not earnings_df.empty:
                        st.markdown("**Net Earnings**")
                        st.bar_chart(earnings_df)
                    else:
                        st.info("Earnings data not available or incomplete.")
                except Exception as e:
                    st.warning(f"Could not process earnings data: {e}")
            else:
                 # Fallback: Try Net Income from financials if earnings is empty or not DataFrame
                 if isinstance(financials, pd.DataFrame) and not financials.empty and 'Net Income' in financials.index:
                     try:
                         net_income_df = financials.loc[['Net Income']].transpose().sort_index()
                         net_income_df = net_income_df.dropna()
                         net_income_df.index = net_income_df.index.strftime('%Y')
                         if not net_income_df.empty:
                             st.markdown("**Net Income**")
                             st.bar_chart(net_income_df)
                         else:
                            st.info("Net Income data not available or incomplete.")
                     except Exception as e:
                        st.warning(f"Could not process Net Income data: {e}")
                 else:
                    st.info("Earnings/Net Income data not available.")

        # --- News Tab ---
        with tab_news:
            st.subheader("Recent News")
            if news: # Check if news list is not empty
                for item in news:
                    # Ensure essential keys exist
                    title = item.get('title')
                    publisher = item.get('publisher')
                    link = item.get('link')
                    # date = pd.to_datetime(item.get('providerPublishTime'), unit='s') # Convert Unix timestamp
                    
                    if title and link:
                        st.markdown(f"**[{title}]({link})**")
                        if publisher:
                             st.caption(f"Source: {publisher}")
                        # st.caption(f"Published: {date.strftime('%Y-%m-%d %H:%M')}")
                        st.divider()
                    else:
                        # Optionally log or display minimal info if link/title missing
                        pass 
            else:
                st.info("No recent news found for this ticker.")

    elif selected_ticker: # If info failed but ticker was selected
         st.error(f"Could not retrieve sufficient data for {selected_ticker} to display details.")

# --- Display Stock Comparison ---
elif len(selected_tickers) > 1:
    # Comparison view remains unchanged, tabs not used here
    st.header(f"🆚 Comparison for: {', '.join(selected_tickers)}")

    comparison_data = {}
    history_data_max = {} # Store max history first
    # Define metrics earlier
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
    # Fetch all data with spinners
    with st.spinner(f'Fetching details for {len(selected_tickers)} stocks...'):
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
                 history_data_max[ticker] = history['Close'] # Store only Close series

    if not has_data:
        st.error("Could not retrieve data for any of the selected stocks.")
        st.stop()

    # --- Comparison Table ---
    with st.container(border=True):
        st.subheader(":bar_chart: Key Metrics Comparison")
        compare_df = pd.DataFrame.from_dict(comparison_data, orient='index')
        compare_df = compare_df[metrics_to_compare]
        compare_df = compare_df.rename(columns=metrics_display_names)
        compare_df = compare_df.transpose()

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

        for metric_display_name in formats:
            if metric_display_name in compare_df.index:
                compare_df.loc[metric_display_name] = pd.to_numeric(compare_df.loc[metric_display_name], errors='coerce')

        styled_df = compare_df.style.apply(highlight_max_min, axis=1).format(formats, na_rep="N/A")
        st.dataframe(styled_df, use_container_width=True)

    st.divider()

    # --- Filter and Prepare Comparison History Data ---
    history_data_filtered = {}
    all_empty = True
    for ticker, hist_max in history_data_max.items():
        filtered = filter_history(hist_max.to_frame('Close'), selected_period) # Convert Series to Frame for filter func
        if not filtered.empty:
             history_data_filtered[ticker] = filtered['Close'] # Store Series again
             all_empty = False
        else:
            history_data_filtered[ticker] = pd.Series(dtype='float64') # Keep ticker key with empty series

    # --- Comparison Charts Container ---
    with st.container(border=True):
         st.subheader(f":chart_with_upwards_trend: Price History Comparison ({selected_period})")
         if not all_empty:
             history_df_filtered = pd.DataFrame(history_data_filtered)
             history_df_filtered = history_df_filtered.dropna(axis=1, how='all') # Drop cols that are purely NaN

             # --- Normalized Performance Chart ---
             st.subheader("Normalized Performance (% Change)")
             # Align dataframes by index (dates) before normalizing
             # Forward fill to handle missing values during normalization period
             aligned_df = history_df_filtered.ffill().bfill()
             # Find first valid index across all columns after alignment
             first_valid_idx = aligned_df.first_valid_index()
             if first_valid_idx is not None:
                 # Get the row corresponding to the first valid index
                 start_values = aligned_df.loc[first_valid_idx]
                 # Avoid division by zero or NaN
                 start_values = start_values.replace(0, pd.NA).ffill().bfill()

                 if not start_values.isna().all():
                     normalized_df = (aligned_df / start_values) * 100
                     st.line_chart(normalized_df)
                 else:
                     st.warning("Could not determine start values for normalization.")
             else:
                 st.warning("No valid data points found for normalization in the selected period.")

             st.divider()

             # --- Individual Price Charts ---
             st.subheader("Individual Price History")
             for ticker in history_df_filtered.columns:
                 # Check if the filtered series for this ticker has data
                 if not history_df_filtered[ticker].dropna().empty:
                     st.line_chart(history_df_filtered[ticker].dropna(), use_container_width=True)
                     st.caption(f"{ticker} Price History") # Use caption for less emphasis
                     st.markdown("----") # Separator between individual charts
                 # else: # Optional: Show message if a specific ticker has no data for the period
                     # st.caption(f"{ticker}: No data for selected period.")

         else:
             st.warning(f"Could not retrieve sufficient price history for comparison charts in the selected '{selected_period}' period.")


# --- Footer Info ---
st.sidebar.markdown("---")
st.sidebar.info("Data sourced from Yahoo Finance via yfinance & Wikipedia.") 