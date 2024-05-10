import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
from datetime import datetime
from constants import (BASE_URL, SWISS_MARKET_ASSETS, STOCK_THRESHOLD, z_score_threshold, ETF_THRESHOLD)
#from keys import API_KEY
from openai import OpenAI
from streamlit_modal import Modal
import base64
import random

# Initialize OpenAI client
client = OpenAI(
    api_key=st.secrets["api_keys"]["OPENAI_API_KEY"],
    organization="org-tlUjsA0VUZEicWZ3HTdlVHJH",
    project="proj_TVJJRleVFZjzvzzMol4NWKPq"
)

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    try:
        api_key = st.secrets["api_keys"]["API_KEY"]
        response = requests.get(f"{BASE_URL}quote/{symbol}?apikey={api_key}")
        response.raise_for_status()
        data = response.json()
        if not data:
            st.warning(f"No data available for {symbol}.")
            return None
        return {'price': data[0]['price'], 'previousClose': data[0]['previousClose'], 'fullName': data[0]['name']}
    except requests.RequestException as e:
        st.error(f"Failed to retrieve data: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_historical_data(symbol):
    try:
        api_key = st.secrets["api_keys"]["API_KEY"]
        url = f"{BASE_URL}historical-price-full/{symbol}?timeseries=180&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['historical']
        sorted_data = sorted(data, key=lambda x: x['date'])
        prices = [day['close'] for day in sorted_data if 'close' in day]
        dates = [day['date'] for day in sorted_data if 'date' in day]
        return prices, dates
    except requests.RequestException as e:
        st.error(f"Failed to retrieve historical data for {symbol}: {e}")
        return [], []

def fetch_news(symbol):
    api_key = st.secrets["api_keys"]["API_KEY"]
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit=5&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch news")
        return []

def get_chatgpt_summary(news_articles, today_change, full_name, symbol):
    news_text = ' '.join([article['text'] for article in news_articles])
    query = f"Summarize the following news related to {full_name} ({symbol}) which experienced a {today_change:.2f}% change yesterday: {news_text}"
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=250
        )
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()
        else:
            return "No summary available."
    except Exception as e:
        st.error(f"Failed to retrieve summary due to an API error: {str(e)}")
        return None

def calculate_std_dev(prices):
    if not prices:
        return 0, 0
    return np.mean(prices), np.std(prices)

def calculate_modified_z_scores(prices):
    median = np.median(prices)
    deviations = [abs(x - median) for x in prices]
    mad = np.median(deviations)
    if mad == 0:
        return [0] * len(prices)
    return [0.6745 * (x - median) / mad for x in prices]

def outlier_detection(today_price, historical_prices, z_score_threshold):
    all_prices = historical_prices + [today_price['price']]
    historical_z_scores = calculate_modified_z_scores(historical_prices)
    all_z_scores = calculate_modified_z_scores(all_prices)
    today_z_score = all_z_scores[-1]
    is_outlier = abs(today_z_score) > z_score_threshold
    return is_outlier, today_z_score, historical_z_scores

def show_info_modal(graph_title, graph_description):
    """Displays a modal with information about a specific graph."""
    modal_key = f"modal_{graph_title.replace(' ', '_')}"
    modal = Modal(key=modal_key, title=graph_title)
    if modal.is_open():
        with modal.container():
            st.markdown(graph_description)
            st.button("Close", on_click=modal.close)
    return modal


def display_results(today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, price_change_threshold, z_score_threshold, today_price_data, selected_asset):
    is_outlier, today_z_score, historical_z_scores = outlier_detection(today_price_data, historical_prices, z_score_threshold)

    # Display the summary table above graphs
    results_df = pd.DataFrame({
        "Symbol": [selected_asset['symbol']],
        "Full Name": [today_price_data['fullName']],
        "Today's Price": [today_price],
        "Yesterday's Price": [today_price_data['previousClose']],
        "Percentage Change": [f"{today_change:.2f}%"],
        "Change Above Threshold": ["Yes" if abs(today_change) > price_change_threshold else "No"],
        "Today's Z-Score": [f"{today_z_score:.2f}"],
        "Outlier Status": ["Yes" if is_outlier else "No"]
    })

    st.write(results_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_historical_prices(historical_dates, historical_prices)
        # Add the threshold check summary under the graph
        if abs(today_change) > price_change_threshold:
            st.error(f"The price change today is {today_change:.2f}%, which is above the {price_change_threshold}% threshold.")
        else:
            st.success(f"The price change today is {today_change:.2f}%, which is below the {price_change_threshold}% threshold.")

    with col2:
        plot_price_distribution(historical_mean, historical_std_dev, today_price)
        # Add the Z-score threshold check summary under the graph
        if is_outlier:
            st.error(f"Today's price change is an outlier with a Z-score of {today_z_score:.2f}.")
        else:
            st.success("Today's price change is not considered an outlier based on Z-scores.")

    with col3:
        display_simulated_exchanges(today_price)

    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{selected_asset["symbol"]}_price_results.csv">Download Results as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)



def display_simulated_exchanges(api_price):
    """Simulates stock exchange prices and generates a bar chart."""
    exchanges = [
        "Deutsche Boerse AG",
        "Boerse Stuttgart",
        "Boerse München",
        "Boerse Hannover",
        "Boerse Düsseldorf",
        "Boerse Berlin",
        "SIX (API Price)"
    ]

    # Initialize simulated prices with API price as base
    simulated_prices = []

    for exchange in exchanges:
        if exchange == "SIX (API Price)":
            simulated_prices.append(api_price)
        else:
            rand = random.random()
            if rand < 0.90:
                simulated_price = api_price
            elif rand < 0.95:
                simulated_price = api_price * random.uniform(0.9, 1.1)  # Slight deviation
            else:
                simulated_price = api_price * random.uniform(0.7, 1.3)  # Strong deviation
            simulated_prices.append(simulated_price)

    simulated_prices = np.round(simulated_prices, 2)
    mean_price = np.mean(simulated_prices)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust width to make the chart less wide

    y_pos = np.arange(len(exchanges))
    colors = ['#888888'] * len(exchanges)
    colors[-1] = '#ff0000'  # Highlight "SIX (API Price)" in red

    ax.barh(y_pos, simulated_prices, color=colors)
    ax.axvline(mean_price, color='orange', linestyle='-', linewidth=2)
    ax.text(mean_price + 0.5, len(exchanges) - 1, 'Mittelwert', color='orange')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(exchanges)
    ax.invert_yaxis()  # Invert y-axis to have the first exchange at the top
    ax.set_xlabel("Preis (in CHF)")
    ax.set_title("Price Development", fontsize=15, fontweight='bold')

    st.pyplot(fig)


def process_asset_data(selected_asset, threshold):
    historical_prices, historical_dates = fetch_historical_data(selected_asset['symbol'])
    today_price_data = fetch_stock_data(selected_asset['symbol'])
    if not today_price_data or not historical_prices:
        st.error("Failed to fetch today's price or historical data for the selected asset.")
        return None
    today_price = today_price_data['price']
    previous_close = today_price_data['previousClose']
    today_change = ((today_price - previous_close) / previous_close) * 100
    historical_mean, historical_std_dev = calculate_std_dev(historical_prices)
    return today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold

def plot_historical_prices(historical_dates, historical_prices):
    df = pd.DataFrame({'Date': pd.to_datetime(historical_dates), 'Price': historical_prices})
    fig = px.line(df, x='Date', y='Price', title='Historical Price Development (Last 180 Days)',
                  labels={'Price': 'Closing Price ($)'})
    fig.update_xaxes(rangeslider_visible=False)  # Remove the range slider
    fig.update_layout(title_font=dict(size=15, family='Arial', color='black', weight='bold'))  # Ensure consistent title style
    st.plotly_chart(fig, use_container_width=True)



def plot_price_distribution(historical_mean, historical_std_dev, today_price):
    x_range = np.linspace(historical_mean - 3 * historical_std_dev, historical_mean + 3 * historical_std_dev, 400)
    y_range = stats.norm.pdf(x_range, historical_mean, historical_std_dev)
    fig, ax = plt.subplots()
    ax.plot(x_range, y_range, label='Historical Price Distribution', color='blue')
    ax.fill_between(x_range, 0, y_range, where=(x_range <= today_price), color='green', alpha=0.5)
    today_price_point = stats.norm.pdf(today_price, historical_mean, historical_std_dev)
    ax.plot([today_price, today_price], [0, today_price_point], color='red', linestyle='dashed', label=f"Today's Price = {today_price}")
    average_price_point = stats.norm.pdf(historical_mean, historical_mean, historical_std_dev)
    ax.plot([historical_mean, historical_mean], [0, average_price_point], color='yellow', linestyle='dashed', label=f"180-Day Average Price = {historical_mean}")
    ax.legend(loc='upper right')
    ax.set_title('Normal Distribution of Historical Prices', fontsize=15, fontweight='bold')
    ax.set_xlabel('Price')
    ax.set_ylabel('Probability Density')
    st.pyplot(fig)



def filter_stocks_above_threshold(stocks, threshold=STOCK_THRESHOLD):
    results = []
    for stock in stocks:
        stock_data = fetch_stock_data(stock['symbol'])
        if stock_data:
            price_change = ((stock_data['price'] - stock_data['previousClose']) / stock_data['previousClose']) * 100
            z_score = calculate_modified_z_scores([stock_data['previousClose'], stock_data['price']])[-1]
            if abs(price_change) > threshold:
                results.append({
                    'symbol': stock['symbol'],
                    'fullName': stock_data['fullName'],
                    'change': f"{price_change:.2f}%",
                    'z_score': f"{z_score:.2f}"
                })
    return pd.DataFrame(results)


def analysis_page():
    st.set_page_config(
        page_title="Stock Analysis App",
        layout="wide"
    )

    threshold = STOCK_THRESHOLD

    
    filtered_stocks_df = filter_stocks_above_threshold(SWISS_MARKET_ASSETS, threshold)

    # Ensure the DataFrame is not empty
    if not filtered_stocks_df.empty:
        # Sidebar: Display Filtered Stocks with Ag-Grid inside the sidebar
        with st.sidebar:
            st.title("Stocks Above Threshold")
            columns_to_display = ['fullName', 'change', 'z_score']
            gb = GridOptionsBuilder.from_dataframe(filtered_stocks_df[columns_to_display])
            gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
            grid_options = gb.build()
            selection = AgGrid(
                filtered_stocks_df[columns_to_display],
                gridOptions=grid_options,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                height=600,
                width='100%',
                key='table',
                theme='material'
            )

        # Main Page: Analyze Selected Stock
        selected_stock = pd.DataFrame(selection['selected_rows'])
        if not selected_stock.empty:
            full_name_selected = selected_stock.iloc[0]['fullName']
            selected_asset = filtered_stocks_df[filtered_stocks_df['fullName'] == full_name_selected].iloc[0]
            st.title(f"Analysis of {selected_asset['symbol']}")
            result = process_asset_data(selected_asset, threshold)
            if result:
                today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, _ = result
                today_price_data = fetch_stock_data(selected_asset['symbol'])
                if today_price_data:
                    display_results(today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold, z_score_threshold, today_price_data, selected_asset)
                else:
                    st.error("Failed to retrieve today's price data.")
        else:
            st.write("Select a stock from the list to view its analysis.")
    else:
        st.sidebar.write("No stocks above the threshold were found.")

