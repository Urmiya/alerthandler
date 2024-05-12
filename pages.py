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
from streamlit.components.v1 import html

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
        url = f"{BASE_URL}historical-price-full/{symbol}?timeseries=90&apikey={api_key}"  # Changed from 180 to 90
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
        return response.json()  # Returns a list of news articles
    else:
        st.error("Failed to fetch news")
        return []


def get_chatgpt_summary(news_articles, symbol):
    if not news_articles:
        return "No recent news to analyze."

    # Create a summary of key points from the articles
    news_summary = ' '.join([article['title'] + ": " + article['text'] for article in news_articles])
    
    # Query asks for specific analysis based on summarized news
    query = f"Based on the following summarized news points, analyze the potential impact on the price of {symbol} and the likelihood of significant price deviations today: {news_summary}"

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=250  # Increased token count to allow a more detailed analysis
        )
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()
        else:
            return "Analysis not available or inconclusive."
    except Exception as e:
        st.error(f"Failed to retrieve analysis due to an API error: {str(e)}")
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

def show_info_modal(title, description, unique_identifier):
    modal_key = f"modal_{unique_identifier}"  # Generating a unique key based on some unique identifier
    if st.button("ℹ️", key=f"info_button_{modal_key}"):
        with Modal(title=title, key=modal_key).container():
            st.write(description)
            if st.button("Close", key=f"close_{modal_key}"):
                pass  # This just closes the modal without any session state handling

def display_news_summary(symbol):
    with st.expander("News Summary"):
        news_articles = fetch_news(symbol)
        if news_articles:
            summary = get_chatgpt_summary(news_articles, symbol)
            st.write(summary)
        else:
            st.write("No recent news articles found.")

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
        "Outlier Status": ["Yes" if is_outlier else "No"],
        "Type": [selected_asset.get('type', 'Unknown')]  # Use .get() with a default value
    })

    col_df, col_buttons = st.columns([8, 2])
    with col_df:
        st.write(results_df)
    with col_buttons:
        col_check, col_mail = st.columns([1, 1])
        with col_check:
            if st.button("✔️", key="check_button"):
                # Mark the stock as removed in session state
                if 'removed_stocks' not in st.session_state:
                    st.session_state['removed_stocks'] = {}
                st.session_state['removed_stocks'][selected_asset['symbol']] = True
                st.experimental_rerun()  # Rerun the app to refresh the changes
        with col_mail:
            if st.button("✉️", key="mail_button"):
                st.write("Mail button clicked!")

    # Manually define graph heights
    graph_height_1 = 300
    graph_height_2 = 400
    graph_height_3 = 600

    # Container for all three graphs
    col_graphs = st.columns(3)
    for i, (graph, title, modal_content, is_outlier_check, change_check) in enumerate([
        (lambda: plot_historical_prices(historical_dates, historical_prices, graph_height_1), 
         "Historical Price Development", 
         "This graph shows the price development of the selected asset over the last 90 days, plotting daily closing prices.",
         None, 
         abs(today_change) > price_change_threshold),
        (lambda: plot_price_distribution(historical_mean, historical_std_dev, today_price, graph_height_2), 
         "Normal Distribution of Historical Prices", 
         "This graph depicts the normal distribution of historical prices, highlighting how today's price compares to the distribution, indicating whether it is considered an outlier.",
         is_outlier, 
         None),
        (lambda: display_simulated_exchanges(today_price, graph_height_3), 
         "Simulated Exchanges", 
         "This graph simulates different exchange prices for the selected asset, showing variations and how the official price compares to these simulations.",
         None, 
         None)
    ]):
        with col_graphs[i]:
            st.markdown(f"### {title}")
            graph()
            subcol_message = st.columns([1])
            with subcol_message[0]:
                if change_check:
                    st.error(f"The price change today is {today_change:.2f}%, which is above the {price_change_threshold}% threshold.")
                if is_outlier_check is not None:
                    if is_outlier_check:
                        st.error(f"Today's price change is an outlier with a Z-score of {today_z_score:.2f}.")
                    else:
                        st.success("Today's price change is not considered an outlier based on Z-scores.")
            
            # Use expanders for additional information instead of modals
            with st.expander("More Info"):
                st.write(modal_content)
    
    display_news_summary(selected_asset['symbol'])

def display_simulated_exchanges(api_price, graph_height=500):
    exchanges = [
        "Deutsche Boerse AG",
        "Boerse Stuttgart",
        "Boerse München",
        "Boerse Hannover",
        "Boerse Düsseldorf",
        "Boerse Berlin",
        "Eval. Price"  # Renamed from "SIX (API Price)"
    ]

    all_deviate = random.random() < 0.05

    simulated_prices = []
    if all_deviate:
        common_price = api_price * random.uniform(0.7, 1.3)
        for exchange in exchanges[:-1]:
            simulated_prices.append(common_price)
        simulated_prices.append(api_price)
    else:
        for exchange in exchanges:
            if exchange == "Eval. Price":
                simulated_prices.append(api_price)
            else:
                rand = random.random()
                if rand < 0.90:
                    simulated_price = api_price
                elif rand < 0.95:
                    simulated_price = api_price * random.uniform(0.9, 1.1)
                else:
                    simulated_price = api_price * random.uniform(0.7, 1.3)
                simulated_prices.append(simulated_price)

    simulated_prices = np.round(simulated_prices, 2)
    mean_price = np.mean(simulated_prices)

    fig, ax = plt.subplots(figsize=(6, graph_height / 100))

    y_pos = np.arange(len(exchanges))
    colors = ['#888888'] * len(exchanges)
    colors[-1] = '#ff0000'  # Keep the Eval. Price bar red
    # Select one random bar (other than Eval. Price) to be blue
    random_bar_index = random.choice([i for i in range(len(colors)-1)])
    colors[random_bar_index] = '#0000ff'

    ax.barh(y_pos, simulated_prices, color=colors)
    ax.axvline(mean_price, color='orange', linestyle='-', linewidth=2)
    ax.text(mean_price + 0.5, len(exchanges) - 1, 'Mittelwert', color='orange')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(exchanges)
    ax.invert_yaxis()
    ax.set_xlabel("Price (in CHF)")

    st.pyplot(fig)

    identical_prices = sum(1 for price in simulated_prices[:-1] if price == api_price)
    different_prices = len(exchanges) - 1 - identical_prices

    st.success(f"{identical_prices} exchanges delivered the same price as Eval. Price.")



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

def plot_historical_prices(historical_dates, historical_prices, graph_height=400):
    df = pd.DataFrame({'Date': pd.to_datetime(historical_dates), 'Price': historical_prices})
    fig = px.line(
        df,
        x='Date',
        y='Price',
        labels={'Price': 'Closing Price ($)'}
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        height=graph_height,
        margin=dict(l=20, r=20, t=20, b=30)
    )

    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)




def plot_price_distribution(historical_mean, historical_std_dev, today_price, graph_height=450):
    x_range = np.linspace(historical_mean - 3 * historical_std_dev, historical_mean + 3 * historical_std_dev, 400)
    y_range = stats.norm.pdf(x_range, historical_mean, historical_std_dev)
    fig, ax = plt.subplots(figsize=(6, graph_height / 100))  # Set consistent figure size

    ax.plot(x_range, y_range, label='Historical Price Distribution', color='blue')
    ax.fill_between(x_range, 0, y_range, where=(x_range <= today_price), color='green', alpha=0.5)
    today_price_point = stats.norm.pdf(today_price, historical_mean, historical_std_dev)
    ax.plot([today_price, today_price], [0, today_price_point], color='red', linestyle='dashed', label=f"Today's Price = {today_price}")
    average_price_point = stats.norm.pdf(historical_mean, historical_mean, historical_std_dev)
    ax.plot([historical_mean, historical_mean], [0, average_price_point], color='yellow', linestyle='dashed', label=f"90-Day Average Price = {historical_mean: .2f}")

    ax.legend(loc='upper right')
    ax.set_xlabel('Price')
    ax.set_ylabel('Probability Density')

    st.pyplot(fig)






def filter_stocks_above_threshold(stocks, threshold=STOCK_THRESHOLD):
    results = []
    for stock in stocks:
        # Check if the stock has been marked as removed in the session state
        if not st.session_state.get('removed_stocks', {}).get(stock['symbol'], False):
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
    st.set_page_config(page_title="Stock Analysis App", layout="wide")
    
    threshold = STOCK_THRESHOLD
    filtered_stocks_df = filter_stocks_above_threshold(SWISS_MARKET_ASSETS, threshold)

    if not filtered_stocks_df.empty:
        with st.sidebar:
            st.title("Stocks Above Threshold")
            columns_to_display = ['fullName', 'change']
            gb = GridOptionsBuilder.from_dataframe(filtered_stocks_df[columns_to_display])
            gb.configure_column('fullName', width=220)
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

        # Ensure selected_rows is safely handled
        selected_rows = selection.get('selected_rows') if selection else pd.DataFrame()
        if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            selected_stock = selected_rows
            selected_asset_name = selected_stock.iloc[0]['fullName']
            matching_assets = filtered_stocks_df[filtered_stocks_df['fullName'] == selected_asset_name]
            if not matching_assets.empty:
                selected_asset = matching_assets.iloc[0]
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
                    st.error("Failed to process the asset data.")
            else:
                st.write("Selected stock was removed or filtered out. Please select another.")
        else:
            st.write("No stock selected or empty selection.")
    else:
        st.sidebar.write("No stocks above the threshold were found.")












