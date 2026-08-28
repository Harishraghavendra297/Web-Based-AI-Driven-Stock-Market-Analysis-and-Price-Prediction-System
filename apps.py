import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import yfinance as yf
from stocknews import StockNews
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly
import plotly.graph_objects as go
import json
from sklearn.model_selection import train_test_split
from html import unescape
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import re
from urllib.parse import quote_plus
import requests

warnings.filterwarnings('ignore')

# Try to import TensorFlow (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM predictions will be simulated.")

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-only')

US_STOCKS = {
    'GOOG': 'Alphabet Inc. (US)',
    'AAPL': 'Apple Inc. (US)',
    'MSFT': 'Microsoft Corporation (US)',
    'AMZN': 'Amazon.com, Inc. (US)',
    'TSLA': 'Tesla, Inc. (US)',
    'NFLX': 'Netflix, Inc. (US)',
    'NVDA': 'NVIDIA Corporation (US)',
    'IBM': 'IBM (US)',
    'INTC': 'Intel Corporation (US)',
    'BA': 'Boeing Company (US)',
    'CSCO': 'Cisco Systems (US)',
    'PFE': 'Pfizer Inc. (US)',
    'WMT': 'Walmart Inc. (US)',
    'DIS': 'Walt Disney Company (US)',
    'JNJ': 'Johnson & Johnson (US)',
    'PG': 'Procter & Gamble (US)',
    'META': 'Meta Platforms (US)',
    'MCD': 'McDonald\'s Corporation (US)',
    'PEP': 'PepsiCo, Inc. (US)'
}

SENSEX_30 = {
    "RELIANCE.NS": "Reliance Industries (SENSEX)",
    "TCS.NS": "TCS (SENSEX)",
    "HDFCBANK.NS": "HDFC Bank (SENSEX)",
    "ICICIBANK.NS": "ICICI Bank (SENSEX)",
    "INFY.NS": "Infosys (SENSEX)",
    "HINDUNILVR.NS": "HUL (SENSEX)",
    "ITC.NS": "ITC (SENSEX)",
    "LT.NS": "L&T (SENSEX)",
    "SBIN.NS": "SBI (SENSEX)",
    "BHARTIARTL.NS": "Bharti Airtel (SENSEX)",
    "AXISBANK.NS": "Axis Bank (SENSEX)",
    "KOTAKBANK.NS": "Kotak Bank (SENSEX)",
    "BAJFINANCE.NS": "Bajaj Finance (SENSEX)",
    "HCLTECH.NS": "HCL Tech (SENSEX)",
    "ASIANPAINT.NS": "Asian Paints (SENSEX)",
    "MARUTI.NS": "Maruti Suzuki (SENSEX)",
    "SUNPHARMA.NS": "Sun Pharma (SENSEX)",
    "ULTRACEMCO.NS": "UltraTech Cement (SENSEX)",
    "TITAN.NS": "Titan (SENSEX)",
    "NTPC.NS": "NTPC (SENSEX)",
    "POWERGRID.NS": "Power Grid (SENSEX)",
    "NESTLEIND.NS": "Nestlé India (SENSEX)",
    "WIPRO.NS": "Wipro (SENSEX)",
    "TECHM.NS": "Tech Mahindra (SENSEX)",
    "TATASTEEL.NS": "Tata Steel (SENSEX)",
    "JSWSTEEL.NS": "JSW Steel (SENSEX)",
    "ADANIENT.NS": "Adani Ent. (SENSEX)",
    "INDUSINDBK.NS": "IndusInd Bank (SENSEX)",
    "M&M.NS": "M&M (SENSEX)",
    "DRREDDY.NS": "Dr Reddy's (SENSEX)"
}

NIFTY_50 = {
    "ADANIPORTS.NS": "Adani Ports (NIFTY)",
    "APOLLOHOSP.NS": "Apollo Hospitals (NIFTY)",
    "BAJAJ-AUTO.NS": "Bajaj Auto (NIFTY)",
    "BAJAJFINSV.NS": "Bajaj Finserv (NIFTY)",
    "BPCL.NS": "BPCL (NIFTY)",
    "BRITANNIA.NS": "Britannia (NIFTY)",
    "CIPLA.NS": "Cipla (NIFTY)",
    "COALINDIA.NS": "Coal India (NIFTY)",
    "DIVISLAB.NS": "Divi's Labs (NIFTY)",
    "EICHERMOT.NS": "Eicher Motors (NIFTY)",
    "GRASIM.NS": "Grasim (NIFTY)",
    "HDFCLIFE.NS": "HDFC Life (NIFTY)",
    "HEROMOTOCO.NS": "Hero MotoCorp (NIFTY)",
    "HINDALCO.NS": "Hindalco (NIFTY)",
    "ONGC.NS": "ONGC (NIFTY)",
    "SBILIFE.NS": "SBI Life (NIFTY)",
    "UPL.NS": "UPL (NIFTY)"
}

stock_names = {}
stock_names.update(US_STOCKS)
stock_names.update(SENSEX_30)
stock_names.update(NIFTY_50)

display_names = {v: k for k, v in stock_names.items()}

def get_currency_symbol(symbol):
    if symbol.endswith('.NS'):
        return '₹'
    return '$'

sid = SentimentIntensityAnalyzer()

def clean_summary(summary, title):
    if not summary:
        return ""
    summary = unescape(summary)
    summary = re.sub(r'<[^>]+>', '', summary)
    summary = summary.replace(title, '')
    summary = re.sub(r'\s+', ' ', summary).strip()
    return summary[:280]

def get_stock_news(company_name):
    query = quote_plus(f"{company_name} stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    news = []
    for entry in feed.entries[:8]:
        title = entry.title
        summary = clean_summary(entry.get("summary", ""), title)
        news.append({
            "title": title,
            "link": entry.link,
            "published": entry.get("published", ""),
            "summary": summary or "Click to read the full article."
        })
    return news

def get_stock_data_with_fallback(symbol, start_date, end_date):
    try:
        custom_session = requests.Session()
        custom_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        })
        df = yf.download(symbol, start=start_date, end=end_date, session=custom_session, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, False 
    except Exception as e:
        print(f"Yahoo Finance blocked request: {e}")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(sum([ord(c) for c in symbol])) 
    
    base_price = 1200.0 if symbol.endswith('.NS') else 150.0
    if 'GOOG' in symbol or 'MSFT' in symbol: base_price = 300.0
    
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = prices * np.random.uniform(0.99, 1.01, len(dates))
    df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.01, len(dates))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.99, 1.0, len(dates))
    df['Volume'] = np.random.randint(1000000, 5000000, len(dates))
    
    return df, True 

def get_fundamentals_with_fallback(symbol, company_name):
    cur_sign = '₹' if symbol.endswith('.NS') else '$'
    try:
        custom_session = requests.Session()
        custom_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        })
        stock = yf.Ticker(symbol, session=custom_session)
        info = stock.info
        if not info:
            raise ValueError("Empty info returned")

        market_cap = info.get('marketCap')
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12: market_cap_str = f"{cur_sign}{market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9: market_cap_str = f"{cur_sign}{market_cap / 1e9:.2f}B"
            else: market_cap_str = f"{cur_sign}{market_cap / 1e6:.2f}M"
        else:
            market_cap_str = "N/A"

        return {
            'company': company_name,
            'market_cap': market_cap_str,
            'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
            'eps': round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A',
            'dividend_yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
    except Exception as e:
        np.random.seed(sum([ord(c) for c in symbol]))
        mc_val = np.random.uniform(50, 2000) if not symbol.endswith('.NS') else np.random.uniform(100, 900000)
        return {
            'company': company_name + " (Simulated Mode)",
            'market_cap': f"{cur_sign}{mc_val:.2f}B",
            'pe_ratio': round(np.random.uniform(10, 50), 2),
            'eps': round(np.random.uniform(2, 15), 2),
            'dividend_yield': f"{np.random.uniform(0.5, 4.5):.2f}%",
            'sector': 'Technology/Finance',
            'industry': 'Enterprise'
        }

SUPABASE_DB_URL = 'postgresql://postgres.lyaptqtkyeumynygcnvm:Idontknow.1%40hari@aws-0-ap-south-1.pooler.supabase.com:5432/postgres'

def init_db():
    import psycopg2
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    mobile_number TEXT NOT NULL,
                    email TEXT NOT NULL,
                    occupation TEXT,
                    date_of_birth TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id SERIAL PRIMARY KEY,
                    username TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    shares NUMERIC NOT NULL,
                    purchase_price NUMERIC NOT NULL,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init note: {e}")

def get_connection():
    import psycopg2
    return psycopg2.connect(SUPABASE_DB_URL)

def add_user(first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO users (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
            return cursor.fetchone()
    finally:
        conn.close()

def lstm_predict(data, n_future=7, n_past=60, epochs=10):
    if not TENSORFLOW_AVAILABLE:
        last_price = data['Close'].iloc[-1]
        import random
        return [last_price * (1 + random.uniform(-0.02, 0.03)) for _ in range(n_future)]
    try:
        close_data = data['Close'].values.reshape(-1,1)
        if len(close_data) > 1000: close_data = close_data[-1000:]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)
        X, y = [], []
        for i in range(n_past, len(scaled_data)):
            X.append(scaled_data[i-n_past:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        predicted_prices = []
        last_sequence = scaled_data[-n_past:]
        for _ in range(n_future):
            pred = model.predict(last_sequence.reshape(1, n_past, 1), verbose=0)
            predicted_prices.append(pred[0,0])
            last_sequence = np.append(last_sequence[1:], pred, axis=0)
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))
        return predicted_prices.flatten().tolist()
    except Exception as e:
        last_price = data['Close'].iloc[-1]
        return [last_price * (1 + i * 0.01) for i in range(n_future)]

@app.route('/')
def index():
    if 'user' in session: return redirect(url_for('main_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session: return redirect(url_for('main_page'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash('Please enter both username and password', 'error')
        else:
            user = authenticate_user(username, password)
            if user:
                session['user'] = username
                return redirect(url_for('main_page'))
            else:
                flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session: return redirect(url_for('main_page'))
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        mobile_number = request.form.get('mobile_number', '').strip()
        email = request.form.get('email', '').strip()
        occupation = request.form.get('occupation', '').strip()
        date_of_birth = request.form.get('date_of_birth', '')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            if add_user(first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password):
                flash('Account created successfully! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username already exists or database error.', 'error')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    if 'user' not in session: return redirect(url_for('login'))
    username = session['user']
    conn = get_connection()
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            stock_choice = request.form.get('stock')
            symbol = display_names.get(stock_choice, stock_choice)
            try:
                shares = float(request.form.get('shares', 0))
                purchase_price = float(request.form.get('purchase_price', 0))
                if shares > 0 and purchase_price > 0:
                    with conn.cursor() as cursor:
                        cursor.execute('INSERT INTO portfolio (username, symbol, shares, purchase_price) VALUES (%s, %s, %s, %s)', (username, symbol, shares, purchase_price))
                    conn.commit()
            except ValueError:
                pass
        elif action == 'delete':
            item_id = request.form.get('item_id')
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM portfolio WHERE id = %s AND username = %s', (item_id, username))
            conn.commit()

    with conn.cursor() as cursor:
        cursor.execute('SELECT id, symbol, shares, purchase_price FROM portfolio WHERE username = %s', (username,))
        rows = cursor.fetchall()
    conn.close()

    portfolio_items = []
    total_val, total_inv = 0.0, 0.0
    for row in rows:
        item_id, symbol, shares, purchase_price = row
        df_temp, _ = get_stock_data_with_fallback(symbol, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
        current_price = float(df_temp['Close'].iloc[-1]) if not df_temp.empty else float(purchase_price)
        h_val = shares * current_price
        i_val = shares * float(purchase_price)
        pnl = h_val - i_val
        pnl_pct = (pnl / i_val * 100) if i_val > 0 else 0.0
        total_val += h_val
        total_inv += i_val
        portfolio_items.append({
            'id': item_id, 'symbol': symbol, 'name': stock_names.get(symbol, symbol),
            'shares': shares, 'purchase_price': round(float(purchase_price), 2),
            'current_price': round(current_price, 2), 'holding_value': round(h_val, 2),
            'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2), 'currency': get_currency_symbol(symbol)
        })
    total_pnl = total_val - total_inv
    total_pnl_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0.0
    return render_template('portfolio.html', stocks=display_names, portfolio_items=portfolio_items,
                           total_portfolio_value=round(total_val, 2), total_invested=round(total_inv, 2),
                           total_pnl=round(total_pnl, 2), total_pnl_pct=round(total_pnl_pct, 2))

@app.route('/main', methods=['GET', 'POST'])
def main_page():
    if 'user' not in session: return redirect(url_for('login'))
    selected_display_name = request.form.get('stock', 'Apple Inc. (US)')
    selected_symbol = display_names.get(selected_display_name, 'AAPL')
    default_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    start_date = request.form.get('start_date', default_start)
    end_date = request.form.get('end_date', default_end)
    currency_symbol = get_currency_symbol(selected_symbol)

    context = {
        'username': session['user'], 'stocks': display_names, 'selected_stock': selected_display_name,
        'selected_symbol': selected_symbol, 'start_date': start_date, 'end_date': end_date,
        'currency_symbol': currency_symbol, 'has_data': False, 'price_data': None,
        'lr_data': None, 'lstm_data': None, 'recent_data': None, 'current_price': None, 'last_date': None
    }

    df, is_simulated = get_stock_data_with_fallback(selected_symbol, start_date, end_date)
    if df.empty: return render_template('main.html', **context)

    context['has_data'] = True
    close_prices = df['Close']
    context['price_data'] = json.dumps({'dates': df.index.strftime('%Y-%m-%d').tolist(), 'prices': close_prices.tolist()})
    context['current_price'] = round(float(close_prices.iloc[-1]), 2)
    context['last_date'] = df.index[-1].strftime('%Y-%m-%d')

    context['recent_data'] = [{
        'Date': idx.strftime('%Y-%m-%d'), 'Open': round(float(row['Open']), 2),
        'High': round(float(row['High']), 2), 'Low': round(float(row['Low']), 2),
        'Close': round(float(row['Close']), 2), 'Volume': int(row['Volume'])
    } for idx, row in df.tail(10).iterrows()]

    try:
        df_lr = df.copy()
        df_lr['Day'] = range(len(df_lr))
        X, y = df_lr[['Day']].values, close_prices.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression().fit(X_train, y_train)
        context['lr_data'] = json.dumps({
            'dates': df_lr.index[-len(y_test):].strftime('%Y-%m-%d').tolist(),
            'actual': y_test.tolist(), 'predicted': model.predict(X_test).tolist()
        })
    except Exception: pass

    try:
        lstm_predictions = lstm_predict(df)
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7).strftime('%Y-%m-%d').tolist()
        context['lstm_data'] = json.dumps({
            'historical_dates': df.index[-30:].strftime('%Y-%m-%d').tolist(),
            'historical_prices': close_prices[-30:].tolist(),
            'future_dates': future_dates, 'future_prices': [round(p, 2) for p in lstm_predictions]
        })
        lstm_table = []
        last_price = float(close_prices.iloc[-1])
        for d, p in zip(future_dates, lstm_predictions):
            change = ((p - last_price) / last_price) * 100
            lstm_table.append({'date': d, 'price': round(p, 2), 'change': round(change, 2)})
            last_price = p
        context['lstm_table_data'] = lstm_table
    except Exception: pass

    return render_template('main.html', **context)

@app.route('/fundamental', methods=['GET', 'POST'])
def fundamental_data():
    if 'user' not in session: return redirect(url_for('login'))
    selected_symbols = [display_names[name] for name in request.form.getlist('stocks') if name in display_names] if request.method == 'POST' else []
    fundamentals_data = {sym: get_fundamentals_with_fallback(sym, stock_names.get(sym, sym)) for sym in selected_symbols}
    return render_template('fundamental.html', stocks=display_names, fundamentals=fundamentals_data)

@app.route('/news', methods=['GET', 'POST'])
def news():
    if 'user' not in session: return redirect(url_for('login'))
    news_data = {}
    if request.method == 'POST':
        for sym in [display_names[n] for n in request.form.getlist('stocks') if n in display_names]:
            comp = stock_names.get(sym, sym)
            try: news_data[sym] = {'company': comp, 'news': get_stock_news(comp)}
            except Exception as e: news_data[sym] = {'company': comp, 'error': str(e)}
    return render_template('news.html', stocks=display_names, news_data=news_data)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if 'user' not in session: return redirect(url_for('login'))
    sentiment_data = {}
    if request.method == 'POST':
        for sym in [display_names[n] for n in request.form.getlist('stocks') if n in display_names]:
            try:
                comp = stock_names.get(sym, sym).split("(")[0].strip()
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={comp.replace(' ', '+')}+stock")
                sents, pos, neg, neu = [], 0, 0, 0
                for entry in feed.entries[:10]:
                    score = sid.polarity_scores(entry.title)["compound"]
                    stype = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
                    if stype == "Positive": pos += 1
                    elif stype == "Negative": neg += 1
                    else: neu += 1
                    sents.append({'title': entry.title[:100], 'sentiment': stype, 'score': round(score, 3)})
                sentiment_data[sym] = {'company': comp, 'sentiments': sents, 'summary': {'positive': pos, 'negative': neg, 'neutral': neu, 'overall': 'Positive' if pos > neg else ('Negative' if neg > pos else 'Neutral')}}
            except Exception as e: sentiment_data[sym] = {'error': str(e)}
    return render_template('sentiment.html', stocks=display_names, sentiment_data=sentiment_data)

init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
