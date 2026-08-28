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

# Stock data dictionaries
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

# Combine all stocks
stock_names = {}
stock_names.update(US_STOCKS)
stock_names.update(SENSEX_30)
stock_names.update(NIFTY_50)

# Reverse mapping for dropdown
display_names = {v: k for k, v in stock_names.items()}

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

def clean_summary(summary, title):
    if not summary:
        return ""

    summary = unescape(summary)                     # remove &nbsp; etc
    summary = re.sub(r'<[^>]+>', '', summary)           # strip HTML tags
    summary = summary.replace(title, '')                # remove repeated title
    summary = re.sub(r'\s+', ' ', summary).strip()      # normalize spaces

    return summary[:280]                                # ~2–3 lines

def get_stock_news(company_name):
    query = quote_plus(f"{company_name} stock")
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )

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


# ==================== DATA FALLBACK GENERATORS ====================
def get_stock_data_with_fallback(symbol, start_date, end_date):
    """Attempts to fetch real data, falls back to simulated data if Yahoo throws 429"""
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
    
    # --- FALLBACK: Generate realistic simulated data ---
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(sum([ord(c) for c in symbol])) 
    
    base_price = 150.0
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
    """Attempts to fetch real info, falls back to simulated stats if Yahoo throws 429"""
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
            if market_cap >= 1e12: market_cap_str = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9: market_cap_str = f"${market_cap / 1e9:.2f}B"
            else: market_cap_str = f"${market_cap / 1e6:.2f}M"
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
        # --- FALLBACK: Generate realistic simulated stats ---
        np.random.seed(sum([ord(c) for c in symbol]))
        return {
            'company': company_name + " (Simulated Mode)",
            'market_cap': f"${np.random.uniform(50, 2000):.2f}B",
            'pe_ratio': round(np.random.uniform(10, 50), 2),
            'eps': round(np.random.uniform(2, 15), 2),
            'dividend_yield': f"{np.random.uniform(0.5, 4.5):.2f}%",
            'sector': 'Technology/Finance',
            'industry': 'Enterprise'
        }


# ==================== DATABASE FUNCTIONS (SUPABASE POSTGRESQL) ====================
# Supabase connection string configured from user settings
SUPABASE_DB_URL = 'postgresql://postgres:YOUR_ACTUAL_PASSWORD@db.lyaptqtkyeumynygcnvm.supabase.co:5432/postgres'

def init_db():
    print("=" * 50)
    print("INITIALIZING SUPABASE DATABASE")
    print("=" * 50)
    
    conn = sqlite3.connect('user_data.db') if 'sqlite' in SUPABASE_DB_URL else None
    # For Supabase PostgreSQL, we use psycopg2
    import psycopg2
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
    
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO users 
                (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
            ''', ('Test', 'User', '1234567890', 'test@email.com', 'Student', '2000-01-01', 'test', 'test123'))
        print("✓ Added test user: test/test123")
    except Exception as e:
        print(f"Test user insert note: {e}")
    
    conn.commit()
    conn.close()
    print("✅ Supabase Database tables initialized successfully.")
    print("=" * 50)

def get_connection():
    import psycopg2
    conn = psycopg2.connect(SUPABASE_DB_URL)
    return conn

def add_user(first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO users (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (first_name, last_name, mobile_number, email, occupation, date_of_birth, username, password))
        conn.commit()
        print(f"✅ User '{username}' added successfully to Supabase")
        return True
    except Exception as e:
        print(f"❌ Error adding user: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
            user = cursor.fetchone()
        return user
    finally:
        conn.close()

# ==================== LSTM PREDICTION ====================
def lstm_predict(data, n_future=7, n_past=60, epochs=10):
    if not TENSORFLOW_AVAILABLE:
        last_price = data['Close'].iloc[-1]
        import random
        predictions = []
        for i in range(n_future):
            predictions.append(last_price * (1 + random.uniform(-0.02, 0.03)))
        return predictions
    
    try:
        close_data = data['Close'].values.reshape(-1,1)
        if len(close_data) > 1000:
            close_data = close_data[-1000:]

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
        print(f"LSTM prediction error: {e}")
        last_price = data['Close'].iloc[-1]
        return [last_price * (1 + i * 0.01) for i in range(n_future)]

# ==================== ROUTES ====================

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('main_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('main_page'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
        else:
            user = authenticate_user(username, password)
            if user:
                session['user'] = username
                flash(f'Welcome back, {username}!', 'success')
                return redirect(url_for('main_page'))
            else:
                flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        return redirect(url_for('main_page'))
    
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
        
        errors = []
        if not all([first_name, last_name, mobile_number, email, username, password, confirm_password]):
            errors.append('Please fill all required fields')
        if password != confirm_password:
            errors.append('Passwords do not match')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters')
        
        if errors:
            for error in errors:
                flash(error, 'error')
        else:
            success = add_user(first_name, last_name, mobile_number, email, 
                             occupation, date_of_birth, username, password)
            if success:
                flash('Account created successfully! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username already exists or database error occurred.', 'error')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    username = session.get('user', 'User')
    session.pop('user', None)
    flash(f'Goodbye, {username}! You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/users')
def view_users():
    if session.get('user') != 'test': 
        return "Access Denied", 403
        
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, first_name, last_name, email, username FROM users')
            users = cursor.fetchall()
    finally:
        conn.close()
    
    html = "<h1>Registered Users on Supabase</h1><ul>"
    for user in users:
        html += f"<li>ID: {user[0]} | Name: {user[1]} {user[2]} | Email: {user[3]} | Username: {user[4]}</li>"
    html += "</ul>"
    return html

@app.route('/main', methods=['GET', 'POST'])
def main_page():
    if 'user' not in session:
        return redirect(url_for('login'))

    selected_display_name = request.form.get('stock', 'Apple Inc. (US)')
    selected_symbol = display_names.get(selected_display_name, 'AAPL')

    default_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')

    start_date = request.form.get('start_date', default_start)
    end_date = request.form.get('end_date', default_end)

    context = {
        'username': session['user'],
        'stocks': display_names,
        'selected_stock': selected_display_name,
        'selected_symbol': selected_symbol,
        'start_date': start_date,
        'end_date': end_date,
        'has_data': False,
        'price_data': None,
        'lr_data': None,
        'lstm_data': None,
        'recent_data': None,
        'current_price': None,
        'last_date': None
    }

    df, is_simulated = get_stock_data_with_fallback(selected_symbol, start_date, end_date)

    if df.empty:
        flash(f'No data available for {selected_display_name}', 'warning')
        return render_template('main.html', **context)
        
    if is_simulated:
        flash(f'Server is in Demo Mode. Displaying simulated market data for {selected_display_name} to bypass Yahoo restrictions.', 'info')

    context['has_data'] = True
    close_prices = df['Close']

    price_data = {
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'prices': close_prices.tolist()
    }
    context['price_data'] = json.dumps(price_data)

    context['current_price'] = round(float(close_prices.iloc[-1]), 2)
    context['last_date'] = df.index[-1].strftime('%Y-%m-%d')

    recent_data = []
    for idx, row in df.tail(10).iterrows():
        recent_data.append({
            'Date': idx.strftime('%Y-%m-%d'),
            'Open': round(float(row['Open']), 2),
            'High': round(float(row['High']), 2),
            'Low': round(float(row['Low']), 2),
            'Close': round(float(row['Close']), 2),
            'Volume': int(row['Volume'])
        })

    context['recent_data'] = recent_data

    # ---- Linear Regression ----
    try:
        df_lr = df.copy()
        df_lr['Day'] = range(len(df_lr))

        X = df_lr[['Day']].values
        y = close_prices.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        lr_data = {
            'dates': df_lr.index[-len(y_test):].strftime('%Y-%m-%d').tolist(),
            'actual': y_test.tolist(),
            'predicted': y_pred.tolist()
        }

        context['lr_data'] = json.dumps(lr_data)

    except Exception as e:
        context['lr_error'] = str(e)

    # ---- LSTM Prediction ----
    try:
        lstm_predictions = lstm_predict(df)

        future_dates = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=7
        ).strftime('%Y-%m-%d').tolist()

        lstm_data = {
            'historical_dates': df.index[-30:].strftime('%Y-%m-%d').tolist(),
            'historical_prices': close_prices[-30:].tolist(),
            'future_dates': future_dates,
            'future_prices': [round(p, 2) for p in lstm_predictions]
        }

        context['lstm_data'] = json.dumps(lstm_data)

        lstm_table = []
        last_price = float(close_prices.iloc[-1])

        for d, p in zip(future_dates, lstm_predictions):
            change = ((p - last_price) / last_price) * 100
            lstm_table.append({
                'date': d,
                'price': round(p, 2),
                'change': round(change, 2)
            })
            last_price = p

        context['lstm_table_data'] = lstm_table

    except Exception as e:
        context['lstm_error'] = str(e)

    return render_template('main.html', **context)

@app.route('/fundamental', methods=['GET', 'POST'])
def fundamental_data():
    if 'user' not in session:
        return redirect(url_for('login'))

    selected_symbols = []

    if request.method == 'POST':
        selected_names = request.form.getlist('stocks')
        selected_symbols = [
            display_names[name]
            for name in selected_names
            if name in display_names
        ]

    fundamentals_data = {}

    for symbol in selected_symbols:
        fundamentals_data[symbol] = get_fundamentals_with_fallback(symbol, stock_names.get(symbol, symbol))

    return render_template(
        'fundamental.html',
        stocks=display_names,
        fundamentals=fundamentals_data
    )

@app.route('/news', methods=['GET', 'POST'])
def news():
    if 'user' not in session:
        return redirect(url_for('login'))

    news_data = {}

    if request.method == 'POST':
        selected_names = request.form.getlist('stocks')
        selected_symbols = [
            display_names[name]
            for name in selected_names
            if name in display_names
        ]

        for symbol in selected_symbols:
            company = stock_names.get(symbol, symbol)

            try:
                news_items = get_stock_news(company)
                news_data[symbol] = {
                    'company': company,
                    'news': news_items
                }
            except Exception as e:
                news_data[symbol] = {
                    'company': company,
                    'error': str(e)
                }

    return render_template(
        'news.html',
        stocks=display_names,
        news_data=news_data
    )

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    selected_symbols = []
    sentiment_data = {}
    
    if request.method == 'POST':
        selected_names = request.form.getlist('stocks')
        selected_symbols = [display_names[n] for n in selected_names if n in display_names]
        
        for symbol in selected_symbols:
            try:
                company = stock_names.get(symbol, symbol)
                company_clean = company.split("(")[0].strip()
                
                query = company_clean.replace(" ", "+")
                rss_url = f"https://news.google.com/rss/search?q={query}+stock"
                
                feed = feedparser.parse(rss_url)
                
                sentiments = []
                pos = neg = neu = 0
                
                for entry in feed.entries[:10]:
                    text = entry.title
                    score = sid.polarity_scores(text)["compound"]
                    
                    sentiment = "Neutral"
                    if score >= 0.05:
                        sentiment = "Positive"
                        pos += 1
                    elif score <= -0.05:
                        sentiment = "Negative"
                        neg += 1
                    else:
                        neu += 1
                    
                    sentiments.append({
                        'title': text[:100] + "..." if len(text) > 100 else text,
                        'sentiment': sentiment,
                        'score': round(score, 3)
                    })
                
                sentiment_data[symbol] = {
                    'company': company,
                    'sentiments': sentiments,
                    'summary': {
                        'positive': pos,
                        'negative': neg,
                        'neutral': neu,
                        'overall': 'Positive' if pos > neg else 'Negative' if neg > pos else 'Neutral'
                    }
                }
            except Exception as e:
                sentiment_data[symbol] = {'error': str(e)}
    
    return render_template('sentiment.html',
                           stocks=display_names,
                           sentiment_data=sentiment_data)

init_db()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
