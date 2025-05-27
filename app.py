import streamlit as st
import time
import sqlite3
from datetime import datetime
import random

# === Placeholder: Real Goodwill API ===
class GoodwillAPI:
    def __init__(self, user_id, password):
        self.user_id = user_id
        self.password = password
        self.authenticated = False

    def login(self):
        self.authenticated = True  # Replace with real login logic
        return self.authenticated

    def get_ltp(self, symbol):
        return round(random.uniform(100, 2500), 2)  # Replace with real tick-level data

    def place_order(self, symbol, action, qty):
        return {
            "status": "success",
            "symbol": symbol,
            "price": self.get_ltp(symbol),
            "time": datetime.now().strftime("%H:%M:%S")
        }

# === Scalping Strategies ===
def strategy_momentum(symbol, ltp): return random.random() > 0.8
def strategy_vwap_bounce(symbol, ltp): return random.random() > 0.85
def strategy_fakeout(symbol, ltp): return random.random() > 0.9
def strategy_ema_cross(symbol, ltp): return random.random() > 0.75
def strategy_vwap_ema(symbol, ltp): return random.random() > 0.88
def strategy_volume_pullback(symbol, ltp): return random.random() > 0.86

strategies = {
    "1-Min Momentum Spike": strategy_momentum,
    "VWAP Bounce Scalping": strategy_vwap_bounce,
    "Fakeout Reversal Catch": strategy_fakeout,
    "EMA 9/21 Crossover": strategy_ema_cross,
    "VWAP + EMA Combo": strategy_vwap_ema,
    "Volume Surge Pullback": strategy_volume_pullback
}

# === UI ===
st.set_page_config(page_title="Scalping Bot", layout="wide")
st.title("Scalping Bot (Goodwill API)")

user_id = st.text_input("Goodwill User ID")
password = st.text_input("Password", type="password")
capital = st.number_input("Total Capital", min_value=10000, value=100000)
mode = st.radio("Mode", ["Paper", "Live"])
symbols = st.multiselect("Symbols to Monitor", [
    'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK',
    'TATAMOTORS', 'TCS', 'INFY', 'LT', 'MARUTI',
    'BAJFINANCE', 'ITC', 'NESTLEIND', 'SUNPHARMA', 'KOTAKBANK',
    'ULTRACEMCO', 'POWERGRID', 'ADANIENT', 'ADANIPORTS', 'HINDUNILVR'
], default=['RELIANCE', 'ICICIBANK', 'TATAMOTORS'])

start_btn = st.button("Start Bot")
log_box = st.empty()

def init_db():
    conn = sqlite3.connect("scalping_log.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, strategy TEXT, entry_time TEXT,
            exit_time TEXT, result TEXT, pnl REAL, mode TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

if start_btn and user_id and password:
    log_box.write("Logging in to Goodwill API...")
    gw = GoodwillAPI(user_id, password)
    if gw.login():
        log_box.write("Login successful. Running strategies...")
        trades_taken = 0
        for symbol in symbols:
            if trades_taken >= 4:
                break
            ltp = gw.get_ltp(symbol)
            for name, func in strategies.items():
                if func(symbol, ltp):
                    entry_price = ltp
                    target_price = round(entry_price * 1.01, 2)
                    stop_loss = round(entry_price * 0.99, 2)
                    time.sleep(1)
                    exit_price = gw.get_ltp(symbol)
                    pnl = round(exit_price - entry_price, 2)
                    trades_taken += 1

                    conn = sqlite3.connect("scalping_log.db")
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO trades (symbol, strategy, entry_time, exit_time, result, pnl, mode) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (symbol, name, datetime.now().strftime('%H:%M:%S'), datetime.now().strftime('%H:%M:%S'),
                         "Closed", pnl, mode))
                    conn.commit()
                    conn.close()

                    log_box.write(f"{symbol} | {name} | Entry: {entry_price} | Exit: {exit_price} | PnL: {pnl}")
    else:
        log_box.write("Login failed. Check credentials.")
elif start_btn:
    log_box.write("Please enter your Goodwill credentials.")
