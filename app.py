#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - Goodwill OAuth Flow
High-frequency scalping with proper Goodwill OAuth authentication
- Per trade risk: 15%
- Max 4 positions  
- 2-3 min max hold time
- Dynamic/adaptive stop loss
- Risk:Reward 1:2
- Dynamic entry logic
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
from collections import deque

try:
    import requests
    REQUESTS_IMPORTED = True
except ImportError:
    REQUESTS_IMPORTED = False
    st.error("❌ requests library not installed. Please install: pip install requests")

# Set page config
st.set_page_config(
    page_title="⚡ FlyingBuddha Scalping Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CORE CLASSES ====================

class GoodwillOAuthHandler:
    """Handles Goodwill OAuth authentication flow"""
    
    def __init__(self):
        self.session = None
        self.access_token = None
        self.api_key = None
        self.api_secret = None
        self.is_connected = False
        self.client_id = None
        self.instrument_token_map = {}
        self.last_login_time = None
        
    def generate_signature(self, api_key, request_token, api_secret):
        """Generate SHA-256 signature for Goodwill API"""
        checksum = f"{api_key}{request_token}{api_secret}"
        signature = hashlib.sha256(checksum.encode()).hexdigest()
        return signature
    
    def get_login_url(self, api_key):
        """Generate Goodwill login URL"""
        self.api_key = api_key
        return f"https://api.gwcindia.in/v1/login?api_key={api_key}"
    
    def exchange_token(self, api_key, request_token, api_secret):
        """Exchange request_token for access_token"""
        if not REQUESTS_IMPORTED:
            st.error("❌ requests library required")
            return False
        
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Generate signature
            signature = self.generate_signature(api_key, request_token, api_secret)
            
            # Exchange token
            url = "https://api.gwcindia.in/v1/login-response"
            payload = {
                "api_key": api_key,
                "request_token": request_token,
                "signature": signature
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "success" and "data" in data:
                self.access_token = data["data"]["access_token"]
                self.client_id = data["data"]["clnt_id"]
                self.is_connected = True
                self.last_login_time = datetime.now()
                
                # Store in session state
                st.session_state["gw_logged_in"] = True
                st.session_state["gw_access_token"] = self.access_token
                st.session_state["gw_client_id"] = self.client_id
                st.session_state["gw_api_key"] = api_key
                
                self.setup_session()
                self.load_instrument_tokens()
                return True
            else:
                st.error(f"❌ Token exchange failed: {data.get('error_msg', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request error: {e}")
            return False
        except Exception as e:
            st.error(f"❌ Token exchange error: {e}")
            return False
    
    def setup_session(self):
        """Setup requests session with headers"""
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        })
    
    def load_instrument_tokens(self):
        """Load instrument tokens for trading symbols"""
        symbols = [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS"
        ]
        
        # For now, we'll fetch tokens dynamically when needed
        # This is more efficient than pre-loading all symbols
        st.success(f"✅ Ready to trade {len(symbols)} symbols")
    
    def get_instrument_token(self, symbol):
        """Get instrument token for a symbol"""
        if symbol in self.instrument_token_map:
            return self.instrument_token_map[symbol]
        
        # Fetch token dynamically
        try:
            response = self.session.post(
                "https://api.gwcindia.in/v1/fetchsymbol",
                json={"s": symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success" and data.get("data"):
                for item in data["data"]:
                    if item["exchange"] == "NSE" and symbol.upper() in item["symbol"]:
                        token = item["token"]
                        self.instrument_token_map[symbol] = token
                        return token
            return None
            
        except Exception as e:
            print(f"Error fetching token for {symbol}: {e}")
            return None
    
    def place_order(self, symbol, action, quantity, price, order_type="MKT"):
        """Place order through Goodwill API"""
        if not self.is_connected or not self.session:
            st.error("❌ Not connected to Goodwill")
            return None
        
        try:
            # Prepare order payload
            payload = {
                "tsym": f"{symbol}-EQ",
                "exchange": "NSE",
                "trantype": action.upper(),
                "validity": "DAY",
                "pricetype": order_type,
                "qty": str(quantity),
                "discqty": "0",
                "price": str(price) if order_type != "MKT" else "0",
                "trgprc": "0",
                "product": "MIS",
                "amo": "NO"
            }
            
            response = self.session.post(
                "https://api.gwcindia.in/v1/placeorder",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success" and "data" in data:
                order_id = data["data"]["nstordno"]
                st.success(f"🎯 Order Placed: {action} {quantity} {symbol} @ ₹{price:.2f} | ID: {order_id}")
                return order_id
            else:
                st.error(f"❌ Order Failed: {data.get('error_msg', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"❌ Order Error: {e}")
            return None
    
    def get_live_data(self, symbol):
        """Get live market data for symbol"""
        if not self.is_connected or not self.session:
            return None
        
        try:
            token = self.get_instrument_token(symbol)
            if not token:
                return None
            
            response = self.session.post(
                "https://api.gwcindia.in/v1/getquote",
                json={"exchange": "NSE", "token": token},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success" and "data" in data:
                quote = data["data"]
                return {
                    'price': float(quote.get("last_price", 0)),
                    'volume': int(quote.get("volume", 0)),
                    'high': float(quote.get("high", 0)),
                    'low': float(quote.get("low", 0)),
                    'timestamp': datetime.now()
                }
            return None
            
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
            return None
    
    def logout(self):
        """Logout and clear session"""
        if self.session:
            try:
                self.session.get("https://api.gwcindia.in/v1/logout", timeout=5)
            except:
                pass
        
        self.session = None
        self.access_token = None
        self.is_connected = False
        
        # Clear session state
        for key in ["gw_logged_in", "gw_access_token", "gw_client_id", "gw_api_key"]:
            if key in st.session_state:
                del st.session_state[key]


class AdaptiveScalpingBot:
    """Advanced scalping bot with adaptive algorithms"""
    
    def __init__(self):
        self.data_feed = GoodwillOAuthHandler()
        self.is_running = False
        self.capital = 100000.0
        self.positions = {}
        self.trades = []
        self.signals = []
        self.pnl = 0.0
        self.mode = "PAPER"  # PAPER or LIVE
        
        # Trading parameters
        self.max_positions = 4
        self.risk_per_trade = 0.15  # 15% per trade
        self.max_hold_time = 180  # 3 minutes in seconds
        self.risk_reward_ratio = 2.0  # 1:2
        
        # Adaptive parameters
        self.volatility_cache = {}
        self.momentum_cache = {}
        self.volume_profile_cache = {}
        
        # Market scanning
        self.symbols = [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS"
        ]
        
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for trade logging"""
        try:
            self.conn = sqlite3.connect('scalping_trades.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    hold_time REAL,
                    strategy TEXT,
                    exit_reason TEXT
                )
            ''')
            self.conn.commit()
        except Exception as e:
            st.error(f"Database init error: {e}")
    
    def calculate_dynamic_volatility(self, symbol):
        """Calculate adaptive volatility measure"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="5d", interval="1m")
            
            if len(data) < 20:
                return 0.02
            
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            volatility = max(0.005, min(0.05, volatility))
            self.volatility_cache[symbol] = volatility
            return volatility
            
        except Exception as e:
            print(f"Volatility calculation error for {symbol}: {e}")
            return 0.02
    
    def calculate_momentum_score(self, symbol):
        """Calculate multi-timeframe momentum"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 15:
                return 0
            
            current_price = data['Close'].iloc[-1]
            mom_1m = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            mom_5m = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100 if len(data) >= 6 else 0
            mom_15m = ((current_price - data['Close'].iloc[-16]) / data['Close'].iloc[-16]) * 100 if len(data) >= 16 else 0
            
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].mean()
            volume_factor = min(recent_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
            
            momentum_score = (mom_1m * 0.5 + mom_5m * 0.3 + mom_15m * 0.2) * volume_factor
            
            self.momentum_cache[symbol] = {
                'score': momentum_score,
                'direction': 1 if momentum_score > 0 else -1,
                'strength': abs(momentum_score),
                'volume_factor': volume_factor
            }
            
            return momentum_score
            
        except Exception as e:
            print(f"Momentum calculation error for {symbol}: {e}")
            return 0
    
    def scan_for_opportunities(self):
        """Dynamic market scanning with adaptive thresholds"""
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Always use broker data feed if connected
                if self.data_feed.is_connected:
                    live_data = self.data_feed.get_live_data(symbol)
                else:
                    # Fallback to yfinance only if broker not connected
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period="1d", interval="1m")
                    if len(hist) > 0:
                        latest = hist.iloc[-1]
                        live_data = {
                            'price': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'timestamp': datetime.now()
                        }
                    else:
                        continue
                
                if not live_data:
                    continue
                
                volatility = self.calculate_dynamic_volatility(symbol)
                momentum = self.calculate_momentum_score(symbol)
                
                min_momentum_threshold = volatility * 50
                volume_threshold = 1.5
                
                momentum_data = self.momentum_cache.get(symbol, {})
                strength = momentum_data.get('strength', 0)
                volume_factor = momentum_data.get('volume_factor', 1)
                
                if (strength > min_momentum_threshold and 
                    volume_factor > volume_threshold and
                    len(self.positions) < self.max_positions):
                    
                    score = strength * volume_factor * (1 + volatility * 10)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'score': score,
                        'momentum': momentum,
                        'volatility': volatility,
                        'volume_factor': volume_factor,
                        'price': live_data['price'],
                        'direction': momentum_data.get('direction', 1)
                    })
                    
            except Exception as e:
                print(f"Scanning error for {symbol}: {e}")
                continue
        
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities[:3]
    
    def calculate_position_size(self, symbol, entry_price, volatility):
        """Calculate adaptive position size based on volatility and risk"""
        try:
            risk_amount = self.capital * self.risk_per_trade
            stop_loss_pct = max(0.002, min(0.01, volatility * 2))
            stop_loss_amount = entry_price * stop_loss_pct
            
            if stop_loss_amount > 0:
                quantity = int(risk_amount / stop_loss_amount)
                quantity = max(1, min(quantity, 1000))
                return quantity, stop_loss_pct
            
            return 1, 0.005
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 1, 0.005
    
    def execute_entry(self, opportunity):
        """Execute dynamic entry with adaptive parameters"""
        try:
            symbol = opportunity['symbol']
            entry_price = opportunity['price']
            direction = opportunity['direction']
            volatility = opportunity['volatility']
            
            quantity, stop_loss_pct = self.calculate_position_size(symbol, entry_price, volatility)
            action = "BUY" if direction > 0 else "SELL"
            
            if action == "BUY":
                stop_loss = entry_price * (1 - stop_loss_pct)
                target = entry_price * (1 + (stop_loss_pct * self.risk_reward_ratio))
            else:
                stop_loss = entry_price * (1 + stop_loss_pct)
                target = entry_price * (1 - (stop_loss_pct * self.risk_reward_ratio))
            
            # Place order based on mode
            if self.mode == "LIVE":
                order_id = self.data_feed.place_order(symbol, action, quantity, entry_price)
            else:
                # Paper mode - simulate order but use real broker data if available
                order_id = f"PAPER_{int(datetime.now().timestamp())}"
                data_source = "Goodwill Data" if self.data_feed.is_connected else "yfinance Data"
                st.info(f"📝 PAPER TRADE: {action} {quantity} {symbol} @ ₹{entry_price:.2f} (Using {data_source})")
            
            if order_id:
                position_id = f"{symbol}_{int(datetime.now().timestamp())}"
                
                self.positions[position_id] = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'entry_time': datetime.now(),
                    'order_id': order_id,
                    'volatility': volatility,
                    'initial_stop_pct': stop_loss_pct,
                    'trailing_stop': stop_loss,
                    'highest_profit': 0,
                    'lowest_profit': 0
                }
                
                st.success(f"🚀 Position Opened: {action} {quantity} {symbol} @ ₹{entry_price:.2f}")
                return position_id
            
            return None
            
        except Exception as e:
            st.error(f"Entry execution error: {e}")
            return None
    
    def update_adaptive_stops(self, position_id, current_price):
        """Update trailing and adaptive stop losses"""
        position = self.positions[position_id]
        
        if position['action'] == 'BUY':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        position['highest_profit'] = max(position['highest_profit'], current_pnl)
        position['lowest_profit'] = min(position['lowest_profit'], current_pnl)
        
        if current_pnl > 0:
            trail_factor = min(0.5, current_pnl / (position['entry_price'] * position['quantity'] * 0.01))
            
            if position['action'] == 'BUY':
                new_stop = current_price * (1 - position['initial_stop_pct'] * (1 - trail_factor))
                position['trailing_stop'] = max(position['trailing_stop'], new_stop)
            else:
                new_stop = current_price * (1 + position['initial_stop_pct'] * (1 - trail_factor))
                position['trailing_stop'] = min(position['trailing_stop'], new_stop)
    
    def check_exit_conditions(self, position_id, current_price):
        """Check dynamic exit conditions"""
        position = self.positions[position_id]
        
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        if hold_time > self.max_hold_time:
            return "TIME_EXIT"
        
        self.update_adaptive_stops(position_id, current_price)
        
        if position['action'] == 'BUY':
            if current_price <= position['trailing_stop']:
                return "STOP_LOSS"
            if current_price >= position['target']:
                return "TARGET_HIT"
        else:
            if current_price >= position['trailing_stop']:
                return "STOP_LOSS"
            if current_price <= position['target']:
                return "TARGET_HIT"
        
        momentum_data = self.momentum_cache.get(position['symbol'], {})
        current_direction = momentum_data.get('direction', 0)
        
        if position['action'] == 'BUY' and current_direction < 0:
            return "MOMENTUM_REVERSAL"
        elif position['action'] == 'SELL' and current_direction > 0:
            return "MOMENTUM_REVERSAL"
        
        return None
    
    def close_position(self, position_id, exit_price, exit_reason):
        """Close position and log trade"""
        if position_id not in self.positions:
            return
        
        position = self.positions.pop(position_id)
        
        if position['action'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        self.capital += pnl
        self.pnl += pnl
        
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        
        trade_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': position['symbol'],
            'action': f"CLOSE_{position['action']}",
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': round(pnl, 2),
            'hold_time': round(hold_time, 1),
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade_record)
        
        # Place exit order based on mode
        exit_action = "SELL" if position['action'] == "BUY" else "BUY"
        if self.mode == "LIVE":
            self.data_feed.place_order(position['symbol'], exit_action, position['quantity'], exit_price)
        else:
            data_source = "Goodwill Data" if self.data_feed.is_connected else "yfinance Data"
            st.info(f"📝 PAPER EXIT: {exit_action} {position['quantity']} {position['symbol']} @ ₹{exit_price:.2f} (Using {data_source})")
        
        st.info(f"✅ Position Closed: {position['symbol']} | P&L: ₹{pnl:.2f} | Reason: {exit_reason}")
    
    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            opportunities = self.scan_for_opportunities()
            
            for opp in opportunities:
                if len(self.positions) < self.max_positions:
                    self.execute_entry(opp)
            
            positions_to_close = []
            for pos_id in list(self.positions.keys()):
                position = self.positions[pos_id]
                
                # Get current price - always use broker data if connected
                if self.data_feed.is_connected:
                    live_data = self.data_feed.get_live_data(position['symbol'])
                    current_price = live_data['price'] if live_data else position['entry_price']
                else:
                    # Fallback to yfinance only if broker not connected
                    ticker = yf.Ticker(f"{position['symbol']}.NS")
                    hist = ticker.history(period="1d", interval="1m")
                    current_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
                
                exit_reason = self.check_exit_conditions(pos_id, current_price)
                if exit_reason:
                    positions_to_close.append((pos_id, current_price, exit_reason))
            
            for pos_id, exit_price, exit_reason in positions_to_close:
                self.close_position(pos_id, exit_price, exit_reason)
                
        except Exception as e:
            st.error(f"Trading cycle error: {e}")
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_trade': 0, 'max_profit': 0, 'max_loss': 0, 'avg_hold_time': 0
            }
        
        pnls = [trade['pnl'] for trade in self.trades]
        hold_times = [trade['hold_time'] for trade in self.trades]
        winning_trades = len([p for p in pnls if p > 0])
        
        return {
            'total_trades': len(self.trades),
            'win_rate': (winning_trades / len(self.trades)) * 100,
            'total_pnl': round(sum(pnls), 2),
            'avg_trade': round(np.mean(pnls), 2),
            'max_profit': round(max(pnls), 2) if pnls else 0,
            'max_loss': round(min(pnls), 2) if pnls else 0,
            'avg_hold_time': round(np.mean(hold_times), 1) if hold_times else 0
        }


# ==================== STREAMLIT UI ====================

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = AdaptiveScalpingBot()

if 'gw_logged_in' not in st.session_state:
    st.session_state.gw_logged_in = False

bot = st.session_state.bot

# Main title
st.title("⚡ FlyingBuddha Adaptive Scalping Bot")

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.header("⚡ Scalping Bot Controls")

# Trading Mode Selection
st.sidebar.subheader("🔄 Trading Mode")

current_mode_icon = "🔴" if bot.mode == "LIVE" else "🟠"
current_mode_text = f"{current_mode_icon} {bot.mode} Mode"
if bot.data_feed.is_connected:
    current_mode_text += " (Connected to Goodwill)"
else:
    current_mode_text += " (Not Connected)"

st.sidebar.markdown(f"**Current:** {current_mode_text}")

mode_col1, mode_col2 = st.sidebar.columns(2)
with mode_col1:
    if st.button("🟠 Paper Mode"):
        bot.mode = "PAPER"
        st.success("✅ Switched to Paper Trading")
        st.rerun()

with mode_col2:
    if st.button("🔴 Live Mode"):
        bot.mode = "LIVE"
        if not st.session_state.gw_logged_in:
            st.warning("⚠️ Please login to Goodwill first!")
        else:
            st.success("✅ Switched to Live Trading Mode")
        st.rerun()

# ==================== FIXED GOODWILL OAUTH LOGIN ====================

st.sidebar.subheader("🔐 Goodwill OAuth Login")

if not st.session_state.gw_logged_in:
    
    # Initialize OAuth state if not exists
    if 'oauth_step' not in st.session_state:
        st.session_state.oauth_step = 1
    if 'oauth_api_key' not in st.session_state:
        st.session_state.oauth_api_key = ""
    if 'oauth_api_secret' not in st.session_state:
        st.session_state.oauth_api_secret = ""
    if 'oauth_login_url' not in st.session_state:
        st.session_state.oauth_login_url = ""
    
    # STEP 1: Enter and Submit API Credentials
    if st.session_state.oauth_step == 1:
        st.sidebar.markdown("**Step 1: Enter API Credentials**")
        
        api_key = st.sidebar.text_input(
            "API Key", 
            value=st.session_state.oauth_api_key,
            type="password",
            help="Your Goodwill API Key from developer portal"
        )
        
        api_secret = st.sidebar.text_input(
            "API Secret", 
            value=st.session_state.oauth_api_secret,
            type="password",
            help="Your Goodwill API Secret from developer portal"
        )
        
        # Submit credentials button
        if st.sidebar.button("📝 Submit Credentials", use_container_width=True):
            if api_key and api_secret:
                # Store credentials
                st.session_state.oauth_api_key = api_key
                st.session_state.oauth_api_secret = api_secret
                
                # Generate login URL
                login_url = bot.data_feed.get_login_url(api_key)
                st.session_state.oauth_login_url = login_url
                
                # Move to step 2
                st.session_state.oauth_step = 2
                st.success("✅ Credentials submitted! Proceeding to login...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter both API Key and Secret")
        
        # Instructions for Step 1
        st.sidebar.markdown("---")
        st.sidebar.info("**Step 1 Instructions:**\n1. Enter your API Key and Secret\n2. Click 'Submit Credentials'\n3. This will validate and prepare the OAuth flow")
    
    # STEP 2: Login to Goodwill
    elif st.session_state.oauth_step == 2:
        st.sidebar.markdown("**Step 2: Login to Goodwill**")
        st.sidebar.success(f"✅ Using API Key: {st.session_state.oauth_api_key[:8]}...")
        
        # Show login link
        st.sidebar.markdown(f"[🚀 **Click Here to Login**]({st.session_state.oauth_login_url})")
        st.sidebar.caption("⚠️ This will open Goodwill login page")
        
        # What to expect
        st.sidebar.info("""
        **What happens next:**
        1. Click the login link above
        2. Login with your Goodwill credentials
        3. After login, you'll get a URL with 'request_token'
        4. Copy the request_token value
        5. Return here and paste it below
        """)
        
        # Button to proceed to step 3
        if st.sidebar.button("✅ I've Logged In - Enter Token", use_container_width=True):
            st.session_state.oauth_step = 3
            st.rerun()
        
        # Back button
        if st.sidebar.button("⬅️ Back to Step 1", use_container_width=True):
            st.session_state.oauth_step = 1
            st.rerun()
    
    # STEP 3: Enter Request Token
    elif st.session_state.oauth_step == 3:
        st.sidebar.markdown("**Step 3: Enter Request Token**")
        st.sidebar.success(f"✅ Credentials Ready: {st.session_state.oauth_api_key[:8]}...")
        
        # Show expected URL format
        st.sidebar.markdown("**Look for URL like:**")
        st.sidebar.code("http://localhost:8501/?request_token=ABC123XYZ...")
        st.sidebar.caption("Copy only the ABC123XYZ... part")
        
        request_token = st.sidebar.text_input(
            "Request Token",
            help="Copy the 'request_token' parameter from redirect URL",
            placeholder="e.g., ABC123XYZ789..."
        )
        
        # Complete OAuth login
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔗 Complete Login", use_container_width=True):
                if request_token:
                    with st.spinner("Exchanging tokens..."):
                        success = bot.data_feed.exchange_token(
                            st.session_state.oauth_api_key, 
                            request_token, 
                            st.session_state.oauth_api_secret
                        )
                        if success:
                            # Reset OAuth state
                            st.session_state.oauth_step = 1
                            st.session_state.oauth_api_key = ""
                            st.session_state.oauth_api_secret = ""
                            st.session_state.oauth_login_url = ""
                            
                            st.success("✅ Successfully logged in to Goodwill!")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Token exchange failed. Please try again.")
                else:
                    st.error("❌ Please enter the request token")
        
        with col2:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.oauth_step = 2
                st.rerun()
        
        # Troubleshooting section
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔧 Troubleshooting:**")
        st.sidebar.markdown("""
        **No request_token in URL?**
        - Make sure you clicked the login link from Step 2
        - Complete the full login process (User ID + Password + OTP)
        - Look for the redirect URL in your browser
        - Check if you're being redirected to localhost:8501
        
        **Token looks wrong?**
        - Don't include 'request_token=' in your input
        - Copy only the value after the = sign
        - Remove any extra characters or spaces
        """)
    
    # Show current step indicator
    st.sidebar.markdown("---")
    step_indicator = f"**Current Step: {st.session_state.oauth_step}/3**"
    if st.session_state.oauth_step == 1:
        step_indicator += " - Enter Credentials"
    elif st.session_state.oauth_step == 2:
        step_indicator += " - Login to Goodwill"
    elif st.session_state.oauth_step == 3:
        step_indicator += " - Enter Token"
    
    st.sidebar.info(step_indicator)

else:
    # Show logged in status
    client_id = st.session_state.get("gw_client_id", "Unknown")
    st.sidebar.success(f"✅ Logged in as: {client_id}")
    
    connection_time = ""
    if bot.data_feed.last_login_time:
        elapsed = datetime.now() - bot.data_feed.last_login_time
        connection_time = f"{int(elapsed.total_seconds() / 60)}m ago"
    
    st.sidebar.info(f"Connected: {connection_time}")
    
    # Test connection button
    if st.sidebar.button("🧪 Test Connection", use_container_width=True):
        with st.spinner("Testing API connection..."):
            # Try to fetch a test symbol
            test_data = bot.data_feed.get_live_data("RELIANCE")
            if test_data:
                st.sidebar.success("✅ API connection working!")
                st.sidebar.json({
                    "Symbol": "RELIANCE",
                    "Price": f"₹{test_data['price']:.2f}",
                    "Volume": f"{test_data['volume']:,}",
                    "Timestamp": test_data['timestamp'].strftime("%H:%M:%S")
                })
            else:
                st.sidebar.error("❌ API connection failed")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        bot.data_feed.logout()
        bot.is_running = False
        
        # Reset OAuth state
        st.session_state.oauth_step = 1
        st.session_state.oauth_api_key = ""
        st.session_state.oauth_api_secret = ""
        st.session_state.oauth_login_url = ""
        
        st.success("✅ Logged out successfully")
        st.rerun()

# ==================== BOT SETTINGS ====================

st.sidebar.subheader("⚙️ Bot Settings")

new_capital = st.sidebar.number_input(
    "Trading Capital (₹)", 
    value=bot.capital, 
    min_value=10000.0, 
    step=5000.0,
    format="%.0f"
)
if new_capital != bot.capital:
    bot.capital = new_capital

risk_per_trade = st.sidebar.slider(
    "Risk per Trade (%)", 
    min_value=5, 
    max_value=25, 
    value=int(bot.risk_per_trade * 100),
    step=1
)
bot.risk_per_trade = risk_per_trade / 100

max_positions = st.sidebar.slider(
    "Max Positions", 
    min_value=1, 
    max_value=6, 
    value=bot.max_positions
)
bot.max_positions = max_positions

max_hold_minutes = st.sidebar.slider(
    "Max Hold Time (minutes)", 
    min_value=1, 
    max_value=5, 
    value=int(bot.max_hold_time / 60),
    step=1
)
bot.max_hold_time = max_hold_minutes * 60

risk_reward = st.sidebar.slider(
    "Risk:Reward Ratio", 
    min_value=1.5, 
    max_value=3.0, 
    value=bot.risk_reward_ratio,
    step=0.1
)
bot.risk_reward_ratio = risk_reward

# ==================== BOT CONTROL BUTTONS ====================

st.sidebar.subheader("🎮 Bot Control")

# Check if we can start the bot
can_start = True
start_button_text = "🚀 Start Bot"

if not st.session_state.gw_logged_in:
    can_start = False
    start_button_text = "🔐 Login Required"
elif not bot.data_feed.is_connected:
    can_start = False
    start_button_text = "❌ Connection Failed"

col_ctrl1, col_ctrl2 = st.sidebar.columns(2)

with col_ctrl1:
    if st.button(start_button_text, disabled=bot.is_running or not can_start, use_container_width=True):
        bot.is_running = True
        st.success("🚀 Bot Started!")
        st.rerun()

with col_ctrl2:
    if st.button("⏹️ Stop Bot", disabled=not bot.is_running, use_container_width=True):
        bot.is_running = False
        st.success("⏹️ Bot Stopped!")
        st.rerun()

# Bot status display
st.sidebar.subheader("📊 Status")
if bot.is_running:
    if bot.mode == "LIVE": 
        st.sidebar.error("🔴 LIVE TRADING ACTIVE")
    else: 
        st.sidebar.warning("🟠 PAPER TRADING ACTIVE")
else:
    st.sidebar.info("⚪ BOT STOPPED")

# Connection status
if bot.data_feed.is_connected:
    st.sidebar.success("🔌 Connected to Goodwill")
else:
    st.sidebar.error("🔌 Not Connected to Goodwill")

# ==================== MAIN DASHBOARD ====================

# Header with current mode banner
if bot.mode == "LIVE":
    if bot.data_feed.is_connected:
        st.error("🔴 **LIVE TRADING ACTIVE** | Connected to Goodwill | ⚠️ REAL MONEY AT RISK")
    else:
        st.warning("🟠 **LIVE MODE** | Not Connected to Goodwill | Login Required")
else:
    if bot.data_feed.is_connected:
        st.warning("🟠 **PAPER TRADING MODE** | Using Goodwill Live Data | Simulated Orders Only")
    else:
        st.info("🔵 **PAPER TRADING MODE** | Not Connected | Login Required for Broker Data")

# ==================== PERFORMANCE METRICS ====================

st.subheader("📊 Performance Dashboard")

# Get performance metrics
perf = bot.get_performance_metrics()

# Main metrics row
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.metric(
        "Capital", 
        f"₹{bot.capital:,.0f}", 
        delta=f"₹{perf['total_pnl']:,.0f}"
    )

with col_m2:
    st.metric(
        "Total Trades", 
        perf['total_trades']
    )

with col_m3:
    st.metric(
        "Win Rate", 
        f"{perf['win_rate']:.1f}%"
    )

with col_m4:
    st.metric(
        "Active Positions", 
        len(bot.positions)
    )

with col_m5:
    st.metric(
        "Avg Hold Time", 
        f"{perf['avg_hold_time']:.1f}s"
    )

# Secondary metrics row
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("Avg Trade P&L", f"₹{perf['avg_trade']:.0f}")

with col_s2:
    st.metric("Max Profit", f"₹{perf['max_profit']:.0f}")

with col_s3:
    st.metric("Max Loss", f"₹{perf['max_loss']:.0f}")

with col_s4:
    mode_display = f"{bot.mode}"
    if bot.data_feed.is_connected:
        mode_display += " (Goodwill Data)"
    else:
        mode_display += " (No Connection)"
    st.metric("Trading Mode", mode_display)

# ==================== ACTIVE POSITIONS ====================

st.subheader("📍 Active Positions")

if bot.positions:
    positions_data = []
    for pos_id, pos in bot.positions.items():
        # Get current price - always use broker data if connected
        if bot.data_feed.is_connected:
            live_data = bot.data_feed.get_live_data(pos['symbol'])
            current_price = live_data['price'] if live_data else pos['entry_price']
        else:
            # Fallback to yfinance only if broker not connected
            try:
                ticker = yf.Ticker(f"{pos['symbol']}.NS")
                hist = ticker.history(period="1d", interval="1m")
                current_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else pos['entry_price']
            except:
                current_price = pos['entry_price']
        
        # Calculate current P&L
        if pos['action'] == 'BUY':
            current_pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:
            current_pnl = (pos['entry_price'] - current_price) * pos['quantity']
        
        # Calculate hold time
        hold_time = (datetime.now() - pos['entry_time']).total_seconds()
        
        positions_data.append({
            'Symbol': pos['symbol'],
            'Action': pos['action'],
            'Qty': pos['quantity'],
            'Entry ₹': f"{pos['entry_price']:.2f}",
            'Current ₹': f"{current_price:.2f}",
            'Stop ₹': f"{pos['trailing_stop']:.2f}",
            'Target ₹': f"{pos['target']:.2f}",
            'P&L ₹': f"{current_pnl:+.0f}",
            'Hold (s)': f"{hold_time:.0f}",
            'Volatility': f"{pos['volatility']*100:.1f}%"
        })
    
    df_positions = pd.DataFrame(positions_data)
    
    # Color code P&L
    def highlight_pnl(val):
        if 'P&L' in val.name:
            color = 'lightgreen' if '+' in str(val) else 'lightcoral' if '-' in str(val) else 'white'
            return [f'background-color: {color}'] * len(val)
        return [''] * len(val)
    
    st.dataframe(
        df_positions.style.apply(highlight_pnl, axis=0),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No active positions")

# ==================== MARKET OPPORTUNITIES ====================

st.subheader("🎯 Live Market Opportunities")

if bot.is_running:
    # Get current opportunities
    opportunities = bot.scan_for_opportunities()
    
    if opportunities:
        opp_data = []
        for opp in opportunities:
            opp_data.append({
                'Symbol': opp['symbol'],
                'Score': f"{opp['score']:.1f}",
                'Price ₹': f"{opp['price']:.2f}",
                'Momentum': f"{opp['momentum']:+.2f}%",
                'Direction': "🔺 BUY" if opp['direction'] > 0 else "🔻 SELL",
                'Volatility': f"{opp['volatility']*100:.1f}%",
                'Volume Factor': f"{opp['volume_factor']:.1f}x"
            })
        
        df_opportunities = pd.DataFrame(opp_data)
        st.dataframe(df_opportunities, use_container_width=True, hide_index=True)
    else:
        st.info("No opportunities meeting criteria currently")
else:
    st.info("Start the bot to scan for opportunities")

# ==================== RECENT TRADES ====================

st.subheader("📈 Recent Trades")

if bot.trades:
    # Show last 10 trades
    recent_trades = bot.trades[-10:]
    trades_data = []
    
    for trade in reversed(recent_trades):  # Most recent first
        trades_data.append({
            'Time': trade['timestamp'].split(' ')[1],  # Just time part
            'Symbol': trade['symbol'],
            'Action': trade['action'],
            'Entry ₹': f"{trade['entry_price']:.2f}",
            'Exit ₹': f"{trade['exit_price']:.2f}",
            'Qty': trade['quantity'],
            'P&L ₹': f"{trade['pnl']:+.0f}",
            'Hold (s)': f"{trade['hold_time']:.0f}",
            'Exit Reason': trade['exit_reason']
        })
    
    df_trades = pd.DataFrame(trades_data)
    
    # Color code P&L
    def highlight_pnl_trades(row):
        if '+' in str(row['P&L ₹']):
            return ['background-color: lightgreen'] * len(row)
        elif '-' in str(row['P&L ₹']):
            return ['background-color: lightcoral'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_trades.style.apply(highlight_pnl_trades, axis=1),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No trades executed yet")

# ==================== LIVE PERFORMANCE CHART ====================

st.subheader("📊 Live Performance Chart")

if bot.trades:
    # Create cumulative P&L chart
    cumulative_pnl = []
    running_total = 0
    trade_numbers = []
    
    for i, trade in enumerate(bot.trades, 1):
        running_total += trade['pnl']
        cumulative_pnl.append(running_total)
        trade_numbers.append(i)
    
    # Create the chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Cumulative P&L Over Time",
        xaxis_title="Trade Number",
        yaxis_title="P&L (₹)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Trade history will appear here once trades are executed")

# ==================== SYSTEM STATUS ====================

st.subheader("🔧 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.info(f"**Bot Status:** {'🟢 Running' if bot.is_running else '⚪ Stopped'}")
    st.info(f"**Trading Mode:** {bot.mode}")

with status_col2:
    st.info(f"**Max Positions:** {bot.max_positions}")
    st.info(f"**Risk per Trade:** {bot.risk_per_trade*100:.0f}%")

with status_col3:
    st.info(f"**Hold Time:** {bot.max_hold_time//60}m")
    st.info(f"**Risk:Reward:** 1:{bot.risk_reward_ratio}")

# Connection status details
connection_col1, connection_col2 = st.columns(2)
with connection_col1:
    if bot.data_feed.is_connected:
        st.success("🟢 **Connected to Goodwill**")
        if bot.data_feed.last_login_time:
            login_time = bot.data_feed.last_login_time.strftime("%H:%M:%S")
            st.caption(f"Logged in at: {login_time}")
    else:
        st.error("🔴 **Not Connected to Goodwill**")
        st.caption("Login required for broker data")

with connection_col2:
    if st.session_state.gw_logged_in:
        client_id = st.session_state.get("gw_client_id", "Unknown")
        st.info(f"**Client:** {client_id}")
    else:
        st.warning("**Status:** Not logged in")

# ==================== AUTO-REFRESH AND TRADING CYCLE ====================

# Auto-refresh logic
if bot.is_running:
    # Run trading cycle
    bot.run_trading_cycle()
    
    # Show activity indicator
    st.info("🔄 Auto-refreshing every 10 seconds...")
    
    # Auto-refresh
    time.sleep(10)
    st.rerun()

# ==================== EMERGENCY CONTROLS ====================

with st.sidebar.expander("🚨 Emergency Controls", expanded=False):
    st.warning("⚠️ Use these controls carefully")
    
    if st.button("🛑 Close All Positions", use_container_width=True):
        if bot.positions:
            positions_to_close = list(bot.positions.keys())
            for pos_id in positions_to_close:
                pos = bot.positions[pos_id]
                
                # Get current price for exit - always use broker data if connected
                if bot.data_feed.is_connected:
                    live_data = bot.data_feed.get_live_data(pos['symbol'])
                    current_price = live_data['price'] if live_data else pos['entry_price']
                else:
                    try:
                        ticker = yf.Ticker(f"{pos['symbol']}.NS")
                        hist = ticker.history(period="1d", interval="1m")
                        current_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else pos['entry_price']
                    except:
                        current_price = pos['entry_price']
                
                bot.close_position(pos_id, current_price, "EMERGENCY_CLOSE")
            
            st.success(f"Closed {len(positions_to_close)} positions")
            st.rerun()
        else:
            st.info("No positions to close")
    
    if st.button("🔄 Reset Bot", use_container_width=True):
        bot.positions = {}
        bot.trades = []
        bot.signals = []
        bot.pnl = 0
        bot.is_running = False
        st.success("Bot reset successfully")
        st.rerun()
    
    if st.button("💾 Export Trades", use_container_width=True):
        if bot.trades:
            df_export = pd.DataFrame(bot.trades)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"scalping_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trades to export")

# ==================== DEBUG INFO ====================

if st.sidebar.checkbox("🔍 Debug Info"):
    st.sidebar.subheader("Debug Information")
    debug_info = {
        "Bot Running": bot.is_running,
        "Trading Mode": bot.mode,
        "GW Logged In": st.session_state.gw_logged_in,
        "GW Connected": bot.data_feed.is_connected,
        "Active Positions": len(bot.positions),
        "Total Trades": len(bot.trades),
        "Capital": bot.capital,
        "P&L": bot.pnl,
        "Cache Sizes": {
            "Volatility": len(bot.volatility_cache),
            "Momentum": len(bot.momentum_cache)
        },
        "Data Feed Status": {
            "Has Session": bot.data_feed.session is not None,
            "Has Token": bot.data_feed.access_token is not None,
            "Token Length": len(bot.data_feed.access_token) if bot.data_feed.access_token else 0,
            "Client ID": st.session_state.get("gw_client_id", "None")
        }
    }
    st.sidebar.json(debug_info)

# ==================== ENHANCED OAUTH HELP SECTION ====================

with st.sidebar.expander("❓ OAuth Help & Troubleshooting", expanded=False):
    st.markdown("### **Prerequisites:**")
    st.markdown("""
    1. **Register at Goodwill Developer Portal:**
       - Go to [developer.gwcindia.in](https://developer.gwcindia.in)
       - Create account and verify email
    
    2. **Create Application:**
       - Create new app in developer portal
       - Note down your **API Key** and **API Secret**
       - Set redirect URI to: `http://localhost:8501`
    """)
    
    st.markdown("### **Step-by-Step Process:**")
    st.markdown("""
    **Step 1: Submit Credentials**
    - Enter API Key and Secret
    - Click "Submit Credentials"
    - System validates and prepares OAuth flow
    
    **Step 2: Login to Goodwill**
    - Click the generated login link
    - Login with Goodwill User ID + Password + OTP
    - You'll be redirected back with request_token
    
    **Step 3: Complete OAuth**
    - Copy the request_token from URL
    - Paste it and click "Complete Login"
    - System exchanges token for access_token
    """)
    
    st.markdown("### **Common Issues:**")
    st.markdown("""
    **❌ No request_token in URL:**
    - Ensure you clicked login link from Step 2
    - Complete full Goodwill login process
    - Check redirect URL format
    
    **❌ Invalid credentials:**
    - Verify API Key and Secret from developer portal
    - Ensure redirect URI is set correctly
    
    **❌ Token exchange fails:**
    - Check request_token format (no extra characters)
    - Try the process again from Step 1
    """)
    
    st.markdown("### **Expected URL Format:**")
    st.code("""
# After successful login, you should see:
http://localhost:8501/?request_token=ABC123XYZ&status=success

# Copy only this part:
ABC123XYZ
    """)

# ==================== DEBUG OAUTH STATE ====================

if st.sidebar.checkbox("🔍 Debug OAuth State"):
    st.sidebar.subheader("OAuth Debug Info")
    debug_oauth = {
        "Current Step": st.session_state.get('oauth_step', 'Not Set'),
        "API Key Stored": bool(st.session_state.get('oauth_api_key', '')),
        "API Secret Stored": bool(st.session_state.get('oauth_api_secret', '')),
        "Login URL Generated": bool(st.session_state.get('oauth_login_url', '')),
        "Currently Logged In": st.session_state.gw_logged_in,
        "Connection Status": bot.data_feed.is_connected
    }
    st.sidebar.json(debug_oauth)
    
    # Reset OAuth state button
    if st.sidebar.button("🔄 Reset OAuth State", use_container_width=True):
        st.session_state.oauth_step = 1
        st.session_state.oauth_api_key = ""
        st.session_state.oauth_api_secret = ""
        st.session_state.oauth_login_url = ""
        st.sidebar.success("OAuth state reset!")
        st.rerun()

# ==================== FOOTER ====================

st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption(f"**FlyingBuddha Scalping Bot** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col_footer2:
    if bot.data_feed.is_connected:
        if bot.mode == "LIVE":
            st.caption("🔴 Live Trading - Connected")
        else:
            st.caption("🟠 Paper Trading - Broker Data")
    else:
        st.caption("🔵 Not Connected to Broker")

with col_footer3:
    st.caption(f"Version 3.0 | OAuth Flow | Adaptive Algorithm")

# Info message when bot is stopped
if not bot.is_running:
    if not st.session_state.gw_logged_in:
        st.info("💡 **Please complete Goodwill OAuth login first, then configure settings and click 'Start Bot' to begin trading.**")
    elif not bot.data_feed.is_connected:
        st.warning("⚠️ **Connection to Goodwill failed. Please try the OAuth login process again.**")
    else:
        st.info("💡 **Bot is ready! Click 'Start Bot' to begin trading with Goodwill live data via OAuth.**")

# ==================== END OF COMPLETE APPLICATION ==================== 
