#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - Complete Production Ready with 8 Strategies
High-frequency scalping with proper Goodwill gwcmodel library integration
- 8 Advanced Scalping Strategies with Dynamic Selection
- Per trade risk: 15%
- Max 4 positions  
- 2-3 min max hold time
- Dynamic/adaptive stop loss
- Risk:Reward 1:2
- Intelligent strategy selection based on market conditions
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
from collections import deque

try:
    import gwcmodel
    GWCMODEL_AVAILABLE = True
except ImportError:
    GWCMODEL_AVAILABLE = False
    st.error("❌ gwcmodel library not installed. Please install: pip install gwcmodel")

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

# ==================== PRODUCTION GOODWILL HANDLER ====================

class ProductionGoodwillHandler:
    """Production-ready Goodwill handler using official gwcmodel library"""
    
    def __init__(self):
        self.gwc = None
        self.access_token = None
        self.api_key = None
        self.api_secret = None
        self.is_connected = False
        self.client_id = None
        self.last_login_time = None
        self.session_token = None
        
    def initialize_gwc(self):
        """Initialize gwcmodel client"""
        if not GWCMODEL_AVAILABLE:
            st.error("❌ gwcmodel library not available")
            return False
        
        try:
            self.gwc = gwcmodel.GWCApi()
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize gwcmodel: {e}")
            return False
    
    def login_with_credentials(self, api_key, user_id, password, totp_code=None):
        """Login using gwcmodel with user credentials"""
        if not self.initialize_gwc():
            return False
        
        try:
            # Store credentials
            self.api_key = api_key
            
            # Login using gwcmodel
            login_params = {
                'api_key': api_key,
                'user_id': user_id,
                'password': password
            }
            
            if totp_code:
                login_params['totp'] = totp_code
            
            # Call gwcmodel login
            response = self.gwc.login(**login_params)
            
            if response and response.get('status') == 'success':
                # Extract tokens from response
                data = response.get('data', {})
                self.access_token = data.get('access_token')
                self.session_token = data.get('session_token')
                self.client_id = data.get('client_id', user_id)
                
                if self.access_token:
                    self.is_connected = True
                    self.last_login_time = datetime.now()
                    
                    # Store in session state
                    st.session_state["gw_logged_in"] = True
                    st.session_state["gw_access_token"] = self.access_token
                    st.session_state["gw_client_id"] = self.client_id
                    st.session_state["gw_api_key"] = api_key
                    
                    st.success("✅ Successfully logged in using gwcmodel!")
                    return True
                else:
                    st.error("❌ No access token received from gwcmodel")
                    return False
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                st.error(f"❌ gwcmodel login failed: {error_msg}")
                return False
                
        except Exception as e:
            st.error(f"❌ gwcmodel login error: {str(e)}")
            return False
    
    def login_with_request_token(self, api_key, request_token, api_secret):
        """Login using gwcmodel with request token"""
        if not self.initialize_gwc():
            return False
        
        try:
            # Store credentials
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Exchange token using gwcmodel
            response = self.gwc.generate_session(
                api_key=api_key,
                request_token=request_token,
                api_secret=api_secret
            )
            
            if response and response.get('status') == 'success':
                # Extract tokens from response
                data = response.get('data', {})
                self.access_token = data.get('access_token')
                self.session_token = data.get('session_token')
                self.client_id = data.get('client_id', 'goodwill_user')
                
                if self.access_token:
                    self.is_connected = True
                    self.last_login_time = datetime.now()
                    
                    # Store in session state
                    st.session_state["gw_logged_in"] = True
                    st.session_state["gw_access_token"] = self.access_token
                    st.session_state["gw_client_id"] = self.client_id
                    st.session_state["gw_api_key"] = api_key
                    
                    st.success("✅ Successfully authenticated using gwcmodel token exchange!")
                    return True
                else:
                    st.error("❌ No access token received from gwcmodel")
                    return False
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                st.error(f"❌ gwcmodel token exchange failed: {error_msg}")
                return False
                
        except Exception as e:
            st.error(f"❌ gwcmodel token exchange error: {str(e)}")
            return False
    
    def get_profile(self):
        """Get user profile using gwcmodel"""
        if not self.is_connected or not self.gwc:
            return None
        
        try:
            response = self.gwc.profile()
            if response and response.get('status') == 'success':
                return response.get('data', {})
            return None
        except Exception as e:
            st.error(f"❌ gwcmodel profile error: {e}")
            return None
    
    def get_live_data(self, symbol):
        """Get live market data using gwcmodel"""
        if not self.is_connected or not self.gwc:
            return None
        
        try:
            # Get quote using gwcmodel
            response = self.gwc.quote(
                exchange='NSE',
                symbol=f'{symbol}-EQ'
            )
            
            if response and response.get('status') == 'success':
                data = response.get('data', {})
                return {
                    'price': float(data.get('ltp', 0) or data.get('last_price', 0)),
                    'volume': int(data.get('volume', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'open': float(data.get('open', 0)),
                    'timestamp': datetime.now()
                }
            return None
            
        except Exception as e:
            print(f"gwcmodel data fetch error for {symbol}: {e}")
            return None
    
    def place_order(self, symbol, action, quantity, price, order_type="MKT"):
        """Place order using gwcmodel"""
        if not self.is_connected or not self.gwc:
            st.error("❌ Not connected to Goodwill via gwcmodel")
            return None
        
        try:
            # Prepare order parameters for gwcmodel
            order_params = {
                'exchange': 'NSE',
                'symbol': f'{symbol}-EQ',
                'transaction_type': action.upper(),
                'quantity': str(quantity),
                'price': str(price) if order_type != "MKT" else '0',
                'product': 'MIS',
                'order_type': order_type,
                'validity': 'DAY'
            }
            
            # Place order using gwcmodel
            response = self.gwc.place_order(**order_params)
            
            if response and response.get('status') == 'success':
                data = response.get('data', {})
                order_id = data.get('order_id') or data.get('oms_order_id')
                st.success(f"🎯 Order Placed via gwcmodel: {action} {quantity} {symbol} @ ₹{price:.2f} | ID: {order_id}")
                return order_id
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                st.error(f"❌ gwcmodel Order Failed: {error_msg}")
                return None
                
        except Exception as e:
            st.error(f"❌ gwcmodel Order Error: {e}")
            return None
    
    def get_positions(self):
        """Get positions using gwcmodel"""
        if not self.is_connected or not self.gwc:
            return []
        
        try:
            response = self.gwc.positions()
            if response and response.get('status') == 'success':
                return response.get('data', [])
            return []
        except Exception as e:
            print(f"gwcmodel positions error: {e}")
            return []
    
    def logout(self):
        """Logout using gwcmodel"""
        if self.gwc:
            try:
                self.gwc.logout()
            except:
                pass
        
        # Clear all connection data
        self.gwc = None
        self.access_token = None
        self.is_connected = False
        self.client_id = None
        self.session_token = None
        
        # Clear session state
        for key in ["gw_logged_in", "gw_access_token", "gw_client_id", "gw_api_key"]:
            if key in st.session_state:
                del st.session_state[key]

# ==================== 8 ADVANCED SCALPING STRATEGIES ====================

class AdvancedScalpingStrategies:
    """8 Advanced Scalping Strategies with Dynamic Selection"""
    
    def __init__(self):
        self.strategies = {
            1: {"name": "Momentum_Breakout", "color": "#FF6B6B"},
            2: {"name": "Mean_Reversion", "color": "#4ECDC4"}, 
            3: {"name": "Volume_Spike", "color": "#45B7D1"},
            4: {"name": "Bollinger_Squeeze", "color": "#96CEB4"},
            5: {"name": "RSI_Divergence", "color": "#FFEAA7"},
            6: {"name": "VWAP_Touch", "color": "#DDA0DD"},
            7: {"name": "Support_Resistance", "color": "#98D8C8"},
            8: {"name": "News_Momentum", "color": "#F7DC6F"}
        }
        
        # Strategy performance tracking
        self.strategy_stats = {
            strategy_id: {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'avg_hold_time': 0.0,
                'success_rate': 0.0,
                'last_used': None,
                'avg_profit': 0.0,
                'avg_loss': 0.0
            } for strategy_id in self.strategies.keys()
        }
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions for strategy selection"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 20:
                return self.get_default_conditions()
            
            closes = data['Close']
            volumes = data['Volume']
            highs = data['High']
            lows = data['Low']
            
            # Technical indicators
            sma_5 = closes.rolling(5).mean()
            sma_20 = closes.rolling(20).mean()
            rsi = self.calculate_rsi(closes, 14)
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(closes, 20)
            vwap = self.calculate_vwap(data)
            
            current_price = closes.iloc[-1]
            
            # Market condition analysis
            conditions = {
                # Trending conditions
                'trending': abs((sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100 > 0.5,
                'trend_strength': abs((sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100,
                'trend_direction': 1 if sma_5.iloc[-1] > sma_20.iloc[-1] else -1,
                
                # Volatility conditions
                'volatile': closes.pct_change().rolling(10).std().iloc[-1] * 100 > 1.5,
                'volatility': closes.pct_change().rolling(10).std().iloc[-1] * 100,
                
                # Volume conditions
                'volume_surge': volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] > 1.8,
                'volume_ratio': volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1],
                
                # Range conditions
                'consolidating': ((highs.tail(10).max() - lows.tail(10).min()) / current_price) * 100 < 1.0,
                'price_range': ((highs.tail(10).max() - lows.tail(10).min()) / current_price) * 100,
                
                # Technical indicators
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
                'rsi_oversold': rsi.iloc[-1] < 30 if len(rsi) > 0 else False,
                'rsi_overbought': rsi.iloc[-1] > 70 if len(rsi) > 0 else False,
                
                # Bollinger Bands
                'bb_squeeze': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] < 0.02,
                'bb_upper_touch': current_price > bb_upper.iloc[-1] * 0.995,
                'bb_lower_touch': current_price < bb_lower.iloc[-1] * 1.005,
                
                # VWAP
                'above_vwap': current_price > vwap.iloc[-1],
                'vwap_distance': abs(current_price - vwap.iloc[-1]) / current_price,
                
                # Support/Resistance
                'near_support': self.check_support_resistance(data, 'support'),
                'near_resistance': self.check_support_resistance(data, 'resistance'),
                
                # News/Event driven
                'news_driven': (volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] > 2.0) and 
                              (closes.pct_change().rolling(5).std().iloc[-1] * 100 > 2.0)
            }
            
            return conditions
            
        except Exception as e:
            print(f"Market condition analysis error for {symbol}: {e}")
            return self.get_default_conditions()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, lower, sma
        except:
            return prices, prices, prices
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap
        except:
            return data['Close']
    
    def check_support_resistance(self, data, level_type='support'):
        """Check if price is near support or resistance"""
        try:
            closes = data['Close']
            current_price = closes.iloc[-1]
            
            # Find recent highs and lows
            recent_data = data.tail(50)
            
            if level_type == 'support':
                levels = recent_data['Low'].rolling(5).min().dropna().unique()
            else:
                levels = recent_data['High'].rolling(5).max().dropna().unique()
            
            # Check if current price is within 0.5% of any level
            for level in levels:
                if abs(current_price - level) / current_price < 0.005:
                    return True
            return False
        except:
            return False
    
    def get_default_conditions(self) -> Dict:
        """Return default market conditions when analysis fails"""
        return {
            'trending': False, 'trend_strength': 0.3, 'trend_direction': 1,
            'volatile': True, 'volatility': 1.0,
            'volume_surge': False, 'volume_ratio': 1.2,
            'consolidating': False, 'price_range': 0.8,
            'rsi': 50, 'rsi_oversold': False, 'rsi_overbought': False,
            'bb_squeeze': False, 'bb_upper_touch': False, 'bb_lower_touch': False,
            'above_vwap': True, 'vwap_distance': 0.002,
            'near_support': False, 'near_resistance': False,
            'news_driven': False
        }
    
    def select_strategy(self, symbol: str, market_conditions: Dict) -> int:
        """Select best strategy based on market conditions and performance"""
        
        strategy_scores = {}
        
        # Strategy 1: Momentum Breakout
        strategy_scores[1] = (
            (50 if market_conditions['trending'] else 10) +
            (30 if market_conditions['volume_surge'] else 5) +
            (20 if market_conditions['volatility'] > 1.0 else 0) +
            self.get_performance_bonus(1)
        )
        
        # Strategy 2: Mean Reversion
        strategy_scores[2] = (
            (60 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 15) +
            (30 if market_conditions['consolidating'] else 5) +
            (20 if not market_conditions['trending'] else 0) +
            self.get_performance_bonus(2)
        )
        
        # Strategy 3: Volume Spike
        strategy_scores[3] = (
            (70 if market_conditions['volume_surge'] else 10) +
            (20 if market_conditions['volatile'] else 5) +
            (10 if market_conditions['news_driven'] else 0) +
            self.get_performance_bonus(3)
        )
        
        # Strategy 4: Bollinger Squeeze
        strategy_scores[4] = (
            (60 if market_conditions['bb_squeeze'] else 10) +
            (25 if market_conditions['consolidating'] else 5) +
            (15 if market_conditions['volatility'] < 1.0 else 0) +
            self.get_performance_bonus(4)
        )
        
        # Strategy 5: RSI Divergence
        strategy_scores[5] = (
            (50 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 20) +
            (30 if market_conditions['trending'] else 10) +
            (20 if 1.0 < market_conditions['volatility'] < 2.5 else 5) +
            self.get_performance_bonus(5)
        )
        
        # Strategy 6: VWAP Touch
        strategy_scores[6] = (
            (50 if market_conditions['vwap_distance'] < 0.003 else 15) +
            (30 if market_conditions['volume_surge'] else 10) +
            (20 if market_conditions['trending'] else 5) +
            self.get_performance_bonus(6)
        )
        
        # Strategy 7: Support/Resistance
        strategy_scores[7] = (
            (60 if market_conditions['near_support'] or market_conditions['near_resistance'] else 15) +
            (25 if market_conditions['consolidating'] else 10) +
            (15 if not market_conditions['trending'] else 5) +
            self.get_performance_bonus(7)
        )
        
        # Strategy 8: News Momentum
        strategy_scores[8] = (
            (80 if market_conditions['news_driven'] else 10) +
            (15 if market_conditions['volatile'] else 5) +
            (5 if market_conditions['volume_surge'] else 0) +
            self.get_performance_bonus(8)
        )
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        return best_strategy, strategy_scores
    
    def get_performance_bonus(self, strategy_id: int) -> float:
        """Calculate performance bonus based on historical success"""
        stats = self.strategy_stats[strategy_id]
        
        if stats['trades'] < 3:
            return 5  # Neutral bonus for new strategies
        
        # Performance bonus (0-20 points)
        success_bonus = (stats['success_rate'] / 100) * 15
        
        # Recent usage penalty for diversity
        if stats['last_used']:
            hours_since = (datetime.now() - stats['last_used']).total_seconds() / 3600
            recency_bonus = min(hours_since * 0.5, 5)
        else:
            recency_bonus = 5
        
        return success_bonus + recency_bonus
    
    def get_strategy_signals(self, strategy_id: int, symbol: str, market_conditions: Dict) -> Dict:
        """Get specific signals for the selected strategy"""
        
        if strategy_id == 1:  # Momentum Breakout
            return self.momentum_breakout_signals(symbol, market_conditions)
        elif strategy_id == 2:  # Mean Reversion
            return self.mean_reversion_signals(symbol, market_conditions)
        elif strategy_id == 3:  # Volume Spike
            return self.volume_spike_signals(symbol, market_conditions)
        elif strategy_id == 4:  # Bollinger Squeeze
            return self.bollinger_squeeze_signals(symbol, market_conditions)
        elif strategy_id == 5:  # RSI Divergence
            return self.rsi_divergence_signals(symbol, market_conditions)
        elif strategy_id == 6:  # VWAP Touch
            return self.vwap_touch_signals(symbol, market_conditions)
        elif strategy_id == 7:  # Support Resistance
            return self.support_resistance_signals(symbol, market_conditions)
        elif strategy_id == 8:  # News Momentum
            return self.news_momentum_signals(symbol, market_conditions)
        
        return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def momentum_breakout_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 1: Momentum Breakout"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 10:
                return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
            current_price = data['Close'].iloc[-1]
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            # Breakout conditions
            breakout_up = current_price > high_20 * 1.001
            breakout_down = current_price < low_20 * 0.999
            
            if breakout_up and conditions['volume_surge'] and conditions['trending']:
                return {
                    'signal': 1,  # Buy
                    'confidence': 0.8,
                    'entry_price': current_price,
                    'stop_pct': 0.008,
                    'reason': 'Upward breakout with volume'
                }
            elif breakout_down and conditions['volume_surge'] and conditions['trend_direction'] < 0:
                return {
                    'signal': -1,  # Sell
                    'confidence': 0.8,
                    'entry_price': current_price,
                    'stop_pct': 0.008,
                    'reason': 'Downward breakout with volume'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def mean_reversion_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 2: Mean Reversion"""
        try:
            current_price = 0  # Get from market data
            rsi = conditions['rsi']
            
            # Mean reversion signals
            if rsi < 25 and conditions['bb_lower_touch']:
                return {
                    'signal': 1,  # Buy oversold
                    'confidence': 0.75,
                    'entry_price': current_price,
                    'stop_pct': 0.006,
                    'reason': 'Oversold mean reversion'
                }
            elif rsi > 75 and conditions['bb_upper_touch']:
                return {
                    'signal': -1,  # Sell overbought
                    'confidence': 0.75,
                    'entry_price': current_price,
                    'stop_pct': 0.006,
                    'reason': 'Overbought mean reversion'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def volume_spike_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 3: Volume Spike"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 5:
                return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].pct_change().iloc[-1] * 100
            
            if conditions['volume_surge'] and abs(price_change) > 0.3:
                signal = 1 if price_change > 0 else -1
                return {
                    'signal': signal,
                    'confidence': 0.85,
                    'entry_price': current_price,
                    'stop_pct': 0.007,
                    'reason': f'Volume spike with {abs(price_change):.2f}% move'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def bollinger_squeeze_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 4: Bollinger Squeeze"""
        try:
            if conditions['bb_squeeze'] and conditions['consolidating']:
                # Wait for breakout from squeeze
                signal = conditions['trend_direction']
                return {
                    'signal': signal,
                    'confidence': 0.7,
                    'entry_price': 0,  # Will be set by caller
                    'stop_pct': 0.005,
                    'reason': 'Bollinger squeeze breakout'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def rsi_divergence_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 5: RSI Divergence"""
        try:
            rsi = conditions['rsi']
            
            # Simple RSI extreme signals (divergence needs more complex analysis)
            if rsi < 30 and conditions['near_support']:
                return {
                    'signal': 1,
                    'confidence': 0.65,
                    'entry_price': 0,
                    'stop_pct': 0.006,
                    'reason': 'RSI oversold at support'
                }
            elif rsi > 70 and conditions['near_resistance']:
                return {
                    'signal': -1,
                    'confidence': 0.65,
                    'entry_price': 0,
                    'stop_pct': 0.006,
                    'reason': 'RSI overbought at resistance'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def vwap_touch_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 6: VWAP Touch"""
        try:
            if conditions['vwap_distance'] < 0.003:  # Very close to VWAP
                # Direction based on trend and volume
                if conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': 1,  # Buy above VWAP with volume
                        'confidence': 0.7,
                        'entry_price': 0,
                        'stop_pct': 0.005,
                        'reason': 'VWAP support with volume'
                    }
                elif not conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': -1,  # Sell below VWAP with volume
                        'confidence': 0.7,
                        'entry_price': 0,
                        'stop_pct': 0.005,
                        'reason': 'VWAP resistance with volume'
                    }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def support_resistance_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 7: Support/Resistance"""
        try:
            if conditions['near_support'] and conditions['rsi'] < 40:
                return {
                    'signal': 1,  # Buy at support
                    'confidence': 0.75,
                    'entry_price': 0,
                    'stop_pct': 0.004,  # Tight stop below support
                    'reason': 'Support level bounce'
                }
            elif conditions['near_resistance'] and conditions['rsi'] > 60:
                return {
                    'signal': -1,  # Sell at resistance
                    'confidence': 0.75,
                    'entry_price': 0,
                    'stop_pct': 0.004,  # Tight stop above resistance
                    'reason': 'Resistance level rejection'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def news_momentum_signals(self, symbol: str, conditions: Dict) -> Dict:
        """Strategy 8: News Momentum"""
        try:
            if conditions['news_driven']:
                # High volume + high volatility = news event
                signal = conditions['trend_direction']
                confidence = min(0.9, conditions['volume_ratio'] * 0.3)
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'entry_price': 0,
                    'stop_pct': 0.01,  # Wider stop for volatile news moves
                    'reason': f'News momentum (Vol: {conditions["volume_ratio"]:.1f}x)'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
            
        except Exception as e:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005}
    
    def update_strategy_performance(self, strategy_id: int, trade_result: Dict):
        """Update strategy performance statistics"""
        stats = self.strategy_stats[strategy_id]
        
        stats['trades'] += 1
        stats['last_used'] = datetime.now()
        
        pnl = trade_result.get('pnl', 0)
        hold_time = trade_result.get('hold_time', 0)
        
        if pnl > 0:
            stats['wins'] += 1
            stats['avg_profit'] = ((stats['avg_profit'] * (stats['wins'] - 1)) + pnl) / stats['wins']
        else:
            loss_count = stats['trades'] - stats['wins']
            if loss_count > 0:
                stats['avg_loss'] = ((stats['avg_loss'] * (loss_count - 1)) + pnl) / loss_count
        
        stats['total_pnl'] += pnl
        stats['success_rate'] = (stats['wins'] / stats['trades']) * 100
        stats['avg_hold_time'] = ((stats['avg_hold_time'] * (stats['trades'] - 1)) + hold_time) / stats['trades']


class AdaptiveScalpingBot:
    """Advanced scalping bot with 8 strategies and gwcmodel integration"""
    
    def __init__(self):
        self.data_feed = ProductionGoodwillHandler()
        self.strategies = AdvancedScalpingStrategies()
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
                    strategy_id INTEGER,
                    strategy_name TEXT,
                    exit_reason TEXT,
                    confidence REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            st.error(f"Database init error: {e}")
    
    def scan_for_opportunities(self):
        """Enhanced market scanning with 8 strategies"""
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Get live data
                if self.data_feed.is_connected:
                    live_data = self.data_feed.get_live_data(symbol)
                else:
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
                
                # Analyze market conditions
                market_conditions = self.strategies.analyze_market_conditions(symbol)
                
                # Select best strategy
                strategy_id, strategy_scores = self.strategies.select_strategy(symbol, market_conditions)
                
                # Get strategy-specific signals
                signals = self.strategies.get_strategy_signals(strategy_id, symbol, market_conditions)
                
                if signals['signal'] != 0 and signals['confidence'] > 0.6:
                    opportunity = {
                        'symbol': symbol,
                        'strategy_id': strategy_id,
                        'strategy_name': self.strategies.strategies[strategy_id]['name'],
                        'signal': signals['signal'],
                        'confidence': signals['confidence'],
                        'price': live_data['price'],
                        'stop_pct': signals['stop_pct'],
                        'reason': signals.get('reason', 'Strategy signal'),
                        'market_conditions': market_conditions,
                        'strategy_scores': strategy_scores
                    }
                    
                    opportunities.append(opportunity)
                    
            except Exception as e:
                print(f"Scanning error for {symbol}: {e}")
                continue
        
        # Sort by confidence and strategy performance
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:3]  # Top 3 opportunities
    
    def calculate_position_size(self, symbol, entry_price, stop_pct):
        """Calculate position size based on risk and strategy stop loss"""
        try:
            risk_amount = self.capital * self.risk_per_trade
            stop_loss_amount = entry_price * stop_pct
            
            if stop_loss_amount > 0:
                quantity = int(risk_amount / stop_loss_amount)
                quantity = max(1, min(quantity, 1000))
                return quantity
            
            return 1
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 1
    
    def execute_entry(self, opportunity):
        """Execute entry with selected strategy"""
        try:
            symbol = opportunity['symbol']
            entry_price = opportunity['price']
            signal = opportunity['signal']
            strategy_id = opportunity['strategy_id']
            strategy_name = opportunity['strategy_name']
            stop_pct = opportunity['stop_pct']
            confidence = opportunity['confidence']
            
            quantity = self.calculate_position_size(symbol, entry_price, stop_pct)
            action = "BUY" if signal > 0 else "SELL"
            
            if action == "BUY":
                stop_loss = entry_price * (1 - stop_pct)
                target = entry_price * (1 + (stop_pct * self.risk_reward_ratio))
            else:
                stop_loss = entry_price * (1 + stop_pct)
                target = entry_price * (1 - (stop_pct * self.risk_reward_ratio))
            
            # Place order based on mode
            if self.mode == "LIVE":
                order_id = self.data_feed.place_order(symbol, action, quantity, entry_price)
            else:
                order_id = f"PAPER_{int(datetime.now().timestamp())}"
                data_source = "gwcmodel Data" if self.data_feed.is_connected else "yfinance Data"
                st.info(f"📝 PAPER TRADE: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name} | Confidence: {confidence:.1%}")
            
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
                    'strategy_id': strategy_id,
                    'strategy_name': strategy_name,
                    'confidence': confidence,
                    'stop_pct': stop_pct,
                    'trailing_stop': stop_loss,
                    'highest_profit': 0,
                    'lowest_profit': 0
                }
                
                st.success(f"🚀 Position Opened: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name}")
                return position_id
            
            return None
            
        except Exception as e:
            st.error(f"Entry execution error: {e}")
            return None
    
    def update_adaptive_stops(self, position_id, current_price):
        """Update trailing stops based on strategy"""
        position = self.positions[position_id]
        
        if position['action'] == 'BUY':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        position['highest_profit'] = max(position['highest_profit'], current_pnl)
        position['lowest_profit'] = min(position['lowest_profit'], current_pnl)
        
        # Strategy-specific trailing logic
        if current_pnl > 0:
            trail_factor = min(0.5, current_pnl / (position['entry_price'] * position['quantity'] * 0.01))
            
            if position['action'] == 'BUY':
                new_stop = current_price * (1 - position['stop_pct'] * (1 - trail_factor))
                position['trailing_stop'] = max(position['trailing_stop'], new_stop)
            else:
                new_stop = current_price * (1 + position['stop_pct'] * (1 - trail_factor))
                position['trailing_stop'] = min(position['trailing_stop'], new_stop)
    
    def check_exit_conditions(self, position_id, current_price):
        """Check exit conditions with strategy-specific logic"""
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
        
        return None
    
    def close_position(self, position_id, exit_price, exit_reason):
        """Close position and update strategy performance"""
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
            'strategy_id': position['strategy_id'],
            'strategy_name': position['strategy_name'],
            'exit_reason': exit_reason,
            'confidence': position['confidence']
        }
        
        self.trades.append(trade_record)
        
        # Update strategy performance
        self.strategies.update_strategy_performance(position['strategy_id'], {
            'pnl': pnl,
            'hold_time': hold_time,
            'exit_reason': exit_reason
        })
        
        # Place exit order
        exit_action = "SELL" if position['action'] == "BUY" else "BUY"
        if self.mode == "LIVE":
            self.data_feed.place_order(position['symbol'], exit_action, position['quantity'], exit_price)
        else:
            data_source = "gwcmodel Data" if self.data_feed.is_connected else "yfinance Data"
            st.info(f"📝 PAPER EXIT: {exit_action} {position['quantity']} {position['symbol']} @ ₹{exit_price:.2f} | Strategy: {position['strategy_name']}")
        
        st.info(f"✅ Position Closed: {position['symbol']} | P&L: ₹{pnl:.2f} | Strategy: {position['strategy_name']} | Reason: {exit_reason}")
    
    def run_trading_cycle(self):
        """Main trading cycle with 8 strategies"""
        try:
            opportunities = self.scan_for_opportunities()
            
            for opp in opportunities:
                if len(self.positions) < self.max_positions:
                    self.execute_entry(opp)
            
            # Check exit conditions for existing positions
            positions_to_close = []
            for pos_id in list(self.positions.keys()):
                position = self.positions[pos_id]
                
                # Get current price
                if self.data_feed.is_connected:
                    live_data = self.data_feed.get_live_data(position['symbol'])
                    current_price = live_data['price'] if live_data else position['entry_price']
                else:
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
        """Calculate enhanced performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_trade': 0, 'max_profit': 0, 'max_loss': 0, 'avg_hold_time': 0,
                'best_strategy': 'N/A', 'strategy_breakdown': {}
            }
        
        pnls = [trade['pnl'] for trade in self.trades]
        hold_times = [trade['hold_time'] for trade in self.trades]
        winning_trades = len([p for p in pnls if p > 0])
        
        # Strategy breakdown
        strategy_breakdown = {}
        for trade in self.trades:
            strategy_name = trade.get('strategy_name', 'Unknown')
            if strategy_name not in strategy_breakdown:
                strategy_breakdown[strategy_name] = {'trades': 0, 'pnl': 0, 'wins': 0}
            
            strategy_breakdown[strategy_name]['trades'] += 1
            strategy_breakdown[strategy_name]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                strategy_breakdown[strategy_name]['wins'] += 1
        
        # Calculate win rates for strategies
        for strategy in strategy_breakdown:
            if strategy_breakdown[strategy]['trades'] > 0:
                strategy_breakdown[strategy]['win_rate'] = (
                    strategy_breakdown[strategy]['wins'] / strategy_breakdown[strategy]['trades']
                ) * 100
        
        # Find best performing strategy
        best_strategy = 'N/A'
        if strategy_breakdown:
            best_strategy = max(strategy_breakdown.keys(), 
                              key=lambda x: strategy_breakdown[x]['pnl'])
        
        return {
            'total_trades': len(self.trades),
            'win_rate': (winning_trades / len(self.trades)) * 100,
            'total_pnl': round(sum(pnls), 2),
            'avg_trade': round(np.mean(pnls), 2),
            'max_profit': round(max(pnls), 2) if pnls else 0,
            'max_loss': round(min(pnls), 2) if pnls else 0,
            'avg_hold_time': round(np.mean(hold_times), 1) if hold_times else 0,
            'best_strategy': best_strategy,
            'strategy_breakdown': strategy_breakdown
        }


# ==================== STREAMLIT UI ====================

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = AdaptiveScalpingBot()

if 'gw_logged_in' not in st.session_state:
    st.session_state.gw_logged_in = False

bot = st.session_state.bot

# Main title
st.title("⚡ FlyingBuddha Scalping Bot - 8 Advanced Strategies")

# Check if gwcmodel is available
if not GWCMODEL_AVAILABLE:
    st.error("❌ **gwcmodel Library Missing**")
    st.code("pip install gwcmodel")
    st.stop()

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.header("⚡ 8-Strategy Scalping Bot")

# Strategy Performance Overview
st.sidebar.subheader("🎯 Strategy Performance")

if bot.strategies.strategy_stats:
    for strategy_id, stats in bot.strategies.strategy_stats.items():
        if stats['trades'] > 0:
            strategy_name = bot.strategies.strategies[strategy_id]['name']
            color = bot.strategies.strategies[strategy_id]['color']
            
            st.sidebar.markdown(f"""
            <div style="background-color: {color}20; padding: 8px; border-radius: 5px; margin: 2px 0;">
                <strong>{strategy_name}</strong><br>
                Trades: {stats['trades']} | Win Rate: {stats['success_rate']:.1f}%<br>
                P&L: ₹{stats['total_pnl']:.2f}
            </div>
            """, unsafe_allow_html=True)

# Trading Mode Selection
st.sidebar.subheader("🔄 Trading Mode")

current_mode_icon = "🔴" if bot.mode == "LIVE" else "🟠"
current_mode_text = f"{current_mode_icon} {bot.mode} Mode"
if bot.data_feed.is_connected:
    current_mode_text += " (gwcmodel Connected)"
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
            st.warning("⚠️ Please login to Goodwill first for Live Mode")
        else:
            st.success("✅ Switched to Live Trading")
        st.rerun()

# ==================== GOODWILL LOGIN SECTION ====================

st.sidebar.subheader("🔐 Goodwill Authentication")

if not st.session_state.gw_logged_in:
    st.sidebar.info("🔗 Connect to Goodwill using gwcmodel library")
    
    login_method = st.sidebar.selectbox(
        "Login Method:",
        ["Credentials", "Request Token"],
        help="Choose your preferred login method"
    )
    
    if login_method == "Credentials":
        st.sidebar.markdown("**📝 Login with Credentials**")
        
        with st.sidebar.form("gw_credentials_login"):
            api_key = st.text_input("API Key", type="password")
            user_id = st.text_input("User ID")
            password = st.text_input("Password", type="password")
            totp_code = st.text_input("TOTP Code (Optional)")
            
            login_submit = st.form_submit_button("🔑 Login")
            
            if login_submit:
                if api_key and user_id and password:
                    with st.spinner("🔄 Logging in via gwcmodel..."):
                        success = bot.data_feed.login_with_credentials(
                            api_key, user_id, password, totp_code if totp_code else None
                        )
                        if success:
                            st.rerun()
                else:
                    st.error("❌ Please fill in all required fields")
    
    else:  # Request Token method
        st.sidebar.markdown("**🎫 Login with Request Token**")
        
        with st.sidebar.form("gw_token_login"):
            api_key = st.text_input("API Key", type="password")
            request_token = st.text_input("Request Token")
            api_secret = st.text_input("API Secret", type="password")
            
            token_submit = st.form_submit_button("🎫 Login with Token")
            
            if token_submit:
                if api_key and request_token and api_secret:
                    with st.spinner("🔄 Exchanging token via gwcmodel..."):
                        success = bot.data_feed.login_with_request_token(
                            api_key, request_token, api_secret
                        )
                        if success:
                            st.rerun()
                else:
                    st.error("❌ Please fill in all required fields")

else:
    st.sidebar.success("✅ Connected to Goodwill via gwcmodel")
    
    if st.sidebar.button("👤 Show Profile"):
        profile = bot.data_feed.get_profile()
        if profile:
            st.sidebar.json(profile)
    
    if st.sidebar.button("🚪 Logout"):
        bot.data_feed.logout()
        st.session_state.gw_logged_in = False
        st.success("✅ Logged out successfully")
        st.rerun()

# ==================== TRADING PARAMETERS ====================

st.sidebar.subheader("⚙️ Trading Parameters")

bot.risk_per_trade = st.sidebar.slider(
    "Risk per Trade (%)", 
    min_value=5, max_value=25, value=15, step=1
) / 100

bot.max_positions = st.sidebar.slider(
    "Max Positions", 
    min_value=1, max_value=8, value=4, step=1
)

bot.max_hold_time = st.sidebar.slider(
    "Max Hold Time (seconds)", 
    min_value=60, max_value=300, value=180, step=30
)

bot.risk_reward_ratio = st.sidebar.slider(
    "Risk:Reward Ratio", 
    min_value=1.5, max_value=3.0, value=2.0, step=0.1
)

bot.capital = st.sidebar.number_input(
    "Trading Capital (₹)", 
    min_value=10000, max_value=10000000, value=100000, step=10000
)

# Symbol selection
st.sidebar.subheader("📊 Market Symbols")
default_symbols = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", 
                  "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS"]

selected_symbols = st.sidebar.multiselect(
    "Select Symbols:",
    options=default_symbols + ["WIPRO", "LT", "TITAN", "MARUTI", "BHARTIARTL"],
    default=default_symbols
)

if selected_symbols:
    bot.symbols = selected_symbols

# ==================== BOT CONTROL BUTTONS ====================

st.sidebar.subheader("🎮 Bot Controls")

if not bot.is_running:
    if st.sidebar.button("🚀 START SCALPING BOT", type="primary"):
        bot.is_running = True
        st.sidebar.success("✅ Bot Started!")
        st.rerun()
else:
    if st.sidebar.button("⏹️ STOP BOT", type="secondary"):
        bot.is_running = False
        st.sidebar.warning("⏸️ Bot Stopped!")
        st.rerun()

if st.sidebar.button("🔄 Run Single Cycle"):
    with st.spinner("🔄 Running trading cycle..."):
        bot.run_trading_cycle()
        st.rerun()

if bot.positions and st.sidebar.button("🛑 EMERGENCY STOP ALL"):
    for pos_id in list(bot.positions.keys()):
        position = bot.positions[pos_id]
        if bot.data_feed.is_connected:
            live_data = bot.data_feed.get_live_data(position['symbol'])
            exit_price = live_data['price'] if live_data else position['entry_price']
        else:
            ticker = yf.Ticker(f"{position['symbol']}.NS")
            hist = ticker.history(period="1d", interval="1m")
            exit_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
        
        bot.close_position(pos_id, exit_price, "EMERGENCY_STOP")
    
    st.sidebar.warning("🛑 All positions closed!")
    st.rerun()

# ==================== MAIN DASHBOARD ====================

# Auto-refresh when bot is running
if bot.is_running:
    bot.run_trading_cycle()
    time.sleep(2)
    st.rerun()

# ==================== STATUS OVERVIEW ====================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    status_color = "🟢" if bot.is_running else "🔴"
    st.metric("Bot Status", f"{status_color} {'Running' if bot.is_running else 'Stopped'}")

with col2:
    connection_status = "🟢 Connected" if bot.data_feed.is_connected else "🔴 Disconnected"
    st.metric("gwcmodel Status", connection_status)

with col3:
    st.metric("Trading Mode", f"{'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}")

with col4:
    st.metric("Active Positions", len(bot.positions))

with col5:
    st.metric("Total P&L", f"₹{bot.pnl:.2f}", delta=f"₹{bot.pnl:.2f}")

# ==================== STRATEGY PERFORMANCE DASHBOARD ====================

st.subheader("🎯 8-Strategy Performance Dashboard")

# Create strategy performance visualization
if any(stats['trades'] > 0 for stats in bot.strategies.strategy_stats.values()):
    
    # Strategy performance metrics
    strategy_perf_data = []
    for strategy_id, stats in bot.strategies.strategy_stats.items():
        if stats['trades'] > 0:
            strategy_name = bot.strategies.strategies[strategy_id]['name']
            strategy_perf_data.append({
                'Strategy': strategy_name,
                'Trades': stats['trades'],
                'Win Rate': f"{stats['success_rate']:.1f}%",
                'Total P&L': f"₹{stats['total_pnl']:.2f}",
                'Avg Profit': f"₹{stats['avg_profit']:.2f}",
                'Avg Loss': f"₹{stats['avg_loss']:.2f}",
                'Avg Hold Time': f"{stats['avg_hold_time']:.1f}s",
                'Last Used': stats['last_used'].strftime("%H:%M:%S") if stats['last_used'] else "Never"
            })
    
    if strategy_perf_data:
        df_strategy_perf = pd.DataFrame(strategy_perf_data)
        st.dataframe(df_strategy_perf, use_container_width=True)
        
        # Strategy performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy P&L chart
            fig_strategy_pnl = go.Figure()
            
            strategy_names = [data['Strategy'] for data in strategy_perf_data]
            strategy_pnls = [float(data['Total P&L'].replace('₹', '')) for data in strategy_perf_data]
            colors = [bot.strategies.strategies[i+1]['color'] for i in range(len(strategy_names))]
            
            fig_strategy_pnl.add_trace(go.Bar(
                x=strategy_names,
                y=strategy_pnls,
                marker_color=colors,
                name="Strategy P&L"
            ))
            
            fig_strategy_pnl.update_layout(
                title="Strategy P&L Performance",
                xaxis_title="Strategy",
                yaxis_title="P&L (₹)",
                height=400
            )
            st.plotly_chart(fig_strategy_pnl, use_container_width=True)
        
        with col2:
            # Strategy win rate chart
            fig_win_rate = go.Figure()
            
            win_rates = [float(data['Win Rate'].replace('%', '')) for data in strategy_perf_data]
            
            fig_win_rate.add_trace(go.Bar(
                x=strategy_names,
                y=win_rates,
                marker_color=colors,
                name="Win Rate"
            ))
            
            fig_win_rate.update_layout(
                title="Strategy Win Rate",
                xaxis_title="Strategy",
                yaxis_title="Win Rate (%)",
                height=400
            )
            st.plotly_chart(fig_win_rate, use_container_width=True)

else:
    st.info("📊 No strategy performance data available yet. Run some trades to see performance metrics.")

# ==================== ENHANCED PERFORMANCE METRICS ====================

st.subheader("📈 Enhanced Performance Metrics")

metrics = bot.get_performance_metrics()

# Main metrics
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric("Total Trades", metrics['total_trades'])
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

with perf_col2:
    st.metric("Total P&L", f"₹{metrics['total_pnl']:.2f}")
    st.metric("Avg Trade", f"₹{metrics['avg_trade']:.2f}")

with perf_col3:
    st.metric("Max Profit", f"₹{metrics['max_profit']:.2f}")
    st.metric("Max Loss", f"₹{metrics['max_loss']:.2f}")

with perf_col4:
    st.metric("Best Strategy", metrics['best_strategy'])
    st.metric("Avg Hold Time", f"{metrics['avg_hold_time']:.1f}s")

# Strategy breakdown
if metrics['strategy_breakdown']:
    st.subheader("📊 Strategy Breakdown")
    
    breakdown_data = []
    for strategy, data in metrics['strategy_breakdown'].items():
        breakdown_data.append({
            'Strategy': strategy,
            'Trades': data['trades'],
            'Wins': data['wins'],
            'Win Rate': f"{data['win_rate']:.1f}%",
            'Total P&L': f"₹{data['pnl']:.2f}"
        })
    
    df_breakdown = pd.DataFrame(breakdown_data)
    st.dataframe(df_breakdown, use_container_width=True)

# ==================== ACTIVE POSITIONS WITH STRATEGY INFO ====================

st.subheader("🎯 Active Positions (Strategy-Enhanced)")

if bot.positions:
    positions_data = []
    
    for pos_id, position in bot.positions.items():
        # Get current price
        if bot.data_feed.is_connected:
            live_data = bot.data_feed.get_live_data(position['symbol'])
            current_price = live_data['price'] if live_data else position['entry_price']
        else:
            try:
                ticker = yf.Ticker(f"{position['symbol']}.NS")
                hist = ticker.history(period="1d", interval="1m")
                current_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
            except:
                current_price = position['entry_price']
        
        # Calculate current P&L
        if position['action'] == 'BUY':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        
        positions_data.append({
            'Symbol': position['symbol'],
            'Strategy': position['strategy_name'],
            'Action': position['action'],
            'Quantity': position['quantity'],
            'Entry Price': f"₹{position['entry_price']:.2f}",
            'Current Price': f"₹{current_price:.2f}",
            'Target': f"₹{position['target']:.2f}",
            'Stop Loss': f"₹{position['trailing_stop']:.2f}",
            'Current P&L': f"₹{current_pnl:.2f}",
            'Confidence': f"{position['confidence']:.1%}",
            'Hold Time': f"{hold_time:.0f}s",
            'Position ID': pos_id
        })
    
    if positions_data:
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True)
        
        # Manual close position
        st.subheader("🔧 Manual Position Management")
        pos_to_close = st.selectbox(
            "Select Position to Close:",
            options=[""] + [f"{pos['Symbol']} ({pos['Strategy']}) - {pos['Action']}" for pos in positions_data]
        )
        
        if pos_to_close and st.button("🔒 Close Selected Position"):
            for pos_data in positions_data:
                if f"{pos_data['Symbol']} ({pos_data['Strategy']}) - {pos_data['Action']}" == pos_to_close:
                    pos_id = pos_data['Position ID']
                    position = bot.positions[pos_id]
                    
                    if bot.data_feed.is_connected:
                        live_data = bot.data_feed.get_live_data(position['symbol'])
                        exit_price = live_data['price'] if live_data else position['entry_price']
                    else:
                        ticker = yf.Ticker(f"{position['symbol']}.NS")
                        hist = ticker.history(period="1d", interval="1m")
                        exit_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
                    
                    bot.close_position(pos_id, exit_price, "MANUAL_CLOSE")
                    st.success(f"✅ Manually closed position: {position['symbol']} ({position['strategy_name']})")
                    st.rerun()
                    break

else:
    st.info("📭 No active positions")

# ==================== ENHANCED MARKET OPPORTUNITIES ====================

st.subheader("🔍 Current Market Opportunities (Multi-Strategy)")

try:
    opportunities = bot.scan_for_opportunities()
    
    if opportunities:
        opp_data = []
        for opp in opportunities:
            strategy_color = bot.strategies.strategies[opp['strategy_id']]['color']
            
            opp_data.append({
                'Symbol': opp['symbol'],
                'Strategy': opp['strategy_name'],
                'Signal': "🟢 LONG" if opp['signal'] > 0 else "🔴 SHORT",
                'Confidence': f"{opp['confidence']:.1%}",
                'Price': f"₹{opp['price']:.2f}",
                'Stop %': f"{opp['stop_pct']:.2%}",
                'Reason': opp['reason'],
                'Strategy Score': f"{opp['strategy_scores'][opp['strategy_id']]:.1f}"
            })
        
        df_opportunities = pd.DataFrame(opp_data)
        st.dataframe(df_opportunities, use_container_width=True)
        
        # Show market conditions for top opportunity
        if opportunities:
            st.subheader("📊 Market Conditions Analysis (Top Opportunity)")
            top_opp = opportunities[0]
            conditions = top_opp['market_conditions']
            
            cond_col1, cond_col2, cond_col3 = st.columns(3)
            
            with cond_col1:
                st.markdown("**Trend Conditions:**")
                st.write(f"Trending: {'✅' if conditions['trending'] else '❌'}")
                st.write(f"Trend Strength: {conditions['trend_strength']:.2f}%")
                st.write(f"Direction: {'📈' if conditions['trend_direction'] > 0 else '📉'}")
            
            with cond_col2:
                st.markdown("**Volume & Volatility:**")
                st.write(f"Volume Surge: {'✅' if conditions['volume_surge'] else '❌'}")
                st.write(f"Volume Ratio: {conditions['volume_ratio']:.2f}x")
                st.write(f"Volatility: {conditions['volatility']:.2f}%")
            
            with cond_col3:
                st.markdown("**Technical Indicators:**")
                st.write(f"RSI: {conditions['rsi']:.1f}")
                st.write(f"Near Support: {'✅' if conditions['near_support'] else '❌'}")
                st.write(f"Near Resistance: {'✅' if conditions['near_resistance'] else '❌'}")
            
            # Strategy scores for top opportunity
            st.subheader("🎯 Strategy Scores Breakdown")
            scores = top_opp['strategy_scores']
            
            fig_scores = go.Figure()
            
            strategy_names = [bot.strategies.strategies[i]['name'] for i in range(1, 9)]
            strategy_scores = [scores[i] for i in range(1, 9)]
            colors = [bot.strategies.strategies[i]['color'] for i in range(1, 9)]
            
            fig_scores.add_trace(go.Bar(
                x=strategy_names,
                y=strategy_scores,
                marker_color=colors,
                name="Strategy Scores"
            ))
            
            fig_scores.update_layout(
                title=f"Strategy Selection Scores for {top_opp['symbol']}",
                xaxis_title="Strategy",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        
        # Manual entry
        st.subheader("🎯 Manual Entry")
        manual_symbol = st.selectbox(
            "Select Opportunity:",
            options=[""] + [f"{opp['symbol']} - {opp['strategy_name']} ({opp['confidence']:.1%})" for opp in opportunities]
        )
        
        if manual_symbol and st.button("🚀 Execute Manual Entry"):
            for opp in opportunities:
                if f"{opp['symbol']} - {opp['strategy_name']} ({opp['confidence']:.1%})" == manual_symbol:
                    with st.spinner("🔄 Executing manual entry..."):
                        bot.execute_entry(opp)
                        st.rerun()
                    break
    
    else:
        st.info("🔍 No opportunities found with current strategy filters")
        
except Exception as e:
    st.error(f"❌ Error scanning opportunities: {e}")

# ==================== ENHANCED TRADE HISTORY ====================

st.subheader("📋 Strategy-Enhanced Trade History")

if bot.trades:
    recent_trades = bot.trades[-15:]  # Show last 15 trades
    df_trades = pd.DataFrame(recent_trades)
    
    # Add strategy colors
    def style_strategy(val):
        for strategy_id, strategy_info in bot.strategies.strategies.items():
            if strategy_info['name'] == val:
                return f'background-color: {strategy_info["color"]}30'
        return ''
    
    # Display trades with strategy styling
    styled_df = df_trades.style.applymap(
        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
        subset=['pnl']
    ).applymap(style_strategy, subset=['strategy_name'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced P&L chart with strategy breakdown
    if len(bot.trades) > 1:
        st.subheader("📊 Strategy Performance Over Time")
        
        # Create strategy-wise cumulative P&L
        strategy_cumulative = {}
        trade_numbers = []
        
        for i, trade in enumerate(bot.trades):
            strategy_name = trade.get('strategy_name', 'Unknown')
            if strategy_name not in strategy_cumulative:
                strategy_cumulative[strategy_name] = []
            
            # Calculate cumulative for this strategy
            prev_cumulative = strategy_cumulative[strategy_name][-1] if strategy_cumulative[strategy_name] else 0
            new_cumulative = prev_cumulative + trade['pnl']
            strategy_cumulative[strategy_name].append(new_cumulative)
            
            # Pad other strategies with their last value
            for other_strategy in strategy_cumulative:
                if other_strategy != strategy_name:
                    if strategy_cumulative[other_strategy]:
                        strategy_cumulative[other_strategy].append(strategy_cumulative[other_strategy][-1])
                    else:
                        strategy_cumulative[other_strategy].append(0)
            
            trade_numbers.append(i + 1)
        
        # Create cumulative chart
        fig = go.Figure()
        
        for strategy_name, cumulative_data in strategy_cumulative.items():
            # Find strategy color
            strategy_color = '#808080'  # Default gray
            for strategy_id, strategy_info in bot.strategies.strategies.items():
                if strategy_info['name'] == strategy_name:
                    strategy_color = strategy_info['color']
                    break
            
            fig.add_trace(go.Scatter(
                x=trade_numbers[:len(cumulative_data)],
                y=cumulative_data,
                mode='lines+markers',
                name=strategy_name,
                line=dict(color=strategy_color, width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Cumulative P&L by Strategy",
            xaxis_title="Trade Number",
            yaxis_title="Cumulative P&L (₹)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📭 No trades executed yet")

# ==================== LIVE MARKET DATA ====================

st.subheader("📊 Live Market Data")

if bot.data_feed.is_connected:
    st.success("✅ Using live gwcmodel data feed")
else:
    st.warning("⚠️ Using fallback yfinance data (slower updates)")

data_cols = st.columns(3)

for i, symbol in enumerate(bot.symbols[:9]):  # Show first 9 symbols
    with data_cols[i % 3]:
        try:
            if bot.data_feed.is_connected:
                live_data = bot.data_feed.get_live_data(symbol)
                if live_data:
                    st.metric(
                        symbol,
                        f"₹{live_data['price']:.2f}",
                        help=f"Volume: {live_data['volume']:,}"
                    )
                else:
                    st.metric(symbol, "No data")
            else:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="1d", interval="1m")
                if len(hist) > 0:
                    current_price = hist.iloc[-1]['Close']
                    st.metric(symbol, f"₹{current_price:.2f}")
                else:
                    st.metric(symbol, "No data")
        except:
            st.metric(symbol, "Error")

# ==================== SYSTEM STATUS ====================

st.subheader("⚙️ System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.info(f"**gwcmodel Status:** {'✅ Available' if GWCMODEL_AVAILABLE else '❌ Missing'}")
    st.info(f"**Connection:** {'✅ Connected' if bot.data_feed.is_connected else '❌ Disconnected'}")

with status_col2:
    st.info(f"**Trading Mode:** {bot.mode}")
    st.info(f"**Bot Status:** {'🟢 Running' if bot.is_running else '🔴 Stopped'}")

with status_col3:
    st.info(f"**Positions:** {len(bot.positions)}/{bot.max_positions}")
    st.info(f"**Active Strategies:** {len([s for s in bot.strategies.strategy_stats.values() if s['trades'] > 0])}/8")

# ==================== STRATEGY CONFIGURATION ====================

with st.expander("🎛️ Strategy Configuration", expanded=False):
    st.subheader("🔧 Individual Strategy Settings")
    
    st.markdown("""
    **Strategy Selection Logic:**
    - Market conditions are analyzed for each symbol
    - Each strategy gets scored based on current conditions
    - Best strategy is selected dynamically
    - Performance history influences future selection
    """)
    
    # Show strategy details
    for strategy_id, strategy_info in bot.strategies.strategies.items():
        with st.container():
            st.markdown(f"""
            <div style="background-color: {strategy_info['color']}20; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <h4>{strategy_id}. {strategy_info['name']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Strategy description
            descriptions = {
                1: "**Momentum Breakout**: Trades breakouts from 20-period highs/lows with volume confirmation",
                2: "**Mean Reversion**: RSI oversold/overbought signals with Bollinger Band touches",
                3: "**Volume Spike**: High volume moves (>1.8x average) with price momentum",
                4: "**Bollinger Squeeze**: Low volatility periods before breakouts",
                5: "**RSI Divergence**: RSI extreme levels at support/resistance zones",
                6: "**VWAP Touch**: Price reactions near Volume Weighted Average Price",
                7: "**Support/Resistance**: Price bounces at identified S/R levels",
                8: "**News Momentum**: High volume + volatility indicating news-driven moves"
            }
            
            st.markdown(descriptions.get(strategy_id, "Strategy description"))
            
            # Performance stats
            stats = bot.strategies.strategy_stats[strategy_id]
            if stats['trades'] > 0:
                st.markdown(f"""
                **Performance**: {stats['trades']} trades | {stats['success_rate']:.1f}% win rate | ₹{stats['total_pnl']:.2f} P&L
                """)
            else:
                st.markdown("**Performance**: No trades yet")

# ==================== FOOTER INFORMATION ====================

st.markdown("---")

st.markdown("""
### 🚀 FlyingBuddha 8-Strategy Scalping Bot

**8 Advanced Strategies:**
1. 🎯 **Momentum Breakout** - High/low breakouts with volume
2. 🔄 **Mean Reversion** - RSI + Bollinger oversold/overbought  
3. 📊 **Volume Spike** - High volume momentum moves
4. 🗜️ **Bollinger Squeeze** - Low volatility breakouts
5. 📈 **RSI Divergence** - RSI extremes at S/R levels
6. ⚖️ **VWAP Touch** - Price reactions at VWAP
7. 🏗️ **Support/Resistance** - S/R level bounces
8. 📰 **News Momentum** - High volume + volatility events

**Smart Selection**: Strategies are automatically selected based on real-time market conditions and historical performance.

**Risk Management**: 15% per trade | Max 4 positions | 3-min hold | 1:2 R:R | Adaptive stops

⚠️ **Disclaimer**: High-risk trading system. Use responsibly and never risk more than you can afford to lose.
""")

# ==================== FINAL AUTO-REFRESH ====================

if bot.is_running:
    st.markdown(f"🔄 **Bot Running - Auto-refresh | Last Update: {datetime.now().strftime('%H:%M:%S')}**")
    time.sleep(1)

# ==================== END OF APPLICATION ====================

st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    ⚡ FlyingBuddha 8-Strategy Scalping Bot v2.0 | Built with Streamlit & gwcmodel<br>
    <strong>🎯 8 Strategies | 🤖 AI Selection | ⚡ Real-time Execution</strong>
</div>
""", unsafe_allow_html=True)
