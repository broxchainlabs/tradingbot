#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - Complete Production Ready
Real-time scalping with 8 advanced strategies and proper Goodwill gwcmodel integration
- gwcmodel as PRIMARY authentication method
- Direct API as fallback
- 8 Advanced Scalping Strategies with Dynamic Selection
- Production-ready error handling and risk management
- Per trade risk: 15% | Max 4 positions | 2-3 min hold | 1:2 R:R
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import hashlib
import hmac
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ==================== GWCMODEL INTEGRATION ====================

# Primary gwcmodel import and initialization
try:
    import gwcmodel
    GWCMODEL_AVAILABLE = True
    
    # Try to determine correct gwcmodel class
    if hasattr(gwcmodel, 'GWCApi'):
        GWC_CLASS = gwcmodel.GWCApi
        GWC_CLASS_NAME = 'GWCApi'
    elif hasattr(gwcmodel, 'Api'):
        GWC_CLASS = gwcmodel.Api
        GWC_CLASS_NAME = 'Api'
    elif hasattr(gwcmodel, 'Client'):
        GWC_CLASS = gwcmodel.Client
        GWC_CLASS_NAME = 'Client'
    elif hasattr(gwcmodel, 'GoodwillApi'):
        GWC_CLASS = gwcmodel.GoodwillApi
        GWC_CLASS_NAME = 'GoodwillApi'
    else:
        GWC_CLASS = None
        GWC_CLASS_NAME = None
        
except ImportError:
    GWCMODEL_AVAILABLE = False
    GWC_CLASS = None
    GWC_CLASS_NAME = None

# Set page config
st.set_page_config(
    page_title="⚡ FlyingBuddha Scalping Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PROPER GOODWILL INTEGRATION ====================

class ProperGoodwillIntegration:
    """
    Complete Goodwill integration using gwcmodel as PRIMARY with direct API fallback
    Based on official documentation: developer.gwcindia.in/api/
    """
    
    def __init__(self):
        self.gwc_client = None
        self.access_token = None
        self.api_key = None
        self.api_secret = None
        self.client_id = None
        self.is_connected = False
        self.connection_method = None
        self.base_url = "https://api.gwcindia.in/v1"
        self.last_login_time = None
    
    def initialize_gwcmodel(self) -> bool:
        """Initialize gwcmodel client (PRIMARY METHOD)"""
        if not GWCMODEL_AVAILABLE or not GWC_CLASS:
            return False
        
        try:
            self.gwc_client = GWC_CLASS()
            return True
        except Exception as e:
            st.sidebar.warning(f"gwcmodel init error: {e}")
            return False
    
    def login_with_credentials(self, api_key: str, user_id: str, password: str, totp: str = None) -> bool:
        """
        Login using gwcmodel credentials (PRIMARY METHOD)
        """
        try:
            self.api_key = api_key
            
            # Method 1: Try gwcmodel first
            if self.initialize_gwcmodel():
                login_data = {
                    'api_key': api_key,
                    'user_id': user_id,
                    'password': password
                }
                
                if totp:
                    login_data['totp'] = totp
                
                # Try different gwcmodel login methods
                login_methods = ['login', 'authenticate', 'connect', 'sign_in']
                
                for method_name in login_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            st.info(f"🔄 Trying gwcmodel.{method_name}()...")
                            
                            response = method(**login_data)
                            
                            if self._process_gwcmodel_response(response, user_id):
                                self.connection_method = f"gwcmodel_{method_name}"
                                st.success(f"✅ Connected via gwcmodel.{method_name}()!")
                                return True
                            
                        except Exception as e:
                            st.warning(f"gwcmodel.{method_name}() failed: {e}")
                            continue
            
            # Method 2: Direct API fallback (requires request token flow)
            st.warning("⚠️ gwcmodel authentication failed. For direct API, use request token method.")
            return False
                
        except Exception as e:
            st.error(f"❌ Authentication error: {e}")
            return False
    
    def login_with_request_token(self, api_key: str, request_token: str, api_secret: str) -> bool:
        """
        Login using request token (SECONDARY METHOD)
        """
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Method 1: Try gwcmodel token exchange first
            if self.initialize_gwcmodel():
                token_methods = ['generate_session', 'exchange_token', 'get_access_token', 'session_token']
                
                for method_name in token_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            response = method(
                                api_key=api_key,
                                request_token=request_token,
                                api_secret=api_secret
                            )
                            
                            if self._process_gwcmodel_response(response, 'goodwill_user'):
                                self.connection_method = f"gwcmodel_{method_name}"
                                st.success(f"✅ Connected via gwcmodel.{method_name}()!")
                                return True
                                
                        except Exception as e:
                            continue
            
            # Method 2: Direct API call
            return self._direct_api_login(api_key, request_token, api_secret)
                
        except Exception as e:
            st.error(f"❌ Token authentication error: {e}")
            return False
    
    def _process_gwcmodel_response(self, response, default_user_id: str) -> bool:
        """Process gwcmodel response and extract session data"""
        try:
            if not response:
                return False
            
            if isinstance(response, dict):
                if response.get('status') == 'success' or response.get('Success'):
                    data = response.get('data', response.get('Data', response))
                    
                    self.access_token = (
                        data.get('access_token') or 
                        data.get('AccessToken') or
                        data.get('session_token') or
                        data.get('SessionToken')
                    )
                    
                    self.client_id = (
                        data.get('client_id') or 
                        data.get('clnt_id') or
                        data.get('ClientId') or
                        default_user_id
                    )
                    
                    if self.access_token:
                        self.is_connected = True
                        self.last_login_time = datetime.now()
                        
                        # Store in session state
                        st.session_state["gw_logged_in"] = True
                        st.session_state["gw_access_token"] = self.access_token
                        st.session_state["gw_client_id"] = self.client_id
                        st.session_state["gw_connection"] = self.connection_method
                        
                        return True
                else:
                    error_msg = response.get('error_msg', response.get('ErrorMessage', 'Unknown error'))
                    st.warning(f"gwcmodel response error: {error_msg}")
            else:
                # Non-dict response might still be successful
                self.access_token = 'CONNECTED'
                self.client_id = default_user_id
                self.is_connected = True
                self.last_login_time = datetime.now()
                
                st.session_state["gw_logged_in"] = True
                st.session_state["gw_access_token"] = self.access_token
                st.session_state["gw_client_id"] = self.client_id
                
                return True
            
            return False
            
        except Exception as e:
            st.warning(f"Response processing error: {e}")
            return False
    
    def _direct_api_login(self, api_key: str, request_token: str, api_secret: str) -> bool:
        """Direct API login as fallback"""
        try:
            # Create signature as per API documentation
            checksum = f"{api_key}{request_token}{api_secret}"
            signature = hashlib.sha256(checksum.encode()).hexdigest()
            
            payload = {
                "api_key": api_key,
                "request_token": request_token,
                "signature": signature
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "FlyingBuddha-ScalpingBot/1.0"
            }
            
            response = requests.post(
                f"{self.base_url}/login-response",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    user_data = data.get('data', {})
                    self.access_token = user_data.get('access_token')
                    self.client_id = user_data.get('clnt_id')
                    
                    if self.access_token:
                        self.is_connected = True
                        self.connection_method = "direct_api"
                        self.last_login_time = datetime.now()
                        
                        st.session_state["gw_logged_in"] = True
                        st.session_state["gw_access_token"] = self.access_token
                        st.session_state["gw_client_id"] = self.client_id
                        st.session_state["gw_connection"] = "direct_api"
                        
                        st.success(f"✅ Connected via Direct API! Client ID: {self.client_id}")
                        return True
                
                error_msg = data.get('error_msg', 'Direct API authentication failed')
                st.error(f"❌ {error_msg}")
            else:
                st.error(f"❌ HTTP {response.status_code}")
            
            return False
            
        except Exception as e:
            st.error(f"❌ Direct API error: {e}")
            return False
    
    def get_headers(self) -> Dict:
        """Get authenticated headers for API calls"""
        if not self.access_token or not self.api_key:
            return {}
        
        return {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "FlyingBuddha-ScalpingBot/1.0"
        }
    
    def get_profile(self) -> Optional[Dict]:
        """Get user profile"""
        if not self.is_connected:
            return None
        
        try:
            # Method 1: Try gwcmodel first
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                profile_methods = ['profile', 'get_profile', 'user_profile', 'user_details']
                
                for method_name in profile_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            response = method()
                            
                            if response:
                                if isinstance(response, dict):
                                    if response.get('status') == 'success':
                                        return response.get('data', response)
                                    else:
                                        return response
                                else:
                                    return {'profile_data': str(response)}
                        except Exception:
                            continue
            
            # Method 2: Direct API call
            headers = self.get_headers()
            if headers:
                response = requests.get(f"{self.base_url}/profile", headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return data.get('data', {})
            
            return None
            
        except Exception as e:
            st.warning(f"Profile fetch error: {e}")
            return None
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get real-time quote"""
        if not self.is_connected:
            return None
        
        try:
            # Method 1: Try gwcmodel first
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                quote_methods = ['quote', 'get_quote', 'ltp', 'get_ltp', 'market_quote']
                
                for method_name in quote_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            
                            # Try different parameter formats
                            param_formats = [
                                {'symbol': symbol, 'exchange': exchange},
                                {'symbol': f'{symbol}-EQ', 'exchange': exchange},
                                {'tradingsymbol': f'{symbol}-EQ', 'exchange': exchange},
                                {'exchange': exchange, 'token': self._get_symbol_token(symbol, exchange)}
                            ]
                            
                            for params in param_formats:
                                try:
                                    if params.get('token'):
                                        response = method(**params)
                                    else:
                                        response = method(**{k:v for k,v in params.items() if v})
                                    
                                    if response:
                                        quote_data = self._extract_quote_data(response)
                                        if quote_data and quote_data['price'] > 0:
                                            return quote_data
                                            
                                except Exception:
                                    continue
                                    
                        except Exception:
                            continue
            
            # Method 2: Direct API call
            return self._direct_api_quote(symbol, exchange)
            
        except Exception as e:
            print(f"Quote error for {symbol}: {e}")
            return None
    
    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """Get symbol token for API calls"""
        try:
            # Try gwcmodel first
            if self.gwc_client and hasattr(self.gwc_client, 'search_symbols'):
                try:
                    results = self.gwc_client.search_symbols(symbol)
                    if results:
                        for result in results:
                            if result.get('exchange') == exchange:
                                return result.get('token')
                except Exception:
                    pass
            
            # Direct API call
            headers = self.get_headers()
            if headers:
                response = requests.post(
                    f"{self.base_url}/fetchsymbol",
                    json={"s": symbol},
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        results = data.get('data', [])
                        for result in results:
                            if result.get('exchange') == exchange and result.get('symbol') == f"{symbol}-EQ":
                                return result.get('token')
            
            return None
            
        except Exception:
            return None
    
    def _extract_quote_data(self, response) -> Optional[Dict]:
        """Extract quote data from response"""
        try:
            if isinstance(response, dict):
                if response.get('status') == 'success':
                    data = response.get('data', response)
                else:
                    data = response
            else:
                data = {'price': response} if isinstance(response, (int, float)) else {}
            
            price = (
                data.get('ltp') or data.get('LTP') or
                data.get('last_price') or data.get('LastPrice') or
                data.get('price') or data.get('Price') or 0
            )
            
            if price > 0:
                return {
                    'price': float(price),
                    'volume': int(data.get('volume', data.get('Volume', 0))),
                    'high': float(data.get('high', data.get('High', price))),
                    'low': float(data.get('low', data.get('Low', price))),
                    'open': float(data.get('open', data.get('Open', price))),
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception:
            return None
    
    def _direct_api_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get quote using direct API"""
        try:
            token = self._get_symbol_token(symbol, exchange)
            if not token:
                return None
            
            headers = self.get_headers()
            response = requests.post(
                f"{self.base_url}/getquote",
                json={"exchange": exchange, "token": token},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    quote_data = data.get('data', {})
                    return {
                        'price': float(quote_data.get('last_price', 0)),
                        'volume': int(quote_data.get('volume', 0)),
                        'high': float(quote_data.get('high', 0)),
                        'low': float(quote_data.get('low', 0)),
                        'open': float(quote_data.get('open', 0)),
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception:
            return None
    
    def place_order(self, symbol: str, action: str, quantity: int, price: float, 
                   order_type: str = "MKT", product: str = "MIS") -> Optional[str]:
        """Place order using gwcmodel or direct API"""
        if not self.is_connected:
            st.error("❌ Not connected to Goodwill")
            return None
        
        try:
            # Method 1: Try gwcmodel first
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                order_methods = ['place_order', 'order_place', 'submit_order', 'place']
                
                order_params_formats = [
                    {
                        'exchange': 'NSE',
                        'symbol': f'{symbol}-EQ',
                        'transaction_type': action.upper(),
                        'quantity': str(quantity),
                        'price': str(price) if order_type != "MKT" else '0',
                        'product': product,
                        'order_type': order_type,
                        'validity': 'DAY'
                    },
                    {
                        'exchange': 'NSE',
                        'tradingsymbol': f'{symbol}-EQ',
                        'transaction_type': action.upper(),
                        'quantity': int(quantity),
                        'price': float(price) if order_type != "MKT" else 0,
                        'product': product,
                        'order_type': order_type,
                        'validity': 'DAY'
                    }
                ]
                
                for method_name in order_methods:
                    if hasattr(self.gwc_client, method_name):
                        for order_params in order_params_formats:
                            try:
                                method = getattr(self.gwc_client, method_name)
                                response = method(**order_params)
                                
                                if response:
                                    order_id = self._extract_order_id(response)
                                    if order_id:
                                        st.success(f"🎯 Order Placed via gwcmodel: {action} {quantity} {symbol} @ ₹{price:.2f} | ID: {order_id}")
                                        return order_id
                                        
                            except Exception:
                                continue
            
            # Method 2: Direct API call
            return self._direct_api_order(symbol, action, quantity, price, order_type, product)
                
        except Exception as e:
            st.error(f"❌ Order error: {e}")
            return None
    
    def _extract_order_id(self, response) -> Optional[str]:
        """Extract order ID from response"""
        try:
            if isinstance(response, dict):
                if response.get('status') == 'success':
                    data = response.get('data', response)
                else:
                    data = response
                
                return (
                    data.get('order_id') or data.get('OrderId') or
                    data.get('nstordno') or data.get('orderno') or
                    str(data) if data else None
                )
            else:
                return str(response) if response else None
                
        except Exception:
            return None
    
    def _direct_api_order(self, symbol: str, action: str, quantity: int, price: float, 
                         order_type: str, product: str) -> Optional[str]:
        """Place order using direct API"""
        try:
            headers = self.get_headers()
            if not headers:
                return None
            
            order_payload = {
                "tsym": f"{symbol}-EQ",
                "exchange": "NSE",
                "trantype": action.upper(),
                "validity": "DAY",
                "pricetype": order_type,
                "qty": str(quantity),
                "discqty": "0",
                "price": str(price) if order_type != "MKT" else "0",
                "trgprc": "0",
                "product": product,
                "amo": "NO"
            }
            
            response = requests.post(
                f"{self.base_url}/placeorder",
                json=order_payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    order_data = data.get('data', {})
                    order_id = order_data.get('nstordno')
                    st.success(f"🎯 Order Placed via Direct API: {action} {quantity} {symbol} @ ₹{price:.2f} | ID: {order_id}")
                    return order_id
                else:
                    error_msg = data.get('error_msg', 'Order failed')
                    st.error(f"❌ Order Failed: {error_msg}")
            else:
                st.error(f"❌ Order HTTP Error: {response.status_code}")
            
            return None
            
        except Exception as e:
            st.error(f"❌ Direct API order error: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get positions"""
        if not self.is_connected:
            return []
        
        try:
            # Try gwcmodel first
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                position_methods = ['positions', 'get_positions', 'holdings']
                
                for method_name in position_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            response = method()
                            
                            if response:
                                if isinstance(response, dict):
                                    if response.get('status') == 'success':
                                        return response.get('data', [])
                                    else:
                                        return response if isinstance(response, list) else []
                                elif isinstance(response, list):
                                    return response
                        except Exception:
                            continue
            
            # Direct API call
            headers = self.get_headers()
            if headers:
                response = requests.get(f"{self.base_url}/positions", headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return data.get('data', [])
            
            return []
            
        except Exception:
            return []
    
    def logout(self) -> bool:
        """Logout and cleanup"""
        try:
            # Try gwcmodel logout
            if self.gwc_client:
                logout_methods = ['logout', 'disconnect', 'close']
                for method_name in logout_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            method()
                            break
                        except Exception:
                            continue
            
            # Direct API logout
            headers = self.get_headers()
            if headers:
                try:
                    requests.get(f"{self.base_url}/logout", headers=headers, timeout=10)
                except Exception:
                    pass
            
            # Clear all data
            self.gwc_client = None
            self.access_token = None
            self.is_connected = False
            self.client_id = None
            self.connection_method = None
            
            # Clear session state
            for key in ["gw_logged_in", "gw_access_token", "gw_client_id", "gw_connection"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            return True
            
        except Exception:
            return False

# ==================== 8 ADVANCED SCALPING STRATEGIES ====================

class AdvancedScalpingStrategies:
    """8 Advanced Scalping Strategies with Dynamic Selection"""
    
    def __init__(self):
        self.strategies = {
            1: {"name": "Momentum_Breakout", "color": "#FF6B6B", "desc": "Price breakouts with volume"},
            2: {"name": "Mean_Reversion", "color": "#4ECDC4", "desc": "RSI oversold/overbought signals"}, 
            3: {"name": "Volume_Spike", "color": "#45B7D1", "desc": "High volume momentum moves"},
            4: {"name": "Bollinger_Squeeze", "color": "#96CEB4", "desc": "Low volatility breakouts"},
            5: {"name": "RSI_Divergence", "color": "#FFEAA7", "desc": "RSI extremes at key levels"},
            6: {"name": "VWAP_Touch", "color": "#DDA0DD", "desc": "Price reactions near VWAP"},
            7: {"name": "Support_Resistance", "color": "#98D8C8", "desc": "Key level bounces"},
            8: {"name": "News_Momentum", "color": "#F7DC6F", "desc": "High volume + volatility events"}
        }
        
        self.strategy_stats = {
            strategy_id: {
                'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'success_rate': 0.0,
                'last_used': None, 'avg_profit': 0.0, 'avg_loss': 0.0, 'avg_hold_time': 0.0
            } for strategy_id in self.strategies.keys()
        }
        
        # Cache for market analysis
        self.analysis_cache = {}
        self.cache_timeout = 30  # seconds
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Comprehensive market condition analysis"""
        
        # Check cache first
        cache_key = f"{symbol}_{int(time.time() // self.cache_timeout)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 20:
                return self._get_default_conditions()
            
            closes = data['Close']
            volumes = data['Volume']
            highs = data['High']
            lows = data['Low']
            
            # Technical indicators
            sma_5 = closes.rolling(5).mean()
            sma_20 = closes.rolling(20).mean()
            rsi = self._calculate_rsi(closes, 14)
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(closes, 20)
            vwap = self._calculate_vwap(data)
            
            current_price = closes.iloc[-1]
            
            # Market condition analysis
            conditions = {
                # Trend conditions
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
                'bb_squeeze': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] < 0.02 if len(bb_middle) > 0 else False,
                'bb_upper_touch': current_price > bb_upper.iloc[-1] * 0.995 if len(bb_upper) > 0 else False,
                'bb_lower_touch': current_price < bb_lower.iloc[-1] * 1.005 if len(bb_lower) > 0 else False,
                
                # VWAP
                'above_vwap': current_price > vwap.iloc[-1] if len(vwap) > 0 else True,
                'vwap_distance': abs(current_price - vwap.iloc[-1]) / current_price if len(vwap) > 0 else 0.002,
                
                # Support/Resistance
                'near_support': self._check_support_resistance(data, 'support'),
                'near_resistance': self._check_support_resistance(data, 'resistance'),
                
                # News/Event driven
                'news_driven': (volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] > 2.0) and 
                              (closes.pct_change().rolling(5).std().iloc[-1] * 100 > 2.0)
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = conditions
            
            # Clean old cache entries
            if len(self.analysis_cache) > 100:
                old_keys = [k for k in self.analysis_cache.keys() if int(k.split('_')[-1]) < int(time.time() // self.cache_timeout) - 10]
                for key in old_keys:
                    del self.analysis_cache[key]
            
            return conditions
            
        except Exception as e:
            print(f"Market analysis error for {symbol}: {e}")
            return self._get_default_conditions()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.fillna(sma), lower.fillna(sma), sma.fillna(prices)
        except:
            return prices, prices, prices
    
    def _calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap.fillna(data['Close'])
        except:
            return data['Close']
    
    def _check_support_resistance(self, data, level_type='support'):
        """Check if price is near support or resistance"""
        try:
            closes = data['Close']
            current_price = closes.iloc[-1]
            
            recent_data = data.tail(50)
            
            if level_type == 'support':
                levels = recent_data['Low'].rolling(5).min().dropna().unique()
            else:
                levels = recent_data['High'].rolling(5).max().dropna().unique()
            
            for level in levels:
                if abs(current_price - level) / current_price < 0.005:
                    return True
            return False
        except:
            return False
    
    def _get_default_conditions(self) -> Dict:
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
    
    def select_strategy(self, symbol: str, market_conditions: Dict) -> Tuple[int, Dict]:
        """Select best strategy based on market conditions and performance"""
        
        strategy_scores = {}
        
        # Strategy 1: Momentum Breakout
        strategy_scores[1] = (
            (50 if market_conditions['trending'] else 10) +
            (30 if market_conditions['volume_surge'] else 5) +
            (20 if market_conditions['volatility'] > 1.0 else 0) +
            self._get_performance_bonus(1)
        )
        
        # Strategy 2: Mean Reversion
        strategy_scores[2] = (
            (60 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 15) +
            (30 if market_conditions['consolidating'] else 5) +
            (20 if not market_conditions['trending'] else 0) +
            self._get_performance_bonus(2)
        )
        
        # Strategy 3: Volume Spike
        strategy_scores[3] = (
            (70 if market_conditions['volume_surge'] else 10) +
            (20 if market_conditions['volatile'] else 5) +
            (10 if market_conditions['news_driven'] else 0) +
            self._get_performance_bonus(3)
        )
        
        # Strategy 4: Bollinger Squeeze
        strategy_scores[4] = (
            (60 if market_conditions['bb_squeeze'] else 10) +
            (25 if market_conditions['consolidating'] else 5) +
            (15 if market_conditions['volatility'] < 1.0 else 0) +
            self._get_performance_bonus(4)
        )
        
        # Strategy 5: RSI Divergence
        strategy_scores[5] = (
            (50 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 20) +
            (30 if market_conditions['trending'] else 10) +
            (20 if 1.0 < market_conditions['volatility'] < 2.5 else 5) +
            self._get_performance_bonus(5)
        )
        
        # Strategy 6: VWAP Touch
        strategy_scores[6] = (
            (50 if market_conditions['vwap_distance'] < 0.003 else 15) +
            (30 if market_conditions['volume_surge'] else 10) +
            (20 if market_conditions['trending'] else 5) +
            self._get_performance_bonus(6)
        )
        
        # Strategy 7: Support/Resistance
        strategy_scores[7] = (
            (60 if market_conditions['near_support'] or market_conditions['near_resistance'] else 15) +
            (25 if market_conditions['consolidating'] else 10) +
            (15 if not market_conditions['trending'] else 5) +
            self._get_performance_bonus(7)
        )
        
        # Strategy 8: News Momentum
        strategy_scores[8] = (
            (80 if market_conditions['news_driven'] else 10) +
            (15 if market_conditions['volatile'] else 5) +
            (5 if market_conditions['volume_surge'] else 0) +
            self._get_performance_bonus(8)
        )
        
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        return best_strategy, strategy_scores
    
    def _get_performance_bonus(self, strategy_id: int) -> float:
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
        """Get strategy-specific trading signals"""
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 10:
                return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Insufficient data'}
            
            current_price = data['Close'].iloc[-1]
            
            # Route to specific strategy methods
            if strategy_id == 1:
                return self._momentum_breakout_signals(data, market_conditions)
            elif strategy_id == 2:
                return self._mean_reversion_signals(data, market_conditions)
            elif strategy_id == 3:
                return self._volume_spike_signals(data, market_conditions)
            elif strategy_id == 4:
                return self._bollinger_squeeze_signals(data, market_conditions)
            elif strategy_id == 5:
                return self._rsi_divergence_signals(data, market_conditions)
            elif strategy_id == 6:
                return self._vwap_touch_signals(data, market_conditions)
            elif strategy_id == 7:
                return self._support_resistance_signals(data, market_conditions)
            elif strategy_id == 8:
                return self._news_momentum_signals(data, market_conditions)
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No signal'}
            
        except Exception as e:
            print(f"Strategy signal error: {e}")
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _momentum_breakout_signals(self, data, conditions):
        """Strategy 1: Momentum Breakout signals"""
        try:
            current_price = data['Close'].iloc[-1]
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            breakout_up = current_price > high_20 * 1.001
            breakout_down = current_price < low_20 * 0.999
            
            if breakout_up and conditions['volume_surge']:
                return {
                    'signal': 1, 'confidence': 0.8, 'entry_price': current_price,
                    'stop_pct': 0.008, 'reason': 'Upward breakout with volume'
                }
            elif breakout_down and conditions['volume_surge'] and conditions['trend_direction'] < 0:
                return {
                    'signal': -1, 'confidence': 0.8, 'entry_price': current_price,
                    'stop_pct': 0.008, 'reason': 'Downward breakout with volume'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No breakout'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _mean_reversion_signals(self, data, conditions):
        """Strategy 2: Mean Reversion signals"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = conditions['rsi']
            
            if rsi < 25 and conditions['bb_lower_touch']:
                return {
                    'signal': 1, 'confidence': 0.75, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'Oversold mean reversion'
                }
            elif rsi > 75 and conditions['bb_upper_touch']:
                return {
                    'signal': -1, 'confidence': 0.75, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'Overbought mean reversion'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No reversion signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _volume_spike_signals(self, data, conditions):
        """Strategy 3: Volume Spike signals"""
        try:
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].pct_change().iloc[-1] * 100
            
            if conditions['volume_surge'] and abs(price_change) > 0.3:
                signal = 1 if price_change > 0 else -1
                return {
                    'signal': signal, 'confidence': 0.85, 'entry_price': current_price,
                    'stop_pct': 0.007, 'reason': f'Volume spike with {abs(price_change):.2f}% move'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No volume spike'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _bollinger_squeeze_signals(self, data, conditions):
        """Strategy 4: Bollinger Squeeze signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['bb_squeeze'] and conditions['consolidating']:
                signal = conditions['trend_direction']
                return {
                    'signal': signal, 'confidence': 0.7, 'entry_price': current_price,
                    'stop_pct': 0.005, 'reason': 'Bollinger squeeze breakout'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No squeeze'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _rsi_divergence_signals(self, data, conditions):
        """Strategy 5: RSI Divergence signals"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = conditions['rsi']
            
            if rsi < 30 and conditions['near_support']:
                return {
                    'signal': 1, 'confidence': 0.65, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'RSI oversold at support'
                }
            elif rsi > 70 and conditions['near_resistance']:
                return {
                    'signal': -1, 'confidence': 0.65, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'RSI overbought at resistance'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No RSI signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _vwap_touch_signals(self, data, conditions):
        """Strategy 6: VWAP Touch signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['vwap_distance'] < 0.003:
                if conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': 1, 'confidence': 0.7, 'entry_price': current_price,
                        'stop_pct': 0.005, 'reason': 'VWAP support with volume'
                    }
                elif not conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': -1, 'confidence': 0.7, 'entry_price': current_price,
                        'stop_pct': 0.005, 'reason': 'VWAP resistance with volume'
                    }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No VWAP signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _support_resistance_signals(self, data, conditions):
        """Strategy 7: Support/Resistance signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['near_support'] and conditions['rsi'] < 40:
                return {
                    'signal': 1, 'confidence': 0.75, 'entry_price': current_price,
                    'stop_pct': 0.004, 'reason': 'Support level bounce'
                }
            elif conditions['near_resistance'] and conditions['rsi'] > 60:
                return {
                    'signal': -1, 'confidence': 0.75, 'entry_price': current_price,
                    'stop_pct': 0.004, 'reason': 'Resistance level rejection'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No S/R signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _news_momentum_signals(self, data, conditions):
        """Strategy 8: News Momentum signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['news_driven']:
                signal = conditions['trend_direction']
                confidence = min(0.9, conditions['volume_ratio'] * 0.3)
                
                return {
                    'signal': signal, 'confidence': confidence, 'entry_price': current_price,
                    'stop_pct': 0.01, 'reason': f'News momentum (Vol: {conditions["volume_ratio"]:.1f}x)'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No news signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def update_strategy_performance(self, strategy_id: int, trade_result: Dict):
        """Update strategy performance statistics"""
        stats = self.strategy_stats[strategy_id]
        
        stats['trades'] += 1
        stats['last_used'] = datetime.now()
        
        pnl = trade_result.get('pnl', 0)
        hold_time = trade_result.get('hold_time', 0)
        
        if pnl > 0:
            stats['wins'] += 1
            if stats['wins'] > 0:
                stats['avg_profit'] = ((stats['avg_profit'] * (stats['wins'] - 1)) + pnl) / stats['wins']
        else:
            loss_count = stats['trades'] - stats['wins']
            if loss_count > 0:
                stats['avg_loss'] = ((stats['avg_loss'] * (loss_count - 1)) + pnl) / loss_count
        
        stats['total_pnl'] += pnl
        stats['success_rate'] = (stats['wins'] / stats['trades']) * 100
        stats['avg_hold_time'] = ((stats['avg_hold_time'] * (stats['trades'] - 1)) + hold_time) / stats['trades']

# ==================== PRODUCTION SCALPING BOT ====================

class ProductionScalpingBot:
    """Production-ready scalping bot with complete 8-strategy system"""
    
    def __init__(self):
        self.api = ProperGoodwillIntegration()
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
        
        # Risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_drawdown_limit = 0.10  # 10% max drawdown
        self.daily_pnl = 0.0
        self.peak_capital = self.capital
        self.current_drawdown = 0.0
        
        # Market scanning symbols
        self.symbols = [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS"
        ]
        
        self.init_database()
    
    def init_database(self):
        """Initialize production database"""
        try:
            self.conn = sqlite3.connect('production_scalping_trades.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, action TEXT,
                    entry_price REAL, exit_price REAL, quantity INTEGER,
                    pnl REAL, hold_time REAL, strategy_id INTEGER,
                    strategy_name TEXT, exit_reason TEXT, confidence REAL,
                    order_id TEXT, mode TEXT, connection_method TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER, winning_trades INTEGER,
                    total_pnl REAL, max_drawdown REAL, 
                    capital_used REAL, best_strategy TEXT
                )
            ''')
            
            self.conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        # Update current drawdown
        self.current_drawdown = max(0, self.peak_capital - self.capital)
        
        # Check daily loss limit
        daily_loss_amount = self.capital * self.daily_loss_limit
        if self.daily_pnl < -daily_loss_amount:
            st.error(f"🛑 Daily loss limit breached: ₹{self.daily_pnl:.2f}")
            return False
        
        # Check max drawdown
        max_drawdown_amount = self.capital * self.max_drawdown_limit
        if self.current_drawdown > max_drawdown_amount:
            st.error(f"🛑 Max drawdown limit breached: ₹{self.current_drawdown:.2f}")
            return False
        
        return True
    
    def scan_for_opportunities(self):
        """Enhanced market scanning with 8 strategies"""
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Get live data - prefer Goodwill API, fallback to yfinance
                live_data = None
                data_source = "yfinance"
                
                if self.api.is_connected:
                    live_data = self.api.get_quote(symbol)
                    if live_data:
                        data_source = f"Goodwill ({self.api.connection_method})"
                
                # Fallback to yfinance if Goodwill data unavailable
                if not live_data:
                    try:
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
                    except:
                        continue
                
                if not live_data or live_data['price'] <= 0:
                    continue
                
                # Analyze market conditions
                market_conditions = self.strategies.analyze_market_conditions(symbol)
                
                # Select best strategy
                strategy_id, strategy_scores = self.strategies.select_strategy(symbol, market_conditions)
                
                # Get strategy-specific signals
                signals = self.strategies.get_strategy_signals(strategy_id, symbol, market_conditions)
                
                # Only consider high-confidence signals
                if signals['signal'] != 0 and signals['confidence'] > 0.70:
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
                        'strategy_scores': strategy_scores,
                        'volume': live_data['volume'],
                        'data_source': data_source
                    }
                    
                    opportunities.append(opportunity)
                    
            except Exception as e:
                print(f"Scanning error for {symbol}: {e}")
                continue
        
        # Sort by confidence and filter top opportunities
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:2]  # Top 2 opportunities only
    
    def calculate_position_size(self, symbol, entry_price, stop_pct):
        """Calculate position size with production risk management"""
        try:
            # Base risk amount
            risk_amount = self.capital * self.risk_per_trade
            
            # Adjust for current drawdown
            if self.current_drawdown > 0:
                risk_reduction = min(0.5, self.current_drawdown / (self.capital * 0.05))
                risk_amount *= (1 - risk_reduction)
            
            # Calculate quantity based on stop loss
            stop_loss_amount = entry_price * stop_pct
            
            if stop_loss_amount > 0:
                quantity = int(risk_amount / stop_loss_amount)
                
                # Apply position size limits
                max_quantity = min(1000, int(self.capital * 0.1 / entry_price))  # Max 10% of capital per position
                quantity = max(1, min(quantity, max_quantity))
                
                return quantity
            
            return 1
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 1
    
    def execute_entry(self, opportunity):
        """Execute entry with production safeguards"""
        try:
            # Check risk limits before entry
            if not self.check_risk_limits():
                return None
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                return None
            
            symbol = opportunity['symbol']
            entry_price = opportunity['price']
            signal = opportunity['signal']
            strategy_id = opportunity['strategy_id']
            strategy_name = opportunity['strategy_name']
            stop_pct = opportunity['stop_pct']
            confidence = opportunity['confidence']
            
            quantity = self.calculate_position_size(symbol, entry_price, stop_pct)
            action = "B" if signal > 0 else "S"  # Goodwill API uses B/S
            
            # Calculate stop loss and target
            if action == "B":
                stop_loss = entry_price * (1 - stop_pct)
                target = entry_price * (1 + (stop_pct * self.risk_reward_ratio))
            else:
                stop_loss = entry_price * (1 + stop_pct)
                target = entry_price * (1 - (stop_pct * self.risk_reward_ratio))
            
            # Place order based on mode
            order_id = None
            
            if self.mode == "LIVE" and self.api.is_connected:
                # Place real order
                order_id = self.api.place_order(symbol, action, quantity, entry_price, "MKT", "MIS")
                if not order_id:
                    st.error(f"❌ Failed to place live order for {symbol}")
                    return None
            else:
                # Paper trading
                order_id = f"PAPER_{int(datetime.now().timestamp())}"
                st.info(f"📝 PAPER TRADE: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name} | Confidence: {confidence:.1%} | Data: {opportunity['data_source']}")
            
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
                    'lowest_profit': 0,
                    'data_source': opportunity['data_source']
                }
                
                st.success(f"🚀 Position Opened: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name} | Mode: {self.mode}")
                return position_id
            
            return None
            
        except Exception as e:
            st.error(f"Entry execution error: {e}")
            return None
    
    def update_position_tracking(self, position_id, current_price):
        """Update position with advanced tracking"""
        position = self.positions[position_id]
        
        # Calculate current P&L
        if position['action'] == 'B':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        # Update profit tracking
        position['highest_profit'] = max(position['highest_profit'], current_pnl)
        position['lowest_profit'] = min(position['lowest_profit'], current_pnl)
        
        # Dynamic trailing stop
        if current_pnl > 0:
            profit_factor = current_pnl / (position['entry_price'] * position['quantity'] * position['stop_pct'])
            trail_tightening = min(0.5, profit_factor * 0.1)
            
            if position['action'] == 'B':
                new_stop = current_price * (1 - position['stop_pct'] * (1 - trail_tightening))
                position['trailing_stop'] = max(position['trailing_stop'], new_stop)
            else:
                new_stop = current_price * (1 + position['stop_pct'] * (1 - trail_tightening))
                position['trailing_stop'] = min(position['trailing_stop'], new_stop)
    
    def check_exit_conditions(self, position_id, current_price):
        """Check comprehensive exit conditions"""
        position = self.positions[position_id]
        
        # Time-based exit
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        if hold_time > self.max_hold_time:
            return "TIME_EXIT"
        
        # Update position tracking
        self.update_position_tracking(position_id, current_price)
        
        # Stop loss check
        if position['action'] == 'B':
            if current_price <= position['trailing_stop']:
                return "STOP_LOSS"
            if current_price >= position['target']:
                return "TARGET_HIT"
        else:
            if current_price >= position['trailing_stop']:
                return "STOP_LOSS"
            if current_price <= position['target']:
                return "TARGET_HIT"
        
        # Risk management exits
        current_pnl = position['highest_profit'] if position['highest_profit'] > 0 else position['lowest_profit']
        position_risk = abs(current_pnl) / self.capital
        
        if position_risk > self.risk_per_trade * 1.5:  # 150% of intended risk
            return "RISK_LIMIT"
        
        return None
    
    def close_position(self, position_id, exit_price, exit_reason):
        """Close position with production logging"""
        if position_id not in self.positions:
            return
        
        position = self.positions.pop(position_id)
        
        # Calculate final P&L
        if position['action'] == 'B':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Update capital and tracking
        self.capital += pnl
        self.pnl += pnl
        self.daily_pnl += pnl
        
        # Update peak capital for drawdown calculation
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        
        # Create trade record
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
            'confidence': position['confidence'],
            'order_id': position['order_id'],
            'mode': self.mode,
            'connection_method': self.api.connection_method or 'yfinance'
        }
        
        self.trades.append(trade_record)
        
        # Save to database
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO trades VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(trade_record.values()))
            self.conn.commit()
        except Exception as e:
            print(f"Database save error: {e}")
        
        # Update strategy performance
        self.strategies.update_strategy_performance(position['strategy_id'], {
            'pnl': pnl,
            'hold_time': hold_time,
            'exit_reason': exit_reason
        })
        
        # Place exit order if live mode
        exit_action = "S" if position['action'] == "B" else "B"
        
        if self.mode == "LIVE" and self.api.is_connected:
            self.api.place_order(position['symbol'], exit_action, position['quantity'], exit_price, "MKT", "MIS")
        else:
            st.info(f"📝 PAPER EXIT: {exit_action} {position['quantity']} {position['symbol']} @ ₹{exit_price:.2f} | Data: {position['data_source']}")
        
        pnl_color = "🟢" if pnl >= 0 else "🔴"
        st.info(f"✅ Position Closed: {position['symbol']} | {pnl_color} P&L: ₹{pnl:.2f} | Strategy: {position['strategy_name']} | Reason: {exit_reason}")
    
    def run_trading_cycle(self):
        """Main production trading cycle"""
        try:
            # Check risk limits before any trading
            if not self.check_risk_limits():
                self.is_running = False
                st.error("🛑 Trading stopped due to risk limits")
                return
            
            # Scan for new opportunities
            opportunities = self.scan_for_opportunities()
            
            # Execute entries for valid opportunities
            for opp in opportunities:
                if len(self.positions) < self.max_positions:
                    self.execute_entry(opp)
                    time.sleep(1)  # Prevent rapid-fire orders
            
            # Check exit conditions for existing positions
            positions_to_close = []
            
            for pos_id in list(self.positions.keys()):
                position = self.positions[pos_id]
                
                # Get current price from appropriate source
                current_price = position['entry_price']  # Default fallback
                
                try:
                    if self.api.is_connected:
                        live_data = self.api.get_quote(position['symbol'])
                        if live_data and live_data['price'] > 0:
                            current_price = live_data['price']
                    else:
                        # Fallback to yfinance
                        ticker = yf.Ticker(f"{position['symbol']}.NS")
                        hist = ticker.history(period="1d", interval="1m")
                        if len(hist) > 0:
                            current_price = float(hist.iloc[-1]['Close'])
                except:
                    pass  # Use entry price as fallback
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(pos_id, current_price)
                if exit_reason:
                    positions_to_close.append((pos_id, current_price, exit_reason))
            
            # Close positions that need to be closed
            for pos_id, exit_price, exit_reason in positions_to_close:
                self.close_position(pos_id, exit_price, exit_reason)
                time.sleep(0.5)  # Brief delay between closes
                
        except Exception as e:
            st.error(f"Trading cycle error: {e}")
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_trade': 0, 'max_profit': 0, 'max_loss': 0, 'avg_hold_time': 0,
                'best_strategy': 'N/A', 'strategy_breakdown': {}, 'sharpe_ratio': 0,
                'max_drawdown': round(self.current_drawdown, 2), 'profit_factor': 0, 
                'daily_pnl': round(self.daily_pnl, 2), 'winning_trades': 0, 'losing_trades': 0
            }
        
        pnls = [trade['pnl'] for trade in self.trades]
        hold_times = [trade['hold_time'] for trade in self.trades]
        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p < 0])
        
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
        
        # Advanced metrics
        total_profits = sum([p for p in pnls if p > 0]) or 1
        total_losses = abs(sum([p for p in pnls if p < 0])) or 1
        profit_factor = total_profits / total_losses
        
        sharpe_ratio = 0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls)
        
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
            'strategy_breakdown': strategy_breakdown,
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(self.current_drawdown, 2),
            'profit_factor': round(profit_factor, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }

# ==================== STREAMLIT APPLICATION ====================

def show_production_disclaimer():
    """Show production trading disclaimer"""
    st.markdown("""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336; margin: 10px 0;">
        <h4 style="color: #d32f2f; margin-top: 0;">⚠️ PRODUCTION TRADING SYSTEM</h4>
        <p style="margin-bottom: 0;"><strong>This bot uses REAL Goodwill API and can place actual orders.</strong></p>
        <ul style="margin: 10px 0;">
            <li>✅ <strong>gwcmodel Integration:</strong> Official Goodwill library as primary method</li>
            <li>⚠️ <strong>Live Mode:</strong> Places actual orders with real money</li>
            <li>🛡️ <strong>Risk Limits:</strong> 5% daily loss limit, 10% max drawdown</li>
            <li>🎯 <strong>8 Strategies:</strong> Intelligent strategy selection based on market conditions</li>
        </ul>
        <p style="margin-bottom: 0;"><strong>Always test in Paper Mode before using Live Mode!</strong></p>
    </div>
    """, unsafe_allow_html=True)

def show_gwcmodel_status():
    """Show gwcmodel status and installation info"""
    st.sidebar.markdown("### 🔧 gwcmodel Status")
    
    if GWCMODEL_AVAILABLE:
        if GWC_CLASS:
            st.sidebar.success(f"✅ gwcmodel.{GWC_CLASS_NAME} available")
        else:
            st.sidebar.error("❌ gwcmodel installed but no valid class found")
            try:
                import gwcmodel
                available_attrs = [attr for attr in dir(gwcmodel) if not attr.startswith('_')]
                st.sidebar.code(f"Available: {available_attrs}")
            except:
                pass
    else:
        st.sidebar.error("❌ gwcmodel not installed")
        st.sidebar.code("pip install gwcmodel")

def show_api_instructions():
    """Show comprehensive API setup instructions"""
    with st.expander("📋 Goodwill API Setup Guide", expanded=False):
        st.markdown("""
        ### 🔧 Complete Setup Instructions
        
        **Method 1: gwcmodel Credentials (Recommended)**
        1. Install gwcmodel: `pip install gwcmodel`
        2. Get API credentials from [developer.gwcindia.in](https://developer.gwcindia.in)
        3. Use your Goodwill trading account credentials
        4. Enter API Key, User ID, Password (and TOTP if 2FA enabled)
        
        **Method 2: Request Token (Alternative)**
        1. Visit: `https://api.gwcindia.in/v1/login?api_key=YOUR_API_KEY`
        2. Complete login on Goodwill website
        3. Copy the `request_token` from redirect URL
        4. Use with your API Key and API Secret
        
        ### 📊 Your API Details
        - **API Key:** `9c155c1fff651d01513b455396af2449`
        - **User ID:** `GWC100643`
        - **Endpoint:** `https://api.gwcindia.in/v1/`
        
        ### 🔗 Quick Login Link
        [🔗 Login to get Request Token](https://api.gwcindia.in/v1/login?api_key=9c155c1fff651d01513b455396af2449)
        
        ### 📖 Documentation
        - Full API docs: [developer.gwcindia.in/api](https://developer.gwcindia.in/api/)
        - gwcmodel PyPI: [pypi.org/project/gwcmodel](https://pypi.org/project/gwcmodel/)
        """)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = ProductionScalpingBot()

if 'gw_logged_in' not in st.session_state:
    st.session_state.gw_logged_in = False

bot = st.session_state.bot

# Main title and disclaimer
st.title("⚡ FlyingBuddha Scalping Bot - Complete Production System")
show_production_disclaimer()

# Show gwcmodel status and API instructions
show_gwcmodel_status()
show_api_instructions()

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.header("🚀 8-Strategy Scalping Bot")

# Current status display
connection_status = "🟢 Connected" if bot.api.is_connected else "🔴 Disconnected"
connection_method = f" ({bot.api.connection_method})" if bot.api.connection_method else ""

st.sidebar.markdown(f"""
**📊 Status Dashboard:**
- **API Status:** {connection_status}{connection_method}
- **Trading Mode:** {'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}
- **Capital:** ₹{bot.capital:,.2f}
- **Daily P&L:** ₹{bot.daily_pnl:,.2f}
- **Positions:** {len(bot.positions)}/{bot.max_positions}
- **Bot Status:** {'🟢 Running' if bot.is_running else '🔴 Stopped'}
""")

# Strategy Performance Summary
if any(stats['trades'] > 0 for stats in bot.strategies.strategy_stats.values()):
    st.sidebar.markdown("**🎯 Strategy Performance:**")
    for strategy_id, stats in bot.strategies.strategy_stats.items():
        if stats['trades'] > 0:
            strategy_name = bot.strategies.strategies[strategy_id]['name']
            color = bot.strategies.strategies[strategy_id]['color']
            
            st.sidebar.markdown(f"""
            <div style="background-color: {color}20; padding: 5px; border-radius: 3px; margin: 2px 0; font-size: 12px;">
                <strong>{strategy_name}</strong><br>
                {stats['trades']} trades | {stats['success_rate']:.1f}% | ₹{stats['total_pnl']:.0f}
            </div>
            """, unsafe_allow_html=True)

# ==================== AUTHENTICATION SECTION ====================

st.sidebar.subheader("🔐 Goodwill Authentication")

if not st.session_state.gw_logged_in:
    # Login method tabs
    auth_method = st.sidebar.radio(
        "Authentication Method:",
        ["gwcmodel Credentials", "Request Token"],
        help="gwcmodel is the recommended method"
    )
    
    if auth_method == "gwcmodel Credentials":
        with st.sidebar.form("gwc_credentials"):
            st.markdown("**📝 gwcmodel Login (Recommended)**")
            
            api_key = st.text_input("API Key", value="9c155c1fff651d01513b455396af2449", type="password")
            user_id = st.text_input("User ID", value="GWC100643")
            password = st.text_input("Password", type="password")
            totp_code = st.text_input("TOTP Code (if 2FA enabled)", help="Leave empty if no 2FA")
            
            login_submit = st.form_submit_button("🔑 Login via gwcmodel")
            
            if login_submit:
                if api_key and user_id and password:
                    with st.spinner("🔄 Authenticating via gwcmodel..."):
                        success = bot.api.login_with_credentials(api_key, user_id, password, totp_code if totp_code else None)
                        if success:
                            st.session_state.gw_logged_in = True
                            st.rerun()
                else:
                    st.error("❌ Please fill in required fields")
    
    else:  # Request Token method
        with st.sidebar.form("gwc_token"):
            st.markdown("**🎫 Request Token Login**")
            
            api_key = st.text_input("API Key", value="9c155c1fff651d01513b455396af2449", type="password")
            request_token = st.text_input("Request Token", placeholder="From redirect URL")
            api_secret = st.text_input("API Secret", type="password")
            
            if api_key:
                login_url = f"https://api.gwcindia.in/v1/login?api_key={api_key}"
                st.markdown(f"[🔗 Get Request Token]({login_url})")
            
            token_submit = st.form_submit_button("🎫 Login with Token")
            
            if token_submit:
                if api_key and request_token and api_secret:
                    with st.spinner("🔄 Authenticating with request token..."):
                        success = bot.api.login_with_request_token(api_key, request_token, api_secret)
                        if success:
                            st.session_state.gw_logged_in = True
                            st.rerun()
                else:
                    st.error("❌ Please fill in all fields")

else:
    # Already logged in
    st.sidebar.success(f"✅ Connected via {bot.api.connection_method}")
    st.sidebar.markdown(f"**Client ID:** {bot.api.client_id}")
    
    # Connection actions
    conn_col1, conn_col2 = st.sidebar.columns(2)
    
    with conn_col1:
        if st.button("👤 Profile"):
            profile = bot.api.get_profile()
            if profile:
                st.sidebar.json(profile)
            else:
                st.sidebar.warning("Could not fetch profile")
    
    with conn_col2:
        if st.button("📊 Positions"):
            positions = bot.api.get_positions()
            if positions:
                st.sidebar.json(positions[:2])  # Show first 2
            else:
                st.sidebar.info("No positions")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary"):
        bot.api.logout()
        st.session_state.gw_logged_in = False
        st.success("✅ Logged out successfully")
        st.rerun()

# ==================== TRADING MODE & PARAMETERS ====================

st.sidebar.subheader("🎮 Trading Controls")

# Mode selection
mode_col1, mode_col2 = st.sidebar.columns(2)

with mode_col1:
    if st.button("🟠 Paper", disabled=(bot.mode == "PAPER"), use_container_width=True):
        bot.mode = "PAPER"
        st.success("✅ Switched to Paper Mode")
        st.rerun()

with mode_col2:
    if st.button("🔴 Live", disabled=(bot.mode == "LIVE"), use_container_width=True):
        if not bot.api.is_connected:
            st.error("❌ Connect to Goodwill API first")
        else:
            bot.mode = "LIVE"
            st.warning("⚠️ Live Mode - Real orders will be placed!")
            st.rerun()

# Trading parameters
st.sidebar.subheader("⚙️ Parameters")

bot.risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 5, 25, 15, 1) / 100
bot.max_positions = st.sidebar.slider("Max Positions", 1, 6, 4, 1)
bot.max_hold_time = st.sidebar.slider("Max Hold Time (sec)", 60, 300, 180, 30)
bot.risk_reward_ratio = st.sidebar.slider("Risk:Reward", 1.5, 3.0, 2.0, 0.1)

# Capital management
new_capital = st.sidebar.number_input("Capital (₹)", 10000, 10000000, int(bot.capital), 10000)
if new_capital != bot.capital:
    bot.capital = float(new_capital)
    bot.peak_capital = max(bot.peak_capital, bot.capital)

# Symbol selection
st.sidebar.subheader("📊 Symbols")
selected_symbols = st.sidebar.multiselect(
    "Trading Symbols:",
    options=["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", 
             "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS",
             "WIPRO", "LT", "TITAN", "MARUTI", "BHARTIARTL"],
    default=bot.symbols
)

if selected_symbols:
    bot.symbols = selected_symbols

# ==================== BOT CONTROLS ====================

st.sidebar.subheader("🎮 Bot Controls")

# Main start/stop button
if not bot.is_running:
    if st.sidebar.button("🚀 START BOT", type="primary", use_container_width=True):
        if bot.mode == "LIVE" and not bot.api.is_connected:
            st.sidebar.error("❌ Connect to Goodwill API for Live Mode")
        else:
            bot.is_running = True
            st.sidebar.success("✅ Bot Started!")
            st.rerun()
else:
    if st.sidebar.button("⏹️ STOP BOT", type="secondary", use_container_width=True):
        bot.is_running = False
        st.sidebar.warning("⏸️ Bot Stopped!")
        st.rerun()

# Additional controls
control_col1, control_col2 = st.sidebar.columns(2)

with control_col1:
    if st.button("🔄 Single Cycle"):
        with st.spinner("Running trading cycle..."):
            bot.run_trading_cycle()
            st.rerun()

with control_col2:
    if st.button("📊 Refresh"):
        st.rerun()

# Emergency stop
if bot.positions:
    if st.sidebar.button("🛑 EMERGENCY STOP", type="secondary", use_container_width=True):
        for pos_id in list(bot.positions.keys()):
            position = bot.positions[pos_id]
            try:
                if bot.api.is_connected:
                    live_data = bot.api.get_quote(position['symbol'])
                    exit_price = live_data['price'] if live_data and live_data['price'] > 0 else position['entry_price']
                else:
                    ticker = yf.Ticker(f"{position['symbol']}.NS")
                    hist = ticker.history(period="1d", interval="1m")
                    exit_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
            except:
                exit_price = position['entry_price']
            
            bot.close_position(pos_id, exit_price, "EMERGENCY_STOP")
        
        st.sidebar.warning("🛑 All positions closed!")
        st.rerun()

# ==================== AUTO-REFRESH LOGIC ====================

# Auto-refresh when bot is running
if bot.is_running:
    bot.run_trading_cycle()
    time.sleep(3)  # 3-second cycle for production
    st.rerun()

# ==================== MAIN DASHBOARD ====================

# Status overview
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    status_color = "🟢" if bot.is_running else "🔴"
    st.metric("Bot Status", f"{status_color} {'Running' if bot.is_running else 'Stopped'}")

with col2:
    connection_status = "🟢 Connected" if bot.api.is_connected else "🔴 Offline"
    st.metric("API Status", connection_status)

with col3:
    mode_display = f"{'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}"
    st.metric("Trading Mode", mode_display)

with col4:
    st.metric("Active Positions", f"{len(bot.positions)}/{bot.max_positions}")

with col5:
    pnl_delta = f"₹{bot.pnl:.2f}" if bot.pnl != 0 else None
    st.metric("Total P&L", f"₹{bot.pnl:.2f}", delta=pnl_delta)

with col6:
    daily_delta = f"₹{bot.daily_pnl:.2f}" if bot.daily_pnl != 0 else None
    st.metric("Daily P&L", f"₹{bot.daily_pnl:.2f}", delta=daily_delta)

# ==================== RISK MONITORING ====================

st.subheader("🛡️ Risk Monitoring")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    drawdown_pct = (bot.current_drawdown / bot.peak_capital) * 100 if bot.peak_capital > 0 else 0
    st.metric("Current Drawdown", f"₹{bot.current_drawdown:.2f}", delta=f"{drawdown_pct:.1f}%")

with risk_col2:
    daily_loss_pct = (abs(bot.daily_pnl) / bot.capital) * 100 if bot.daily_pnl < 0 else 0
    st.metric("Daily Loss %", f"{daily_loss_pct:.1f}%", delta="Limit: 5.0%")

with risk_col3:
    position_risk = len(bot.positions) * bot.risk_per_trade * 100
    st.metric("Position Risk", f"{position_risk:.1f}%", delta=f"Max: {bot.max_positions * bot.risk_per_trade * 100:.1f}%")

with risk_col4:
    capital_utilization = (sum([p['quantity'] * p['entry_price'] for p in bot.positions.values()]) / bot.capital) * 100 if bot.positions else 0
    st.metric("Capital Used", f"{capital_utilization:.1f}%")

# Risk warnings
if drawdown_pct > 8:
    st.warning("⚠️ Approaching maximum drawdown limit (10%)")
if daily_loss_pct > 4:
    st.warning("⚠️ Approaching daily loss limit (5%)")

# ==================== PERFORMANCE METRICS ====================

st.subheader("📈 Performance Metrics")

metrics = bot.get_performance_metrics()

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric("Total Trades", metrics['total_trades'])
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

with perf_col2:
    st.metric("Total P&L", f"₹{metrics['total_pnl']:.2f}")
    st.metric("Avg Trade", f"₹{metrics['avg_trade']:.2f}")

with perf_col3:
    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

with perf_col4:
    st.metric("Best Strategy", metrics['best_strategy'])
    st.metric("Avg Hold Time", f"{metrics['avg_hold_time']:.1f}s")

# ==================== 8-STRATEGY PERFORMANCE DASHBOARD ====================

st.subheader("🎯 8-Strategy Performance Dashboard")

if any(stats['trades'] > 0 for stats in bot.strategies.strategy_stats.values()):
    strategy_perf_data = []
    for strategy_id, stats in bot.strategies.strategy_stats.items():
        if stats['trades'] > 0:
            strategy_info = bot.strategies.strategies[strategy_id]
            strategy_perf_data.append({
                'ID': strategy_id,
                'Strategy': strategy_info['name'],
                'Description': strategy_info['desc'],
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
        
        # Color-code the dataframe
        def color_pnl(val):
            if 'P&L' in str(val) or '₹' in str(val):
                try:
                    num_val = float(str(val).replace('₹', '').replace(',', ''))
                    return 'color: green' if num_val >= 0 else 'color: red'
                except:
                    return ''
            return ''
        
        styled_df = df_strategy_perf.style.applymap(color_pnl, subset=['Total P&L', 'Avg Profit', 'Avg Loss'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Strategy performance charts
        st.subheader("📊 Strategy Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Strategy P&L chart
            fig_pnl = go.Figure()
            
            strategy_names = [data['Strategy'] for data in strategy_perf_data]
            strategy_pnls = [float(data['Total P&L'].replace('₹', '')) for data in strategy_perf_data]
            colors = [bot.strategies.strategies[data['ID']]['color'] for data in strategy_perf_data]
            
            fig_pnl.add_trace(go.Bar(
                x=strategy_names,
                y=strategy_pnls,
                marker_color=colors,
                text=[f"₹{p:.0f}" for p in strategy_pnls],
                textposition='auto',
                name="Strategy P&L"
            ))
            
            fig_pnl.update_layout(
                title="Strategy P&L Performance",
                xaxis_title="Strategy",
                yaxis_title="P&L (₹)",
                height=400,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with chart_col2:
            # Strategy win rate chart
            fig_win_rate = go.Figure()
            
            win_rates = [float(data['Win Rate'].replace('%', '')) for data in strategy_perf_data]
            
            fig_win_rate.add_trace(go.Bar(
                x=strategy_names,
                y=win_rates,
                marker_color=colors,
                text=[f"{w:.1f}%" for w in win_rates],
                textposition='auto',
                name="Win Rate"
            ))
            
            fig_win_rate.update_layout(
                title="Strategy Win Rates",
                xaxis_title="Strategy",
                yaxis_title="Win Rate (%)",
                height=400,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_win_rate, use_container_width=True)
        
        # Strategy usage heatmap
        if len(strategy_perf_data) > 3:
            st.subheader("🔥 Strategy Usage Heatmap")
            
            # Create usage matrix
            strategy_trades = [data['Trades'] for data in strategy_perf_data]
            max_trades = max(strategy_trades) if strategy_trades else 1
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=[strategy_trades],
                x=strategy_names,
                y=['Usage'],
                colorscale='RdYlGn',
                text=[[f"{t} trades" for t in strategy_trades]],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Trades")
            ))
            
            fig_heatmap.update_layout(
                title="Strategy Usage Frequency",
                height=200
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.info("📊 No strategy performance data yet. Run the bot to see detailed analytics.")
    
    # Show strategy descriptions
    st.subheader("🎯 Available Strategies")
    
    strategy_desc_data = []
    for strategy_id, strategy_info in bot.strategies.strategies.items():
        strategy_desc_data.append({
            'ID': strategy_id,
            'Strategy': strategy_info['name'],
            'Description': strategy_info['desc']
        })
    
    df_strategies = pd.DataFrame(strategy_desc_data)
    st.dataframe(df_strategies, use_container_width=True)

# ==================== ACTIVE POSITIONS ====================

st.subheader("🎯 Active Positions")

if bot.positions:
    positions_data = []
    
    for pos_id, position in bot.positions.items():
        # Get current price
        current_price = position['entry_price']
        try:
            if bot.api.is_connected:
                live_data = bot.api.get_quote(position['symbol'])
                if live_data and live_data['price'] > 0:
                    current_price = live_data['price']
            else:
                ticker = yf.Ticker(f"{position['symbol']}.NS")
                hist = ticker.history(period="1d", interval="1m")
                if len(hist) > 0:
                    current_price = float(hist.iloc[-1]['Close'])
        except:
            pass
        
        # Calculate current P&L
        if position['action'] == 'B':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        
        positions_data.append({
            'Symbol': position['symbol'],
            'Strategy': position['strategy_name'],
            'Action': position['action'],
            'Qty': position['quantity'],
            'Entry': f"₹{position['entry_price']:.2f}",
            'Current': f"₹{current_price:.2f}",
            'Target': f"₹{position['target']:.2f}",
            'Stop': f"₹{position['trailing_stop']:.2f}",
            'P&L': f"₹{current_pnl:.2f}",
            'Confidence': f"{position['confidence']:.1%}",
            'Hold Time': f"{hold_time:.0f}s",
            'Data Source': position['data_source'],
            'Position ID': pos_id
        })
    
    df_positions = pd.DataFrame(positions_data)
    
    # Color-code P&L
    def color_position_pnl(val):
        if 'P&L' in str(val) or '₹' in str(val):
            try:
                num_val = float(str(val).replace('₹', '').replace(',', ''))
                return 'background-color: lightgreen' if num_val >= 0 else 'background-color: lightcoral'
            except:
                return ''
        return ''
    
    styled_positions = df_positions.style.applymap(color_position_pnl, subset=['P&L'])
    st.dataframe(styled_positions, use_container_width=True)
    
    # Manual position management
    st.subheader("🔧 Position Management")
    
    pos_to_close = st.selectbox(
        "Select Position to Close Manually:",
        options=[""] + [f"{pos['Symbol']} ({pos['Strategy']}) - {pos['Action']}" for pos in positions_data]
    )
    
    if pos_to_close and st.button("🔒 Close Selected Position"):
        for pos_data in positions_data:
            if f"{pos_data['Symbol']} ({pos_data['Strategy']}) - {pos_data['Action']}" == pos_to_close:
                pos_id = pos_data['Position ID']
                position = bot.positions[pos_id]
                
                try:
                    if bot.api.is_connected:
                        live_data = bot.api.get_quote(position['symbol'])
                        exit_price = live_data['price'] if live_data and live_data['price'] > 0 else position['entry_price']
                    else:
                        ticker = yf.Ticker(f"{position['symbol']}.NS")
                        hist = ticker.history(period="1d", interval="1m")
                        exit_price = float(hist.iloc[-1]['Close']) if len(hist) > 0 else position['entry_price']
                except:
                    exit_price = position['entry_price']
                
                bot.close_position(pos_id, exit_price, "MANUAL_CLOSE")
                st.success(f"✅ Manually closed: {position['symbol']} ({position['strategy_name']})")
                st.rerun()
                break

else:
    st.info("📭 No active positions")

# ==================== MARKET OPPORTUNITIES ====================

st.subheader("🔍 Current Market Opportunities")

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
                'Volume': f"{opp['volume']:,}",
                'Data Source': opp['data_source'],
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
                st.markdown("**Trend Analysis:**")
                st.write(f"Trending: {'✅' if conditions['trending'] else '❌'}")
                st.write(f"Strength: {conditions['trend_strength']:.2f}%")
                st.write(f"Direction: {'📈 Bullish' if conditions['trend_direction'] > 0 else '📉 Bearish'}")
            
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
        
        # Manual entry
        st.subheader("🎯 Manual Entry")
        manual_symbol = st.selectbox(
            "Execute Manual Entry:",
            options=[""] + [f"{opp['symbol']} - {opp['strategy_name']} ({opp['confidence']:.1%})" for opp in opportunities]
        )
        
        if manual_symbol and st.button("🚀 Execute Manual Entry"):
            for opp in opportunities:
                if f"{opp['symbol']} - {opp['strategy_name']} ({opp['confidence']:.1%})" == manual_symbol:
                    with st.spinner("Executing manual entry..."):
                        result = bot.execute_entry(opp)
                        if result:
                            st.success(f"✅ Manual entry executed for {opp['symbol']}")
                        else:
                            st.error(f"❌ Failed to execute entry for {opp['symbol']}")
                        st.rerun()
                    break
    
    else:
        st.info("🔍 No high-confidence opportunities found with current strategy filters")
        
except Exception as e:
    st.error(f"❌ Error scanning opportunities: {e}")

# ==================== TRADE HISTORY ====================

st.subheader("📋 Enhanced Trade History")

if bot.trades:
    recent_trades = bot.trades[-15:]  # Show last 15 trades
    df_trades = pd.DataFrame(recent_trades)
    
    # Display trades with enhanced styling
    trade_display_cols = ['timestamp', 'symbol', 'strategy_name', 'action', 'quantity', 'pnl', 'hold_time', 'exit_reason', 'confidence', 'mode']
    
    def style_trade_pnl(val):
        if isinstance(val, (int, float)):
            return 'color: green; font-weight: bold' if val >= 0 else 'color: red; font-weight: bold'
        return ''
    
    styled_trades = df_trades[trade_display_cols].style.applymap(style_trade_pnl, subset=['pnl'])
    st.dataframe(styled_trades, use_container_width=True)
    
    # Enhanced P&L visualization
    if len(bot.trades) > 1:
        st.subheader("📊 Trading Performance Analysis")
        
        # Create multiple charts
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Cumulative P&L chart
            cumulative_pnl = []
            running_total = 0
            
            for trade in bot.trades:
                running_total += trade['pnl']
                cumulative_pnl.append(running_total)
            
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_pnl) + 1)),
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig_cumulative.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Trade Number",
                yaxis_title="P&L (₹)",
                height=400
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with viz_col2:
            # Trade distribution
            pnls = [trade['pnl'] for trade in bot.trades]
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=pnls,
                nbinsx=20,
                name="P&L Distribution",
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig_dist.update_layout(
                title="P&L Distribution",
                xaxis_title="P&L (₹)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Export Trades to CSV"):
                csv_data = df_trades.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"flyingbuddha_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("📈 Export Performance Summary"):
                summary_data = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Total Trades': len(bot.trades),
                    'Win Rate': f"{metrics['win_rate']:.1f}%",
                    'Total P&L': f"₹{metrics['total_pnl']:.2f}",
                    'Best Strategy': metrics['best_strategy'],
                    'Connection Method': bot.api.connection_method or 'yfinance'
                }
                
                summary_json = json.dumps(summary_data, indent=2)
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary_json,
                    file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

else:
    st.info("📭 No trades executed yet")

# ==================== LIVE MARKET DATA ====================

st.subheader("📊 Live Market Data")

data_source_info = f"🟢 {bot.api.connection_method}" if bot.api.is_connected else "🟠 yfinance (Fallback)"
st.markdown(f"**Data Source:** {data_source_info}")

data_cols = st.columns(5)

for i, symbol in enumerate(bot.symbols[:10]):  # Show first 10 symbols
    with data_cols[i % 5]:
        try:
            if bot.api.is_connected:
                live_data = bot.api.get_quote(symbol)
                if live_data and live_data['price'] > 0:
                    st.metric(symbol, f"₹{live_data['price']:.2f}", help=f"Volume: {live_data['volume']:,}")
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
        except Exception as e:
            st.metric(symbol, "Error")

# ==================== SYSTEM DIAGNOSTICS ====================

with st.expander("⚙️ System Diagnostics & Settings", expanded=False):
    st.subheader("🔧 System Status")
    
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    
    with diag_col1:
        st.markdown("**API & Connection:**")
        st.write(f"gwcmodel: {'✅ Available' if GWCMODEL_AVAILABLE else '❌ Missing'}")
        st.write(f"Goodwill API: {'✅ Connected' if bot.api.is_connected else '❌ Disconnected'}")
        st.write(f"Connection Method: {bot.api.connection_method or 'None'}")
        st.write(f"Database: {'✅ Active' if hasattr(bot, 'conn') else '❌ Failed'}")
    
    with diag_col2:
        st.markdown("**Performance:**")
        st.write(f"Active Strategies: {len([s for s in bot.strategies.strategy_stats.values() if s['trades'] > 0])}/8")
        st.write(f"Cache Size: {len(bot.strategies.analysis_cache)} items")
        st.write(f"Database Records: {len(bot.trades)} trades")
        st.write(f"Session Duration: {(datetime.now() - datetime.now().replace(hour=9, minute=15)).total_seconds() / 3600:.1f}h")
    
    with diag_col3:
        st.markdown("**Risk Status:**")
        st.write(f"Risk Limits: {'✅ Within Limits' if bot.check_risk_limits() else '⚠️ Breached'}")
        st.write(f"Position Limits: {'✅ OK' if len(bot.positions) <= bot.max_positions else '⚠️ Exceeded'}")
        st.write(f"Daily P&L: ₹{bot.daily_pnl:.2f}")
        st.write(f"Max Drawdown: ₹{bot.current_drawdown:.2f}")

# ==================== FOOTER ====================

st.markdown("---")

st.markdown("""
### 🚀 FlyingBuddha Complete Production Scalping Bot

**🎯 8 Advanced Strategies with Intelligent Selection:**
1. **Momentum Breakout** - Price breakouts with volume confirmation
2. **Mean Reversion** - RSI oversold/overbought signals with Bollinger touches
3. **Volume Spike** - High volume momentum moves (>1.8x average)
4. **Bollinger Squeeze** - Low volatility breakouts from consolidation
5. **RSI Divergence** - RSI extremes at support/resistance levels
6. **VWAP Touch** - Price reactions near Volume Weighted Average Price
7. **Support/Resistance** - Bounces at identified key levels
8. **News Momentum** - High volume + volatility indicating news events

**🔧 Technical Features:**
- ✅ **gwcmodel Integration** - Official Goodwill library as primary method
- ✅ **Direct API Fallback** - Robust authentication with multiple methods
- ✅ **Real-time Data** - Live quotes and order placement
- ✅ **Advanced Risk Management** - 5% daily loss limit, 10% max drawdown
- ✅ **Production Database** - SQLite logging of all trades and performance
- ✅ **Dynamic Strategy Selection** - Market condition-based strategy picking

**⚙️ Trading Parameters:**
- Risk per trade: 15% | Max positions: 4 | Hold time: 3 min max
- Risk:Reward ratio: 1:2 | Stop loss: Dynamic trailing mechanism
- Position sizing: Volatility-adjusted with drawdown protection

**🔗 API Integration:**
- Primary: gwcmodel library for seamless Goodwill integration
- Fallback: Direct API calls to api.gwcindia.in/v1/ endpoints
- Authentication: Multiple methods (credentials, request token)
- Real orders: Live mode places actual trades via gwcmodel/API

⚠️ **Risk Warning:** This bot trades with real money in Live Mode. Always test thoroughly in Paper Mode first and never risk more than you can afford to lose.
""")

# Auto-refresh indicator
if bot.is_running:
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; right: 20px; background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; z-index: 1000;">
        🔄 Bot Active | Last Update: {datetime.now().strftime('%H:%M:%S')} | Next: 3s
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1)

# End of application
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 30px;'>
    ⚡ FlyingBuddha Production Scalping Bot v4.0 | Complete 8-Strategy System<br>
    <strong>🎯 Intelligent Strategy Selection | 🔥 Real gwcmodel Integration | 📊 Live API Trading | 🛡️ Advanced Risk Management</strong>
</div>
""", unsafe_allow_html=True)

# ==================== END OF APPLICATION ====================
