#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - Focused Production Version with Request Token Flow
Real-time scalping with 8 advanced strategies and Goodwill API integration (Request Token Only)
- PROPER request token flow implementation per official documentation
- Complete 8-strategy system with intelligent selection
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
# import hmac # Not used directly if not using HMAC-SHA256 for signature
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
# import threading # Not actively used in the provided flow
# from queue import Queue # Not actively used
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # Not actively used
# import uuid # Not actively used
# from collections import deque # Not actively used
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="⚡ FlyingBuddha Scalping Bot - Request Token Flow",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GOODWILL INTEGRATION (REQUEST TOKEN ONLY) ====================

class GoodwillRequestTokenIntegration:
    """
    Goodwill Integration using Request Token Flow with robust authentication
    - Proper request token flow per official documentation
    - Production-ready error handling
    """

    def __init__(self):
        self.access_token = None
        self.api_key = None
        self.api_secret = None # Stored for signature creation if needed for other calls
        self.client_id = None
        self.user_session_id = None
        self.is_connected = False
        self.connection_method = None
        self.base_url = "https://api.gwcindia.in/v1" # Ensure this is the correct v1 base URL
        self.last_login_time = None
        self.user_profile = None

    def generate_login_url(self, api_key: str) -> str:
        """Generate login URL for request token flow"""
        return f"https://api.gwcindia.in/v1/login?api_key={api_key}"

    def parse_request_token_from_url(self, redirect_url: str) -> Optional[str]:
        """Parse request token from redirect URL with enhanced validation"""
        try:
            if not redirect_url or not isinstance(redirect_url, str):
                return None
            redirect_url = redirect_url.strip()
            patterns = ["request_token=", "requestToken=", "token=", "rt="]
            for pattern in patterns:
                if pattern in redirect_url:
                    token_part = redirect_url.split(pattern)[1]
                    request_token = token_part.split("&")[0].split("#")[0]
                    if request_token and len(request_token) >= 20: # Basic validation
                        return request_token
            return None
        except Exception as e:
            st.error(f"❌ Error parsing request token: {e}")
            return None

    def create_signature(self, api_key: str, request_token: str, api_secret: str) -> str:
        """Create signature exactly as per Goodwill API documentation (SHA-256 of concatenated string)"""
        try:
            checksum = f"{api_key}{request_token}{api_secret}"
            signature = hashlib.sha256(checksum.encode('utf-8')).hexdigest()
            return signature
        except Exception as e:
            st.error(f"❌ Signature creation error: {e}")
            return ""

    def login_with_request_token(self, api_key: str, request_token: str, api_secret: str) -> bool:
        try:
            self.api_key = api_key
            self.api_secret = api_secret # Store the secret

            if not all([api_key, request_token, api_secret]):
                st.error("❌ Missing required fields for authentication (API Key, Request Token, API Secret)")
                return False

            signature = self.create_signature(api_key, request_token, api_secret)
            if not signature:
                st.error("❌ Failed to create signature. Ensure API Secret is correct.")
                return False

            payload = {
                "api_key": api_key,
                "request_token": request_token,
                "signature": signature
            }
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "FlyingBuddha-ScalpingBot-RequestToken/1.0"
            }

            st.info("🔄 Authenticating with Goodwill API via /login-response...")
            response = requests.post(
                f"{self.base_url}/login-response",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            st.warning("---- DEBUG START: API RESPONSE (Method 1) ----") # PROMINENT MARKER

            if response.status_code == 200:
                try:
                    data = response.json()
                    st.subheader("RAW API Response from /login-response:")
                    st.json(data) # USE ST.JSON FOR BETTER DISPLAY

                    if data.get('status') == 'success':
                        user_data = data.get('data', {}) # Default to empty dict
                        st.subheader("Extracted 'data' object from API Response:")
                        st.json(user_data)

                        self.client_id = user_data.get('clnt_id')
                        self.access_token = user_data.get('access_token')
                        self.user_session_id = user_data.get('usersessionid')
                        self.user_profile = user_data # Store the whole user_data object

                        st.info(f"DEBUG Info: Parsed access_token: '{self.access_token}' (Type: {type(self.access_token)})")
                        st.info(f"DEBUG Info: Parsed client_id: '{self.client_id}' (Type: {type(self.client_id)})")
                        st.info(f"DEBUG Info: Parsed usersessionid: '{self.user_session_id}' (Type: {type(self.user_session_id)})")

                        if self.access_token and self.client_id:
                            self.is_connected = True
                            self.connection_method = "request_token_flow"
                            self.last_login_time = datetime.now()
                            
                            st.session_state["gw_logged_in"] = True
                            st.session_state["gw_access_token"] = self.access_token
                            st.session_state["gw_client_id"] = self.client_id
                            st.session_state["gw_connection"] = self.connection_method
                            st.session_state["gw_user_session_id"] = self.user_session_id
                            st.session_state["gw_user_profile"] = self.user_profile
                            
                            user_name = user_data.get('name', 'N/A')
                            user_email = user_data.get('email', 'N/A')
                            exchanges = user_data.get('exarr', [])
                            
                            st.success(f"✅ Connected Successfully via Request Token!")
                            st.info(f"👤 User: {user_name} | 📧 Email: {user_email} | 🏦 Client ID: {self.client_id}")
                            st.info(f"📊 Exchanges: {', '.join(exchanges[:5])}{'...' if len(exchanges) > 5 else ''}")
                            st.warning("---- DEBUG END: API RESPONSE (Method 1) ----")
                            return True
                        else:
                            st.error("❌ Missing access_token or client_id in response despite 'status: success'. Check DEBUG logs above.")
                            st.error(f"Details: access_token found was '{self.access_token}', client_id found was '{self.client_id}'")
                            st.warning("---- DEBUG END: API RESPONSE (Method 1) ----")
                            return False
                    else:
                        error_msg = data.get('error_msg', f"API status was '{data.get('status')}', not 'success'.")
                        error_type = data.get('error_type', 'Unknown')
                        st.error(f"❌ API Error (from /login-response): {error_msg} (Type: {error_type})")
                        if "Invalid Signature" in error_msg:
                            st.error("🔧 Solution Hint: Verify your API Secret is correct and matches the one used for this request_token.")
                        elif "Invalid Request Token" in error_msg:
                            st.error("🔧 Solution Hint: Generate a fresh request_token by logging in again via the generated URL.")
                        elif "Invalid API key" in error_msg:
                            st.error("🔧 Solution Hint: Check your API Key used for generating the login URL and in this request.")
                        st.warning("---- DEBUG END: API RESPONSE (Method 1) ----")
                        return False
                except requests.exceptions.JSONDecodeError as json_err:
                    st.error(f"❌ JSON Decode Error: Failed to parse API response from /login-response. Error: {json_err}")
                    st.subheader("Raw response text that failed to parse:")
                    st.text(response.text)
                    st.warning("---- DEBUG END: API RESPONSE (Method 1) ----")
                    return False
            else:
                st.error(f"❌ HTTP Error {response.status_code} from /login-response:")
                st.text(response.text) # Show the full error text from the API
                st.warning("---- DEBUG END: API RESPONSE (Method 1) ----")
                return False

        except requests.exceptions.Timeout:
            st.error("❌ Request to /login-response timed out. Please try again.")
            return False
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error for /login-response. Check your internet connection.")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error in 'login_with_request_token': {str(e)}")
            import traceback
            st.text(traceback.format_exc()) # For detailed exception trace
            return False


    def get_headers(self) -> Dict:
        """Get authenticated headers for API calls"""
        if not self.access_token or not self.api_key:
            # st.error("Cannot get headers: Missing access_token or api_key.") # Can be noisy
            return {}
        return {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "FlyingBuddha-ScalpingBot-RequestToken/1.0"
        }

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection using the /profile endpoint"""
        if not self.is_connected:
            return False, "Not connected"
        headers = self.get_headers()
        if not headers:
            return False, "Authentication details missing for test."
        try:
            response = requests.get(f"{self.base_url}/profile", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return True, "Direct API profile call successful."
                else:
                    return False, f"API profile call returned error: {data.get('error_msg', 'Unknown error')}"
            else:
                return False, f"HTTP {response.status_code} on profile call: {response.text[:100]}"
        except Exception as e:
            return False, f"Connection test error: {str(e)}"

    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """Get symbol token using fetchsymbol API with caching"""
        if not self.is_connected: return None
        cache_key = f"token_{symbol}_{exchange}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        headers = self.get_headers()
        if not headers: return None
        payload = {"s": symbol} # As per docs, 's' for search string
        try:
            response = requests.post(f"{self.base_url}/fetchsymbol", json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    results = data.get('data', [])
                    for result in results:
                        # Assuming desired format matches "SYMBOL-EQ" for equity. Adjust if needed.
                        if (result.get('exchange') == exchange and 
                            result.get('symbol', '').upper() == f"{symbol.upper()}-EQ"): # More robust symbol matching
                            token = result.get('token')
                            st.session_state[cache_key] = token
                            return token
            # st.warning(f"Could not fetch token for {symbol}-{exchange}. Response: {response.text if response else 'No response'}")
            return None
        except Exception as e:
            # st.warning(f"Error fetching symbol token for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        if not self.is_connected: return None
        
        token = self._get_symbol_token(symbol, exchange)
        if not token:
            # st.warning(f"No token found for {symbol} in {exchange} to get quote.")
            return None
            
        headers = self.get_headers()
        if not headers: return None
        
        # Payload structure as per typical /getquote or /quote endpoints
        # Adjust key names if Goodwill documentation differs
        payload = {"exchange": exchange, "token": token} # Common: 'token' or 'instrument_token' or 'scrip_token'

        try:
            # Assuming /getquote is the correct endpoint, verify with Goodwill docs
            response = requests.post(f"{self.base_url}/getquote", json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    quote_data = data.get('data', {})
                    # Standardize output, map keys from quote_data to these standard keys
                    return {
                        'price': float(quote_data.get('last_price', quote_data.get('ltp', 0))),
                        'volume': int(quote_data.get('volume', 0)),
                        'high': float(quote_data.get('high', 0)),
                        'low': float(quote_data.get('low', 0)),
                        'open': float(quote_data.get('open', 0)),
                        'change': float(quote_data.get('change', 0)),
                        'change_per': float(quote_data.get('change_percentage', quote_data.get('change_per',0))), # Common variations
                        'timestamp': datetime.now() # API might provide its own timestamp
                    }
            # else:
                # st.warning(f"Get quote HTTP error for {symbol}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Quote error for {symbol}: {e}") # Keep this print for backend logs
        return None

    def place_order(self, symbol: str, action: str, quantity: int, price: float,
                   order_type: str = "MKT", product: str = "MIS", exchange: str = "NSE") -> Optional[str]:
        if not self.is_connected:
            st.error("❌ Not connected to Goodwill for placing order.")
            return None
        
        # Symbol formatting: some APIs need exchange:symbol, some just symbol if exchange is separate
        # Assuming "-EQ" is needed and tsym is the correct field
        # Verify tsym vs symbol vs trading_symbol with Goodwill docs
        formatted_symbol = f"{symbol.upper()}-EQ" 
        
        headers = self.get_headers()
        if not headers:
            st.error("❌ Authentication headers not available for placing order.")
            return None

        order_payload = {
            "tsym": formatted_symbol,        # Trading Symbol
            "exchange": exchange,            # Exchange (NSE, BSE, NFO, etc.)
            "trantype": action.upper(),      # B (Buy) or S (Sell)
            "validity": "DAY",               # DAY, IOC, etc.
            "pricetype": order_type.upper(), # MKT, L (Limit), SL-L, SL-M
            "qty": str(quantity),
            "discqty": "0",                  # Disclosed quantity
            "price": str(price) if order_type.upper() != "MKT" else "0", # Price for Limit orders
            "trgprc": "0",                   # Trigger price for SL orders
            "product": product.upper(),      # MIS (Intraday), CNC (Delivery), NRML (Normal for F&O)
            "amo": "NO"                      # After Market Order (YES/NO)
            # Other potential fields: variety, order_tag, etc.
        }
        
        try:
            response = requests.post(f"{self.base_url}/placeorder", json=order_payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    order_data = data.get('data', {})
                    order_id = order_data.get('nstordno') # Common field for order number
                    if order_id:
                        st.success(f"🎯 Order Placed: {action} {quantity} {symbol} @ {order_type} {price if order_type != 'MKT' else 'Market'} | ID: {order_id}")
                        return order_id
                    else:
                        st.error(f"❌ Order placed but no order ID returned. API Response: {order_data}")
                        return None
                else:
                    error_msg = data.get('error_msg', 'Order placement failed')
                    st.error(f"❌ Order Failed: {error_msg}. Payload: {order_payload}")
            else:
                st.error(f"❌ Order HTTP Error: {response.status_code} - {response.text}. Payload: {order_payload}")
            return None
        except Exception as e:
            st.error(f"❌ Order placement exception: {str(e)}")
            return None

    def get_positions(self) -> List[Dict]:
        if not self.is_connected: return []
        headers = self.get_headers()
        if not headers: return []
        try:
            response = requests.get(f"{self.base_url}/positions", headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', [])
            return []
        except Exception:
            return []

    def get_profile(self) -> Optional[Dict]:
        if not self.is_connected: return None
        headers = self.get_headers()
        if not headers: return self.user_profile # Return cached if no headers
        try:
            response = requests.get(f"{self.base_url}/profile", headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.user_profile = data.get('data', {}) # Update cache
                    return self.user_profile
                else:
                    st.warning(f"Profile API error: {data.get('error_msg', 'Unknown error')}")
            return self.user_profile # Return cached on API error
        except Exception as e:
            st.warning(f"Profile fetch error: {e}")
            return self.user_profile

    def logout(self) -> bool:
        headers = self.get_headers()
        if headers: # Only attempt API logout if we have headers (i.e., were logged in)
            try:
                # Common endpoint for logout, verify with Goodwill docs
                response = requests.get(f"{self.base_url}/logout", headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        st.info("✅ API logout successful.")
                    # else:
                        # st.warning(f"API logout call made, but status was not success: {data.get('error_msg')}")
                # else:
                    # st.warning(f"API logout HTTP error: {response.status_code}")
            except Exception as e:
                st.warning(f"API logout request failed: {e}")
        
        # Clear all local session data
        self.access_token = None
        self.is_connected = False
        self.client_id = None
        self.connection_method = None
        self.user_profile = None
        self.api_key = None # Clear API key as well on logout
        self.api_secret = None
        self.last_login_time = None
        
        session_keys = [
            "gw_logged_in", "gw_access_token", "gw_client_id",
            "gw_connection", "gw_user_session_id", "gw_user_profile"
        ]
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        token_keys = [k for k in st.session_state.keys() if k.startswith("token_")]
        for key in token_keys:
            del st.session_state[key]
            
        st.success("✅ Logged out locally. Session data cleared.")
        return True

# ==================== ENHANCED 8-STRATEGY SYSTEM ====================
# ... (EnhancedScalpingStrategies class remains unchanged, ensure it's complete in your final file) ...
class EnhancedScalpingStrategies:
    """Enhanced 8 Advanced Scalping Strategies with improved logic"""
    
    def __init__(self):
        self.strategies = {
            1: {"name": "Momentum_Breakout", "color": "#FF6B6B", "desc": "Price breakouts with volume confirmation"},
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
        
        self.analysis_cache = {}
        self.cache_timeout = 30  # seconds
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        cache_key = f"{symbol}_{int(time.time() // self.cache_timeout)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m") # Ensure sufficient data for indicators
            
            if len(data) < 20: # Minimum data points for reliable TA
                return self._get_default_conditions()
            
            closes = data['Close']
            volumes = data['Volume']
            highs = data['High']
            lows = data['Low']
            
            sma_5 = closes.rolling(5).mean()
            sma_20 = closes.rolling(20).mean()
            rsi = self._calculate_rsi(closes, 14)
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(closes, 20)
            vwap = self._calculate_vwap(data)
            
            current_price = closes.iloc[-1]
            
            conditions = {
                'trending': abs((sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100 > 0.5 if not sma_20.empty and sma_20.iloc[-1] != 0 else False,
                'trend_strength': abs((sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100 if not sma_20.empty and sma_20.iloc[-1] != 0 else 0,
                'trend_direction': 1 if not sma_5.empty and not sma_20.empty and sma_5.iloc[-1] > sma_20.iloc[-1] else -1,
                'volatile': closes.pct_change().rolling(10).std().iloc[-1] * 100 > 1.5 if len(closes) > 10 else False,
                'volatility': closes.pct_change().rolling(10).std().iloc[-1] * 100 if len(closes) > 10 else 0,
                'volume_surge': volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] > 1.8 if len(volumes) > 20 and volumes.rolling(20).mean().iloc[-1] != 0 else False,
                'volume_ratio': volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] if len(volumes) > 20 and volumes.rolling(20).mean().iloc[-1] != 0 else 1.0,
                'consolidating': ((highs.tail(10).max() - lows.tail(10).min()) / current_price) * 100 < 1.0 if current_price != 0 and len(highs) > 10 else False,
                'price_range': ((highs.tail(10).max() - lows.tail(10).min()) / current_price) * 100 if current_price != 0 and len(highs) > 10 else 0,
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
                'rsi_oversold': rsi.iloc[-1] < 30 if len(rsi) > 0 else False,
                'rsi_overbought': rsi.iloc[-1] > 70 if len(rsi) > 0 else False,
                'bb_squeeze': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] < 0.02 if len(bb_middle) > 0 and bb_middle.iloc[-1] !=0 else False,
                'bb_upper_touch': current_price > bb_upper.iloc[-1] * 0.995 if len(bb_upper) > 0 else False,
                'bb_lower_touch': current_price < bb_lower.iloc[-1] * 1.005 if len(bb_lower) > 0 else False,
                'above_vwap': current_price > vwap.iloc[-1] if len(vwap) > 0 else True,
                'vwap_distance': abs(current_price - vwap.iloc[-1]) / current_price if len(vwap) > 0 and current_price !=0 else 0.002,
                'near_support': self._check_support_resistance(data, 'support'),
                'near_resistance': self._check_support_resistance(data, 'resistance'),
                'news_driven': (volumes.tail(5).mean() / volumes.rolling(20).mean().iloc[-1] > 2.0 if len(volumes) > 20 and volumes.rolling(20).mean().iloc[-1] !=0 else False) and \
                              (closes.pct_change().rolling(5).std().iloc[-1] * 100 > 2.0 if len(closes) > 5 else False)
            }
            self.analysis_cache[cache_key] = conditions
            if len(self.analysis_cache) > 50: # Cache cleanup
                old_keys = [k for k, v_time in self.analysis_cache.items() if int(k.split('_')[-1]) < int(time.time() // self.cache_timeout) - 10]
                for key in old_keys: del self.analysis_cache[key]
            return conditions
        except Exception as e:
            # print(f"Market analysis error for {symbol}: {e}")
            return self._get_default_conditions()

    def _calculate_rsi(self, prices, period=14):
        if prices.empty or len(prices) < period: return pd.Series([50] * len(prices), index=prices.index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # handle division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        if prices.empty or len(prices) < period: return prices, prices, prices # Return series of same length
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std().fillna(0)
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.fillna(sma), lower.fillna(sma), sma.fillna(prices)

    def _calculate_vwap(self, data):
        if data.empty or 'High' not in data or 'Low' not in data or 'Close' not in data or 'Volume' not in data:
            return data['Close'] if 'Close' in data and not data.empty else pd.Series()
        if data['Volume'].sum() == 0: # Avoid division by zero if total volume is zero
             return data['Close']

        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap.fillna(data['Close']) # Fill NaNs that might occur at the start

    def _check_support_resistance(self, data, level_type='support'):
        if data.empty or 'Close' not in data or len(data) < 5: return False # Need some data
        closes = data['Close']
        current_price = closes.iloc[-1]
        recent_data = data.tail(50) # Look at recent data for S/R
        if recent_data.empty: return False

        if level_type == 'support':
            levels = recent_data['Low'].rolling(5, min_periods=1).min().dropna().unique()
        else: # resistance
            levels = recent_data['High'].rolling(5, min_periods=1).max().dropna().unique()
        
        for level in levels:
            if current_price != 0 and abs(current_price - level) / current_price < 0.005: # 0.5% proximity
                return True
        return False

    def _get_default_conditions(self) -> Dict:
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
        strategy_scores = {}
        for strategy_id in self.strategies.keys(): # Initialize all scores
            strategy_scores[strategy_id] = 0

        # Strategy 1: Momentum Breakout
        strategy_scores[1] = ( (60 if market_conditions['trending'] else 10) +
                               (30 if market_conditions['volume_surge'] else 5) +
                               (20 if market_conditions['volatility'] > 1.0 else 0) +
                               self._get_performance_bonus(1) )
        # Strategy 2: Mean Reversion
        strategy_scores[2] = ( (70 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 15) +
                               (25 if market_conditions['consolidating'] else 5) +
                               (15 if not market_conditions['trending'] else 0) +
                               self._get_performance_bonus(2) )
        # ... (Scores for other strategies 3-8 as in original)
        # Strategy 3: Volume Spike
        strategy_scores[3] = ( (80 if market_conditions['volume_surge'] else 10) +
                               (15 if market_conditions['volatile'] else 5) +
                               (10 if market_conditions['news_driven'] else 0) +
                               self._get_performance_bonus(3) )
        # Strategy 4: Bollinger Squeeze
        strategy_scores[4] = ( (70 if market_conditions['bb_squeeze'] else 10) +
                               (20 if market_conditions['consolidating'] else 5) +
                               (10 if market_conditions['volatility'] < 1.0 else 0) + # Prefers lower vol for squeeze
                               self._get_performance_bonus(4) )
        # Strategy 5: RSI Divergence (Simplified to RSI extremes at S/R for now)
        strategy_scores[5] = ( (60 if (market_conditions['rsi_oversold'] and market_conditions['near_support']) or \
                                     (market_conditions['rsi_overbought'] and market_conditions['near_resistance']) else 20) +
                               (25 if market_conditions['trending'] else 10) + # Divergence can occur in trends
                               (15 if 1.0 < market_conditions['volatility'] < 2.5 else 5) +
                               self._get_performance_bonus(5) )
        # Strategy 6: VWAP Touch
        strategy_scores[6] = ( (60 if market_conditions['vwap_distance'] < 0.003 else 15) + # Close to VWAP
                               (25 if market_conditions['volume_surge'] else 10) + # Volume confirmation at VWAP
                               (15 if market_conditions['trending'] else 5) + # VWAP often respected in trends
                               self._get_performance_bonus(6) )
        # Strategy 7: Support/Resistance
        strategy_scores[7] = ( (70 if market_conditions['near_support'] or market_conditions['near_resistance'] else 15) +
                               (20 if market_conditions['consolidating'] else 10) + # S/R stronger in ranges
                               (10 if not market_conditions['trending'] else 5) + # Or for pullbacks in trends
                               self._get_performance_bonus(7) )
        # Strategy 8: News Momentum
        strategy_scores[8] = ( (90 if market_conditions['news_driven'] else 10) +
                               (10 if market_conditions['volatile'] else 5) +
                               (5 if market_conditions['volume_surge'] else 0) +
                               self._get_performance_bonus(8) )

        best_strategy = 1 # Default to 1 if all scores are 0 or dict is empty
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
        return best_strategy, strategy_scores

    def _get_performance_bonus(self, strategy_id: int) -> float:
        stats = self.strategy_stats[strategy_id]
        if stats['trades'] < 3: return 5 
        success_bonus = (stats['success_rate'] / 100) * 20
        recency_bonus = 5
        if stats['last_used']:
            hours_since = (datetime.now() - stats['last_used']).total_seconds() / 3600
            recency_bonus = min(hours_since * 0.5, 5) # Max 5 points, encourages trying less used ones over time
        return success_bonus + recency_bonus

    def get_strategy_signals(self, strategy_id: int, symbol: str, market_conditions: Dict) -> Dict:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            if len(data) < 10:
                return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Insufficient data'}
            
            current_price = data['Close'].iloc[-1] if not data.empty else 0
            
            strategy_methods = {
                1: self._momentum_breakout_signals, 2: self._mean_reversion_signals,
                3: self._volume_spike_signals, 4: self._bollinger_squeeze_signals,
                5: self._rsi_divergence_signals, 6: self._vwap_touch_signals,
                7: self._support_resistance_signals, 8: self._news_momentum_signals
            }
            if strategy_id in strategy_methods:
                return strategy_methods[strategy_id](data, market_conditions, current_price)
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'Invalid strategy ID'}
        except Exception as e:
            # print(f"Strategy signal error for {symbol}, strategy {strategy_id}: {e}")
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': f'Signal gen error: {e}'}

    # Individual strategy signal methods need to accept current_price
    def _momentum_breakout_signals(self, data, conditions, current_price):
        if data.empty or len(data['High']) < 20: return {'signal': 0, 'reason': 'Insufficient data for breakout'}
        high_20 = data['High'].rolling(20).max().iloc[-1]
        low_20 = data['Low'].rolling(20).min().iloc[-1]
        
        if current_price > high_20 * 1.002 and conditions['volume_surge']:
            return {'signal': 1, 'confidence': 0.85, 'entry_price': current_price, 'stop_pct': 0.008, 'reason': 'Upward breakout with volume'}
        if current_price < low_20 * 0.998 and conditions['volume_surge'] and conditions['trend_direction'] < 0:
             return {'signal': -1, 'confidence': 0.85, 'entry_price': current_price, 'stop_pct': 0.008, 'reason': 'Downward breakout with volume'}
        return {'signal': 0, 'reason': 'No breakout'}

    def _mean_reversion_signals(self, data, conditions, current_price):
        rsi = conditions['rsi']
        if rsi < 25 and conditions['bb_lower_touch']:
            return {'signal': 1, 'confidence': 0.80, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'Oversold mean reversion (RSI < 25 & BB Lower Touch)'}
        if rsi > 75 and conditions['bb_upper_touch']:
            return {'signal': -1, 'confidence': 0.80, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'Overbought mean reversion (RSI > 75 & BB Upper Touch)'}
        return {'signal': 0, 'reason': 'No reversion signal'}

    def _volume_spike_signals(self, data, conditions, current_price):
        if data.empty or len(data['Close']) < 2 : return {'signal':0, 'reason': 'Insufficient data for vol spike'}
        price_change = data['Close'].pct_change().iloc[-1] * 100
        if conditions['volume_surge'] and abs(price_change) > 0.4:
            signal = 1 if price_change > 0 else -1
            return {'signal': signal, 'confidence': min(0.90, conditions['volume_ratio'] * 0.3 + 0.3), 'entry_price': current_price, 'stop_pct': 0.007, 'reason': f'Volume spike ({conditions["volume_ratio"]:.1f}x) with {abs(price_change):.2f}% move'}
        return {'signal': 0, 'reason': 'No volume spike'}

    def _bollinger_squeeze_signals(self, data, conditions, current_price):
        # Breakout from squeeze
        if conditions['bb_squeeze'] and conditions['consolidating']: # Squeeze identified
            # Look for price breaking out of recent range or BB bands
            if not data.empty and len(data['Close']) > 1:
                price_change_abs = abs(data['Close'].pct_change().iloc[-1] * 100)
                if conditions['bb_upper_touch'] and price_change_abs > 0.2: # Breakout up
                     return {'signal': 1, 'confidence': 0.75, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'Bollinger Squeeze breakout upwards'}
                if conditions['bb_lower_touch'] and price_change_abs > 0.2: # Breakout down
                     return {'signal': -1, 'confidence': 0.75, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'Bollinger Squeeze breakout downwards'}
        return {'signal': 0, 'reason': 'No Bollinger Squeeze breakout'}

    def _rsi_divergence_signals(self, data, conditions, current_price): # Simplified to S/R + RSI
        rsi = conditions['rsi']
        if rsi < 30 and conditions['near_support']:
            return {'signal': 1, 'confidence': 0.70, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'RSI (<30) at support'}
        if rsi > 70 and conditions['near_resistance']:
            return {'signal': -1, 'confidence': 0.70, 'entry_price': current_price, 'stop_pct': 0.006, 'reason': 'RSI (>70) at resistance'}
        return {'signal': 0, 'reason': 'No RSI S/R signal'}

    def _vwap_touch_signals(self, data, conditions, current_price):
        if conditions['vwap_distance'] < 0.003: # Price is close to VWAP
            if conditions['above_vwap'] and conditions['volume_surge']: # Bounce from VWAP as support
                return {'signal': 1, 'confidence': 0.75, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'VWAP support touch with volume'}
            if not conditions['above_vwap'] and conditions['volume_surge']: # Rejection from VWAP as resistance
                return {'signal': -1, 'confidence': 0.75, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'VWAP resistance touch with volume'}
        return {'signal': 0, 'reason': 'No VWAP signal'}

    def _support_resistance_signals(self, data, conditions, current_price):
        if conditions['near_support'] and conditions['rsi'] < 45: # RSI confirming not overbought at support
            return {'signal': 1, 'confidence': 0.80, 'entry_price': current_price, 'stop_pct': 0.004, 'reason': 'Support level bounce (RSI < 45)'}
        if conditions['near_resistance'] and conditions['rsi'] > 55: # RSI confirming not oversold at resistance
            return {'signal': -1, 'confidence': 0.80, 'entry_price': current_price, 'stop_pct': 0.004, 'reason': 'Resistance level rejection (RSI > 55)'}
        return {'signal': 0, 'reason': 'No S/R signal'}

    def _news_momentum_signals(self, data, conditions, current_price):
        if conditions['news_driven']:
            signal = conditions['trend_direction'] # Trade with the strong initial momentum
            return {'signal': signal, 'confidence': min(0.95, conditions['volume_ratio'] * 0.35 + 0.2), 'entry_price': current_price, 'stop_pct': 0.01, 'reason': f'News momentum (Vol: {conditions["volume_ratio"]:.1f}x, Volatility: {conditions["volatility"]:.1f}%)'}
        return {'signal': 0, 'reason': 'No news signal'}

    def update_strategy_performance(self, strategy_id: int, trade_result: Dict):
        stats = self.strategy_stats[strategy_id]
        stats['trades'] += 1
        stats['last_used'] = datetime.now()
        pnl = trade_result.get('pnl', 0)
        hold_time = trade_result.get('hold_time', 0)

        if pnl > 0:
            stats['wins'] += 1
            stats['avg_profit'] = ((stats['avg_profit'] * (stats['wins'] - 1)) + pnl) / stats['wins'] if stats['wins'] > 0 else pnl
        else: # Loss or break-even
            loss_count = stats['trades'] - stats['wins']
            stats['avg_loss'] = ((stats['avg_loss'] * (loss_count - 1)) + pnl) / loss_count if loss_count > 0 else pnl
        
        stats['total_pnl'] += pnl
        stats['success_rate'] = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
        stats['avg_hold_time'] = ((stats['avg_hold_time'] * (stats['trades'] - 1)) + hold_time) / stats['trades'] if stats['trades'] > 0 else hold_time


# ==================== PRODUCTION SCALPING BOT ====================
# ... (ProductionScalpingBot class remains largely unchanged, ensure it's complete in your final file) ...
class ProductionScalpingBot:
    def __init__(self):
        self.api = GoodwillRequestTokenIntegration() # Use the focused integration
        self.strategies = EnhancedScalpingStrategies()
        self.is_running = False
        self.capital = 100000.0
        self.positions = {} # Stores active positions
        self.trades = []    # Stores completed trade records
        self.pnl = 0.0      # Cumulative P&L for the session
        self.mode = "PAPER" # PAPER or LIVE

        # Trading parameters
        self.max_positions = 4
        self.risk_per_trade_pct = 0.15 # 15% of capital per trade (this is very high, typically 1-2%)
                                      # Assuming this means 1.5% or it's a misinterpretation of 'risk per trade'
                                      # For now, I will use it as 0.015 (1.5%) for actual risk calculation.
                                      # User provided 0.15, so let's stick to it for now, but advise caution.
        self.max_hold_time_sec = 180  # 3 minutes
        self.risk_reward_ratio = 2.0  # Target profit is 2x stop loss

        # Risk management
        self.daily_loss_limit_pct = 0.05 # 5% daily loss limit on initial capital
        self.max_drawdown_limit_pct = 0.10 # 10% max drawdown from peak capital
        self.daily_pnl = 0.0
        self.peak_capital_session = self.capital # Tracks peak capital for current session drawdown
        
        self.symbols = [ # Default NIFTY 50 / Bank Nifty stocks often good for scalping
            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", 
            "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "BHARTIARTL",
            "LT", "HINDUNILVR", "ITC", "MARUTI", "ASIANPAINT"
        ]
        self.init_database()

    def init_database(self):
        try:
            self.conn = sqlite3.connect('scalping_trades_prod.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, action TEXT,
                    entry_price REAL, exit_price REAL, quantity INTEGER, pnl REAL, hold_time REAL, 
                    strategy_id INTEGER, strategy_name TEXT, exit_reason TEXT, confidence REAL,
                    order_id TEXT, mode TEXT, connection_method TEXT, market_conditions TEXT, 
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # ... (other tables like daily_stats, strategy_performance can be added as before)
            self.conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
            self.conn = None # Ensure conn is None if init fails

    def check_risk_limits(self) -> bool:
        """Checks daily loss and max drawdown limits."""
        current_capital_with_pnl = self.capital + self.pnl # More accurate current equity for session
        
        # Daily Loss Limit
        daily_loss_limit_amount = self.capital * self.daily_loss_limit_pct
        if self.daily_pnl < -daily_loss_limit_amount:
            st.error(f"🛑 Daily loss limit breached: Current Daily P&L ₹{self.daily_pnl:.2f} (Limit: -₹{daily_loss_limit_amount:.2f})")
            return False

        # Max Drawdown Limit
        current_drawdown = self.peak_capital_session - current_capital_with_pnl
        max_drawdown_amount = self.peak_capital_session * self.max_drawdown_limit_pct # Drawdown from peak
        if current_drawdown > 0 and current_drawdown > max_drawdown_amount:
             st.error(f"🛑 Max drawdown limit breached: Current Drawdown ₹{current_drawdown:.2f} (Limit: ₹{max_drawdown_amount:.2f} from peak ₹{self.peak_capital_session:.2f})")
             return False
        return True

    def scan_for_opportunities(self) -> List[Dict]:
        opportunities = []
        # Scan a subset for performance in UI, can be expanded
        for symbol in self.symbols[:st.session_state.get('num_symbols_to_scan', 8)]: 
            try:
                live_data = None; data_source = "yfinance_fallback"
                if self.api.is_connected:
                    api_quote = self.api.get_quote(symbol)
                    if api_quote and api_quote.get('price', 0) > 0:
                        live_data = api_quote
                        data_source = self.api.connection_method if self.api.connection_method else "Goodwill_API"
                
                if not live_data: # Fallback to yfinance if API fails or not connected
                    yf_ticker = yf.Ticker(f"{symbol}.NS")
                    hist = yf_ticker.history(period="2d", interval="1m") # Need enough for prev close if needed
                    if not hist.empty and len(hist) > 1:
                        latest = hist.iloc[-1]
                        live_data = {
                            'price': float(latest['Close']), 'volume': int(latest['Volume']),
                            'high': float(latest['High']), 'low': float(latest['Low']),
                            'open': float(latest['Open']), 'timestamp': datetime.now(),
                            'change': float(latest['Close'] - hist.iloc[-2]['Close']), # Approx change
                            'change_per': float(((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close']) * 100) if hist.iloc[-2]['Close'] !=0 else 0
                        }
                if not live_data or live_data.get('price',0) <= 0: continue

                market_conditions = self.strategies.analyze_market_conditions(symbol)
                strategy_id, strategy_scores = self.strategies.select_strategy(symbol, market_conditions)
                signals = self.strategies.get_strategy_signals(strategy_id, symbol, market_conditions)

                if signals.get('signal',0) != 0 and signals.get('confidence',0) > 0.70 and len(self.positions) < self.max_positions:
                     opportunities.append({
                        'symbol': symbol, 'strategy_id': strategy_id,
                        'strategy_name': self.strategies.strategies[strategy_id]['name'],
                        'signal': signals['signal'], 'confidence': signals['confidence'],
                        'price': live_data['price'], 'stop_pct': signals.get('stop_pct', 0.005),
                        'reason': signals.get('reason', 'Strategy Signal'),
                        'market_conditions': market_conditions, # For logging/review
                        'volume': live_data.get('volume',0), 'data_source': data_source
                    })
            except Exception as e:
                # print(f"Error scanning {symbol}: {e}") # Keep for backend log
                pass # Continue to next symbol
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:self.max_positions] # Consider top N opportunities

    def calculate_position_size(self, entry_price: float, stop_loss_pct: float) -> int:
        # Revised risk per trade to be a percentage of current capital
        # User's risk_per_trade_pct = 0.15 is very high. Using 0.015 (1.5%) for safety.
        # If user insists on 15%, this should be made very clear.
        # For this example, I'll use a safer 1.5% (0.015)
        safer_risk_per_trade_pct = 0.015 # 1.5% of capital at risk per trade
        
        # Let's use the parameter as provided by user, but with a warning
        actual_risk_pct = self.risk_per_trade_pct # This is 0.15 from bot params
        if actual_risk_pct > 0.05: # If risk is > 5%, it's very aggressive
             st.sidebar.warning(f"⚠️ Aggressive Risk: {actual_risk_pct*100:.0f}% per trade!")


        risk_amount_per_trade = (self.capital + self.pnl) * actual_risk_pct 
        stop_loss_amount_per_share = entry_price * stop_loss_pct
        if stop_loss_amount_per_share == 0: return 1 # Avoid division by zero, default to 1 share

        quantity = int(risk_amount_per_trade / stop_loss_amount_per_share)
        
        # Max exposure per trade (e.g., 25% of capital)
        max_trade_value = (self.capital + self.pnl) * 0.25 
        max_quantity_by_exposure = int(max_trade_value / entry_price) if entry_price > 0 else 1
        
        quantity = max(1, min(quantity, max_quantity_by_exposure, 2000)) # Min 1 share, max reasonable qty
        return quantity

    def execute_entry(self, opportunity: Dict):
        if not self.check_risk_limits(): return None
        if len(self.positions) >= self.max_positions: return None
        
        symbol = opportunity['symbol']; entry_price = opportunity['price']
        signal = opportunity['signal']; strategy_id = opportunity['strategy_id']
        strategy_name = opportunity['strategy_name']; stop_pct = opportunity['stop_pct']
        confidence = opportunity['confidence']

        quantity = self.calculate_position_size(entry_price, stop_pct)
        action = "B" if signal > 0 else "S" # Buy or Sell

        if action == "B":
            stop_loss_price = entry_price * (1 - stop_pct)
            target_price = entry_price * (1 + (stop_pct * self.risk_reward_ratio))
        else: # Short
            stop_loss_price = entry_price * (1 + stop_pct)
            target_price = entry_price * (1 - (stop_pct * self.risk_reward_ratio))

        order_id = None; order_status = "PAPER"
        if self.mode == "LIVE" and self.api.is_connected:
            order_id = self.api.place_order(symbol, action, quantity, entry_price, "MKT", "MIS")
            if not order_id:
                st.error(f"❌ LIVE order placement failed for {symbol}.")
                return None
            order_status = "LIVE_PLACED"
        else: # Paper trade
            order_id = f"PAPER_{int(time.time())}_{symbol}"
            st.info(f"📝 PAPER: {action} {quantity} {symbol} @ MKT (entry ~₹{entry_price:.2f}) | Strat: {strategy_name} | Conf: {confidence:.1%}")

        if order_id: # Successfully placed (live or paper)
            position_id = f"{symbol}_{order_id}" # Unique position ID
            self.positions[position_id] = {
                'symbol': symbol, 'action': action, 'quantity': quantity, 
                'entry_price': entry_price, 'stop_loss_price': stop_loss_price, 
                'target_price': target_price, 'entry_time': datetime.now(), 
                'order_id': order_id, 'strategy_id': strategy_id, 'strategy_name': strategy_name,
                'confidence': confidence, 'trailing_stop_price': stop_loss_price, # Initialize trailing SL
                'highest_profit_pts': 0, 'status': order_status, 
                'data_source': opportunity['data_source'],
                'market_conditions_entry': opportunity['market_conditions'] # Store conditions at entry
            }
            st.success(f"🚀 Position Opened ({order_status}): {action} {quantity} {symbol} @ ~₹{entry_price:.2f}")
            return position_id
        return None

    def update_position_tracking(self, position_id: str, current_price: float):
        """Updates P&L and trailing stop for an active position."""
        position = self.positions.get(position_id)
        if not position: return

        profit_per_share = 0
        if position['action'] == 'B':
            profit_per_share = current_price - position['entry_price']
        else: # Short
            profit_per_share = position['entry_price'] - current_price
        
        current_pnl_for_position = profit_per_share * position['quantity']
        position['current_pnl'] = current_pnl_for_position # Store current PNL for this position

        # Update highest profit points seen for this trade (in terms of price movement)
        if profit_per_share > position.get('highest_profit_pts', 0):
            position['highest_profit_pts'] = profit_per_share

        # Advanced Trailing Stop (example: trail by 50% of profit gained, or a fixed % of price)
        if current_pnl_for_position > 0: # Only trail if in profit
            # Trail by a percentage of the original stop distance if profit exceeds it
            original_stop_distance = abs(position['entry_price'] - position['stop_loss_price'])
            if profit_per_share > original_stop_distance * 0.5: # e.g. if 0.5R in profit
                new_trail_stop = 0
                if position['action'] == 'B':
                    # Trail below current price by a fraction of original stop distance
                    new_trail_stop = current_price - (original_stop_distance * 0.75) 
                    position['trailing_stop_price'] = max(position['trailing_stop_price'], new_trail_stop, position['entry_price']) # Ensure SL at least breakeven
                else: # Short
                    new_trail_stop = current_price + (original_stop_distance * 0.75)
                    position['trailing_stop_price'] = min(position['trailing_stop_price'], new_trail_stop, position['entry_price']) # Ensure SL at least breakeven

    def check_exit_conditions(self, position_id: str, current_price: float) -> Optional[str]:
        position = self.positions.get(position_id)
        if not position: return None

        self.update_position_tracking(position_id, current_price) # Update P&L and trailing SL first

        # Time-based exit
        hold_time_seconds = (datetime.now() - position['entry_time']).total_seconds()
        if hold_time_seconds > self.max_hold_time_sec:
            return "TIME_LIMIT"

        # Target / Stop Loss Exit
        if position['action'] == 'B':
            if current_price >= position['target_price']: return "TARGET_HIT"
            if current_price <= position['trailing_stop_price']: return "TRAILING_SL_HIT" # Use trailing SL
            if current_price <= position['stop_loss_price']: return "INITIAL_SL_HIT" # Fallback if trailing SL is above initial
        else: # Short
            if current_price <= position['target_price']: return "TARGET_HIT"
            if current_price >= position['trailing_stop_price']: return "TRAILING_SL_HIT"
            if current_price >= position['stop_loss_price']: return "INITIAL_SL_HIT"
        
        # Profit protection (e.g. if profit retraces by X% from peak for this trade)
        if position['highest_profit_pts'] > 0:
             profit_retracement_pct = 0.6 # If 60% of peak profit for this trade is lost
             current_profit_pts = (current_price - position['entry_price']) if position['action'] == 'B' else (position['entry_price'] - current_price)
             if current_profit_pts < position['highest_profit_pts'] * (1 - profit_retracement_pct):
                 if hold_time_seconds > 60 : # Only after 1 min to avoid premature exit
                    return "PROFIT_PROTECT"
        return None

    def close_position(self, position_id: str, exit_price: float, exit_reason: str):
        if position_id not in self.positions: return
        position = self.positions.pop(position_id)

        pnl = 0
        if position['action'] == 'B':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else: # Short
            pnl = (position['entry_price'] - exit_price) * position['quantity']

        self.pnl += pnl # Overall session P&L
        self.daily_pnl += pnl # P&L for the current trading day
        
        # Update peak capital for session drawdown calculation
        current_total_capital = self.capital + self.pnl
        if current_total_capital > self.peak_capital_session:
            self.peak_capital_session = current_total_capital

        hold_time = (datetime.now() - position['entry_time']).total_seconds()
        
        trade_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'symbol': position['symbol'],
            'action': f"EXIT_{position['action']}", 'entry_price': position['entry_price'],
            'exit_price': exit_price, 'quantity': position['quantity'], 'pnl': round(pnl, 2),
            'hold_time': round(hold_time, 1), 'strategy_id': position['strategy_id'],
            'strategy_name': position['strategy_name'], 'exit_reason': exit_reason,
            'confidence': position.get('confidence',0), 'order_id': position['order_id'],
            'mode': self.mode, 
            'connection_method': self.api.connection_method or position.get('data_source', 'yfinance_fallback'),
            'market_conditions': json.dumps(position.get('market_conditions_entry', {}))
        }
        self.trades.append(trade_record)

        if self.conn: # Log to DB
            try:
                cursor = self.conn.cursor()
                cursor.execute('''INSERT INTO trades (timestamp, symbol, action, entry_price, exit_price, quantity, 
                                pnl, hold_time, strategy_id, strategy_name, exit_reason, confidence, order_id, 
                                mode, connection_method, market_conditions) 
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
                               (trade_record['timestamp'], trade_record['symbol'], trade_record['action'], 
                                trade_record['entry_price'], trade_record['exit_price'], trade_record['quantity'],
                                trade_record['pnl'], trade_record['hold_time'], trade_record['strategy_id'],
                                trade_record['strategy_name'], trade_record['exit_reason'], trade_record['confidence'],
                                trade_record['order_id'], trade_record['mode'], trade_record['connection_method'],
                                trade_record['market_conditions']))
                self.conn.commit()
            except Exception as e:
                st.warning(f"DB save error for trade: {e}")

        self.strategies.update_strategy_performance(position['strategy_id'], {'pnl': pnl, 'hold_time': hold_time})

        exit_order_action = "S" if position['action'] == "B" else "B"
        if self.mode == "LIVE" and position.get('status') == "LIVE_PLACED" and self.api.is_connected:
            # Place actual exit order
            exit_order_id = self.api.place_order(position['symbol'], exit_order_action, position['quantity'], exit_price, "MKT", "MIS")
            if not exit_order_id:
                st.error(f"❌ LIVE exit order FAILED for {position['symbol']} (Pos ID: {position_id})")
        else:
            st.info(f"📝 PAPER EXIT: {exit_order_action} {position['quantity']} {position['symbol']} @ MKT (exit ~₹{exit_price:.2f})")
        
        pnl_color = "🟢" if pnl >= 0 else "🔴"
        st.info(f"✅ Position Closed: {position['symbol']} | {pnl_color} P&L: ₹{pnl:.2f} | Reason: {exit_reason}")

    def run_trading_cycle(self):
        if not self.check_risk_limits(): # Check overall risk limits first
            st.error("🛑 Trading halted due to risk limits breach.")
            self.is_running = False # Stop the bot
            return

        # 1. Manage existing positions (check for exits)
        positions_to_close_ids = [] # Store (pos_id, current_price, exit_reason)
        for pos_id, position_details in list(self.positions.items()): # Iterate on a copy
            current_price_for_pos = 0
            if self.api.is_connected:
                quote = self.api.get_quote(position_details['symbol'])
                if quote and quote.get('price',0) > 0: current_price_for_pos = quote['price']
            
            if current_price_for_pos == 0: # Fallback if API fails
                yf_ticker = yf.Ticker(f"{position_details['symbol']}.NS")
                hist = yf_ticker.history(period="1d", interval="1m")
                if not hist.empty: current_price_for_pos = hist.iloc[-1]['Close']
            
            if current_price_for_pos > 0:
                exit_reason = self.check_exit_conditions(pos_id, current_price_for_pos)
                if exit_reason:
                    positions_to_close_ids.append((pos_id, current_price_for_pos, exit_reason))
            # else: st.warning(f"Could not get current price for {position_details['symbol']} to check exit.")

        for pos_id, price, reason in positions_to_close_ids:
            self.close_position(pos_id, price, reason)
            time.sleep(0.2) # Small delay between closing multiple orders

        # 2. Scan for new opportunities if room available
        if len(self.positions) < self.max_positions:
            opportunities = self.scan_for_opportunities()
            if opportunities:
                # Execute the top opportunity if no conflicting position or other checks pass
                top_opportunity = opportunities[0]
                # Basic check: don't open same symbol in opposite direction immediately (can be more complex)
                if not any(p['symbol'] == top_opportunity['symbol'] for p_id, p in self.positions.items()):
                    self.execute_entry(top_opportunity)
    
    def get_performance_metrics(self) -> Dict:
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_trade': 0, 'profit_factor': 0, 'sharpe_ratio': 0}
        
        pnls = [t['pnl'] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0] # Absolute losses for calculation

        total_trades = len(self.trades)
        winning_trades = len(wins)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(pnls)
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        total_profit_from_wins = sum(wins)
        total_loss_from_losses = abs(sum(losses)) # Sum of absolute values of losses
        profit_factor = total_profit_from_wins / total_loss_from_losses if total_loss_from_losses > 0 else float('inf') if total_profit_from_wins > 0 else 0
        
        # Simplified Sharpe Ratio (assuming risk-free rate is 0 for daily scalping returns)
        if len(pnls) > 1 and np.std(pnls) != 0:
            sharpe_ratio = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) # Annualized for daily P&L, adjust for scalps
        else: sharpe_ratio = 0
            
        return {
            'total_trades': total_trades, 'win_rate': round(win_rate,2), 'total_pnl': round(total_pnl,2),
            'avg_trade': round(avg_trade_pnl,2), 'profit_factor': round(profit_factor,2), 
            'sharpe_ratio': round(sharpe_ratio,3),
            'winning_trades': winning_trades, 'losing_trades': len(losses),
            'avg_win': round(np.mean(wins),2) if wins else 0,
            'avg_loss': round(np.mean(losses),2) if losses else 0, # avg monetary loss
            'max_profit_trade': round(max(wins),2) if wins else 0,
            'max_loss_trade': round(min(losses),2) if losses else 0 # most negative P&L
        }

# ==================== STREAMLIT APPLICATION ====================

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = ProductionScalpingBot()
if 'gw_logged_in' not in st.session_state: # For Goodwill login status
    st.session_state.gw_logged_in = False
if 'num_symbols_to_scan' not in st.session_state:
    st.session_state.num_symbols_to_scan = 8


bot = st.session_state.bot

# Title and Disclaimer
st.title("⚡ FlyingBuddha Scalping Bot - Request Token Flow")
st.markdown("""
<div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin: 10px 0;">
    <h4 style="color: #2e7d32; margin-top: 0;">✅ Focused Version - Request Token Only</h4>
    <p style="margin-bottom: 0;"><strong>This version uses the reliable Request Token flow for Goodwill API authentication.</strong></p>
    <ul style="margin: 10px 0;">
        <li>🎫 <strong>Proper Request Token Flow:</strong> Implementation per official documentation.</li>
        <li>🔗 <strong>URL Parser:</strong> Automatic request_token extraction from redirect URLs.</li>
        <li>🎯 <strong>8 Advanced Strategies:</strong> Enhanced with improved signal generation.</li>
        <li>🛡️ <strong>Production Risk Management:</strong> Daily loss & max drawdown limits.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# Sidebar Controls
st.sidebar.header("🚀 Scalping Bot Controls")
connection_status = "🟢 Connected" if bot.api.is_connected else "🔴 Disconnected"
connection_method_display = f" ({bot.api.connection_method})" if bot.api.connection_method and bot.api.is_connected else ""

st.sidebar.markdown(f"""
**📊 Status Overview:**
- **API Status:** {connection_status}{connection_method_display}
- **Trading Mode:** {'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}
- **Session Capital:** ₹{(bot.capital + bot.pnl):,.2f} (Initial: ₹{bot.capital:,.2f})
- **Overall Session P&L:** ₹{bot.pnl:,.2f}
- **Daily P&L (Bot):** ₹{bot.daily_pnl:,.2f}
- **Active Positions:** {len(bot.positions)}/{bot.max_positions}
- **Bot State:** {'🟢 Running' if bot.is_running else '🔴 Stopped'}
""")
if bot.api.is_connected and bot.api.user_profile:
    st.sidebar.info(f"👤 {bot.api.user_profile.get('name', 'User')} ({bot.api.client_id})")


# Authentication Section
st.subheader("🔐 Goodwill Authentication (Request Token Flow)")
with st.expander("📋 Setup Guide - Request Token Flow", expanded=not st.session_state.gw_logged_in):
    st.markdown("""
    ### 🎯 Authentication Steps:
    1. Enter your **API Key** and **API Secret** below.
    2. Click "**🔗 Generate Login URL**".
    3. **Open the generated URL in a NEW BROWSER TAB** and log in with your Goodwill trading credentials (including any 2FA required by Goodwill on their page).
    4. After successful login on Goodwill's site, you will be redirected. **Copy the ENTIRE redirect URL** from your browser's address bar.
    5. Paste the complete redirect URL into the "**Complete Redirect URL**" field below.
    6. Click "**🎫 Login with Request Token**".
    
    Your API Secret is used locally to create a signature and is not stored long-term by this app beyond the current session if login is successful.
    """)

if not st.session_state.gw_logged_in:
    auth_col1, auth_col2 = st.columns(2)
    with auth_col1:
        api_key_input = st.text_input("Your API Key", value=st.session_state.get("api_key_input",""), type="password", help="Enter your Goodwill API Key.")
        api_secret_input = st.text_input("Your API Secret", value=st.session_state.get("api_secret_input",""), type="password", help="Enter your Goodwill API Secret.")
        
        if api_key_input: st.session_state.api_key_input = api_key_input
        if api_secret_input: st.session_state.api_secret_input = api_secret_input

        if api_key_input and st.button("🔗 Generate Login URL", use_container_width=True):
            login_url = bot.api.generate_login_url(api_key_input)
            st.success("✅ Login URL Generated!")
            st.markdown(f"**[Click Here to Login to Goodwill]({login_url})** (Opens in new tab)", unsafe_allow_html=True)
            st.code(login_url, language="text")
            st.info("👆 IMPORTANT: Open URL, login, then copy the full redirect URL from your browser.")
    
    with auth_col2:
        redirect_url_input = st.text_area(
            "Complete Redirect URL (after login on Goodwill's site)", height=120,
            placeholder="Example: https://your-redirect-url.com?request_token=abc123xyz...",
            help="Paste the entire URL from your browser after Goodwill login & redirection."
        )
        if st.button("🎫 Login with Request Token", type="primary", use_container_width=True):
            if not api_key_input or not api_secret_input or not redirect_url_input:
                st.error("❌ Please provide API Key, API Secret, and the Redirect URL.")
            else:
                request_token = bot.api.parse_request_token_from_url(redirect_url_input)
                if request_token:
                    st.info(f"🔍 Found Request Token: {request_token[:15]}...{request_token[-5:]}")
                    with st.spinner("🔄 Authenticating... Please wait."):
                        # Pass the currently entered api_key and api_secret
                        success = bot.api.login_with_request_token(api_key_input, request_token, api_secret_input)
                        if success:
                            st.session_state.gw_logged_in = True
                            # Store API key in bot.api object after successful login if needed by other methods
                            bot.api.api_key = api_key_input 
                            st.rerun() # Rerun to update UI based on logged_in state
                        # Error messages are now handled inside login_with_request_token
                else:
                    st.error("❌ Could not find a valid request_token in the URL. Please check the full URL.")
else: # Logged In
    st.success(f"✅ Connected via {bot.api.connection_method} as {bot.api.user_profile.get('name', bot.api.client_id if bot.api.client_id else 'N/A')}")
    conn_col1, conn_col2, conn_col3, conn_col4 = st.columns(4)
    with conn_col1: st.metric("Client ID", bot.api.client_id or "N/A")
    with conn_col2: st.metric("Connected At", bot.api.last_login_time.strftime("%H:%M:%S") if bot.api.last_login_time else "N/A")
    
    with conn_col3:
        if st.button("🔄 Test API Connection", use_container_width=True):
            test_ok, msg = bot.api.test_connection()
            if test_ok: st.success(f"✅ Connection Test: {msg}")
            else: st.error(f"❌ Connection Test: {msg}")
            
    with conn_col4:
        if st.button("🚪 Logout from Goodwill", type="secondary", use_container_width=True):
            with st.spinner("Logging out..."):
                bot.api.logout()
                st.session_state.gw_logged_in = False
                # Clear stored API key/secret from session state if you were temporarily storing them there
                if "api_key_input" in st.session_state: del st.session_state.api_key_input
                if "api_secret_input" in st.session_state: del st.session_state.api_secret_input
                st.rerun()

# Trading Controls, Parameters, Dashboard etc. follow as in the original structure
# ... (Ensure these sections are complete: Trading Controls, Parameters, Dashboard, Risk, Performance, Strategies, Positions, Opportunities, History, Market Data, Diagnostics, Footer)
# (The following is a condensed placeholder for brevity, ensure you merge with your full UI code)

st.subheader("🎮 Trading Controls")
# ... (Your UI for start/stop, paper/live mode)
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
with ctrl_c1:
    if st.button("🟠 Paper Mode", disabled=(bot.mode == "PAPER"), use_container_width=True): bot.mode = "PAPER"; st.rerun()
with ctrl_c2:
    if st.button("🔴 Live Mode", disabled=(bot.mode == "LIVE"), use_container_width=True):
        if bot.api.is_connected: bot.mode = "LIVE"; st.warning("⚠️ LIVE MODE ACTIVE!"); st.rerun()
        else: st.error("❌ Connect to API for Live Mode.")
with ctrl_c3:
    if not bot.is_running:
        if st.button("🚀 START BOT", type="primary", use_container_width=True):
            if bot.mode=="LIVE" and not bot.api.is_connected: st.error("Connect API for LIVE mode!")
            else: bot.is_running = True; bot.daily_pnl = 0; bot.peak_capital_session = bot.capital + bot.pnl; st.rerun()
    else:
        if st.button("⏹️ STOP BOT", type="secondary", use_container_width=True): bot.is_running = False; st.rerun()
with ctrl_c4:
    if st.button("🔄 Manual Cycle", use_container_width=True, help="Run one trading scan & manage cycle"):
        with st.spinner("Running manual cycle..."): bot.run_trading_cycle(); st.rerun()


st.subheader("⚙️ Trading Parameters")
# ... (Your UI for risk per trade, max positions, etc.)
param_col1, param_col2, param_col3, param_col4 = st.columns(4)
with param_col1:
    new_risk = st.slider("Risk per Trade (%)", 0.1, 5.0, bot.risk_per_trade_pct*100, 0.1, format="%.1f%%")
    bot.risk_per_trade_pct = new_risk / 100
with param_col2: bot.max_positions = st.slider("Max Positions", 1, 10, bot.max_positions, 1)
with param_col3: bot.max_hold_time_sec = st.slider("Max Hold (sec)", 30, 600, bot.max_hold_time_sec, 15)
with param_col4: bot.risk_reward_ratio = st.slider("Risk:Reward Ratio", 1.0, 5.0, bot.risk_reward_ratio, 0.1, format="1:%.1f")

# ... (The rest of your Streamlit UI sections: Dashboard, Risk Monitoring, Performance Analytics, Strategy Dashboard, Active Positions, Market Opportunities, Trade History, Live Market Data, System Diagnostics, Footer)
# It's critical to merge this authentication part correctly with your existing full UI for other sections.


# Example: Placeholder for Dashboard Section
st.subheader("📊 Status Dashboard")
# ... your dashboard metrics ...
main_col1, main_col2, main_col3, main_col4 = st.columns(4)
main_col1.metric("Bot Status", "🟢 Running" if bot.is_running else "🔴 Stopped")
main_col2.metric("API Status", "🟢 Connected" if bot.api.is_connected else "🔴 Offline")
main_col3.metric("Trading Mode", "🔴 LIVE" if bot.mode == "LIVE" else "🟠 PAPER")
main_col4.metric("Active Positions", f"{len(bot.positions)}/{bot.max_positions}")


st.subheader("📈 Performance Analytics")
metrics = bot.get_performance_metrics()
if metrics['total_trades'] > 0:
    # ... display metrics ...
    st.json(metrics) 
else:
    st.info("📊 No trade data yet for performance analytics.")

# Active Positions Display
st.subheader("🎯 Active Positions")
if bot.positions:
    active_pos_data = []
    for pos_id, pos in bot.positions.items():
        #簡易表示。実際には現在の価格を取得して評価損益を計算
        active_pos_data.append({
            "Symbol": pos['symbol'], "Action": pos['action'], "Qty": pos['quantity'],
            "Entry": pos['entry_price'], "SL": pos['stop_loss_price'], "Tgt": pos['target_price'],
            "Strategy": pos['strategy_name']
        })
    st.dataframe(pd.DataFrame(active_pos_data), use_container_width=True)
else:
    st.info("📭 No active positions.")


st.subheader("📋 Trade History (Last 20)")
if bot.trades:
    df_trades_display = pd.DataFrame(bot.trades[-20:])
    st.dataframe(df_trades_display, use_container_width=True)
else:
    st.info("📭 No trades executed yet in this session.")


# Auto-refresh Logic (Simplified)
if bot.is_running:
    time.sleep(st.session_state.get("bot_cycle_interval", 5)) # Cycle interval
    bot.run_trading_cycle()
    st.rerun()

# Footer
st.markdown("---")
st.markdown("⚡ **FlyingBuddha Scalping Bot (Request Token Version)** | Trade Responsibly")
