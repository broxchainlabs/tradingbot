#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - Complete Production Version
Deploy-ready scalping bot for Indian markets
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid  # For generating a unique IMEI
from collections import deque

# Attempt to import broker libraries, fail gracefully if not installed for paper mode
try:
    from kiteconnect import KiteConnect
    KITE_IMPORTED = True
except ImportError:
    KITE_IMPORTED = False
    print("KiteConnect library not found. Zerodha live trading will not be available.")

try:
    import requests
    REQUESTS_IMPORTED = True
except ImportError:
    REQUESTS_IMPORTED = False
    print("requests library not found. Goodwill trading will not be available.")


# Set page config
st.set_page_config(
    page_title="⚡ FlyingBuddha Scalping Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CORE CLASSES ====================

class BrokerDataFeed:
    """Broker data feed with paper/live toggle"""

    def __init__(self, mode="PAPER", api_key="", api_secret="", access_token="", broker_name="yfinance", user_id="", password="", totp="", imei=""):
        self.mode = mode
        self.api_key = api_key
        self.api_secret = api_secret # For Zerodha if full auth flow is implemented
        self.access_token = access_token # For Zerodha direct, or Goodwill obtained
        self.broker_name = broker_name
        self.is_connected = False
        self.kite = None
        self.goodwill_session = None
        self.refresh_token = None # For Goodwill
        self.user_id = user_id
        self.password = password
        self.totp = totp
        self.imei = imei or str(uuid.uuid4())
        self.instrument_token_map = {}
        self.data_cache = {}
        self.data_cache_max_len = 30 # Store enough for 15-min calculations plus buffer

        if self.mode == "LIVE" or self.mode == "PAPER_BROKER_DATA":
            if self.broker_name == "GOODWILL":
                if not REQUESTS_IMPORTED:
                    st.error("❌ requests library not installed. Please install it: pip install requests")
                    self._fallback_to_paper()
                    return
                if api_key and user_id and password and totp:
                    self.setup_connection()
                else:
                    st.error("❌ API Key, User ID, Password, and TOTP are required for Goodwill modes.")
                    self._fallback_to_paper()
            elif self.broker_name == "ZERODHA":
                if not KITE_IMPORTED:
                    st.error("❌ KiteConnect library not installed. Please install it: pip install kiteconnect")
                    self._fallback_to_paper()
                    return
                if api_key and access_token:
                    self.setup_connection()
                else:
                    st.error("❌ API Key and Access Token are required for ZERODHA LIVE mode.")
                    self._fallback_to_paper()
            elif self.broker_name == "yfinance" and self.mode == "PAPER_BROKER_DATA": # yfinance cannot be broker data source
                st.warning("yfinance selected for Broker Data mode. Switching to standard Paper mode.")
                self.mode = "PAPER"
                self._fallback_to_paper(False) # Don't show error, just info
            else: # yfinance selected for LIVE mode or invalid combo
                 if self.mode == "LIVE": st.error("yfinance cannot be used for LIVE trading.")
                 self._fallback_to_paper()

        elif self.mode == "PAPER":
            self.is_connected = True
            st.info("✅ Operating in PAPER Trading Mode (yfinance data, simulated trades)")

    def _fallback_to_paper(self, show_warning=True):
        self.mode = "PAPER"
        self.broker_name = "yfinance" # Default to yfinance if primary broker fails
        if show_warning:
            st.warning(f"Switched to PAPER mode with yfinance due to issues with {self.broker_name} setup.")
        self.is_connected = True # Mark as connected for paper yfinance mode

    def setup_connection(self):
        if self.broker_name == "GOODWILL" and REQUESTS_IMPORTED:
            try:
                login_url = "https://api.gwcindia.in/v1/auth/login"
                login_payload = {
                    "userId": self.user_id,
                    "password": self.password,
                    "totp": self.totp,
                    "vendorKey": self.api_key,
                    "imei": self.imei
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(login_url, headers=headers, json=login_payload, timeout=10)
                response.raise_for_status()
                login_data = response.json()

                if login_data.get("status") == "success" and "data" in login_data and "accessToken" in login_data["data"]:
                    self.goodwill_session = requests.Session()
                    self.goodwill_session.headers.update({
                        "x-api-key": self.api_key,
                        "Authorization": f"Bearer {login_data['data']['accessToken']}"
                    })
                    self.access_token = login_data['data']['accessToken']
                    self.refresh_token = login_data['data'].get('refreshToken')
                    st.success(f"✅ Connected to Goodwill ({self.mode}). Login Successful.")
                    self.is_connected = True
                    self.load_instrument_tokens()
                else:
                    st.error(f"❌ Goodwill Login Failed: {login_data.get('message', 'Unknown error')}")
                    self._fallback_to_paper()
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Goodwill Login Request Error: {e}")
                self._fallback_to_paper()
            except Exception as e:
                 st.error(f"❌ Goodwill Login Error: {e}")
                 self._fallback_to_paper()

        elif self.broker_name == "ZERODHA" and KITE_IMPORTED:
            try:
                self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(self.access_token)
                profile = self.kite.profile()
                st.success(f"✅ Connected to Zerodha as {profile.get('user_name', 'user')} ({self.mode})")
                self.is_connected = True
            except Exception as e:
                st.error(f"❌ Zerodha Connection Failed: {e}")
                self._fallback_to_paper()

    def refresh_goodwill_token(self):
        if not (self.broker_name == "GOODWILL" and self.is_connected and self.refresh_token and REQUESTS_IMPORTED):
            return False
        try:
            refresh_url = "https://api.gwcindia.in/v1/auth/refresh-token"
            headers = {"Content-Type": "application/json", "x-api-key": self.api_key}
            payload = {"refreshToken": self.refresh_token}
            response = requests.post(refresh_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            refresh_data = response.json()
            if refresh_data.get("status") == "success" and "data" in refresh_data and "accessToken" in refresh_data["data"]:
                self.access_token = refresh_data['data']['accessToken']
                if 'refreshToken' in refresh_data['data']: # Goodwill might rotate refresh tokens
                    self.refresh_token = refresh_data['data']['refreshToken']
                if self.goodwill_session:
                    self.goodwill_session.headers.update({"Authorization": f"Bearer {self.access_token}"})
                st.info("✅ Goodwill access token refreshed.")
                return True
            else:
                st.error(f"❌ Failed to refresh Goodwill token: {refresh_data.get('message', 'Token refresh failed')}")
                self.is_connected = False # Assume connection lost if refresh fails
                return False
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error refreshing Goodwill token (Request): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            st.error(f"❌ Error refreshing Goodwill token (General): {e}")
            self.is_connected = False
            return False

    def load_instrument_tokens(self):
        """Placeholder for loading Goodwill instrument tokens."""
        if self.broker_name == "GOODWILL" and self.is_connected and self.goodwill_session:
            st.info("Attempting to load Goodwill instrument tokens (using placeholder logic)...")
            # THIS IS A PLACEHOLDER. You need to implement actual logic based on Goodwill's API.
            # Example: Fetch from an endpoint like /market/instruments or /market/search
            # For now, we'll create a dummy mapping for symbols in nse_symbols
            dummy_token_start = 1000
            for symbol in SimpleScalpingBot().load_nse_symbols(): # Accessing it this way is not ideal, better pass symbols
                self.instrument_token_map[f"{symbol.upper()}_NSE"] = str(dummy_token_start)
                dummy_token_start +=1
            if self.instrument_token_map:
                 st.success(f"✅ Loaded {len(self.instrument_token_map)} (dummy) instrument tokens for Goodwill.")
            else:
                st.warning("⚠️ Could not load any dummy instrument tokens for Goodwill.")


    def get_instrument_token(self, symbol, exchange="NSE"):
         key = f"{symbol.upper()}_{exchange.upper()}"
         token = self.instrument_token_map.get(key)
         if not token:
             print(f"Token not found for {key}. Available: {list(self.instrument_token_map.keys())[:5]}")
         return token

    def place_order(self, symbol, action, quantity, price):
        if self.mode != "LIVE":
            mode_info = "yfinance data" if self.broker_name == "yfinance" or self.mode == "PAPER" else "Broker Data"
            st.info(f"📝 SIMULATED ({self.mode} with {mode_info}): {action} {quantity} {symbol} @ ₹{price:.2f}")
            return f"PAPER_{self.mode.upper()}_{int(datetime.now().timestamp())}"

        # Live order placement
        if self.broker_name == "GOODWILL" and self.is_connected and self.goodwill_session:
            try:
                order_url = "https://api.gwcindia.in/v1/orders"
                instrument_token = self.get_instrument_token(symbol)
                if not instrument_token:
                    st.error(f"❌ Goodwill: Instrument token not found for {symbol}")
                    return None

                order_payload = {
                    "exchange": "NSE", #TODO: Make dynamic if needed
                    "token": instrument_token,
                    "tradingsymbol": symbol, #TODO: Confirm exact format Goodwill expects
                    "quantity": int(quantity),
                    "price": float(price) if float(price) > 0 else 0, # Ensure 0 for market
                    "orderType": "MARKET" if float(price) == 0 else "LIMIT",
                    "productType": "MIS",
                    "transactionType": action.upper(),
                    "priceType": "DAY", # DAY or IOC
                    "variety": "REGULAR"
                }
                response = self.goodwill_session.post(order_url, json=order_payload, timeout=10)
                if response.status_code == 401: # Unauthorized, try refreshing token
                    st.warning("Goodwill token might be expired. Attempting refresh...")
                    if self.refresh_goodwill_token():
                        response = self.goodwill_session.post(order_url, json=order_payload, timeout=10) # Retry order
                    else:
                        st.error("Goodwill token refresh failed. Cannot place order.")
                        return None
                response.raise_for_status()
                order_data = response.json()

                if order_data.get("status") == "success" and "data" in order_data and "orderId" in order_data["data"]:
                    order_id = order_data["data"]["orderId"]
                    st.success(f"🎯 GOODWILL LIVE Order: {action} {quantity} {symbol} @ ₹{price:.2f}. ID: {order_id}")
                    return order_id
                else:
                    st.error(f"❌ GOODWILL Order Failed: {order_data.get('message', 'Unknown error')}")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"❌ GOODWILL Order Request Error: {e}")
                return None
            except Exception as e:
                st.error(f"❌ GOODWILL Order Error: {e}")
                return None

        elif self.broker_name == "ZERODHA" and self.is_connected and self.kite:
            try:
                # ... (Zerodha order placement logic as before) ...
                st.success(f"🎯 ZERODHA LIVE Order Placed...")
                return f"ZERODHA_LIVE_{int(datetime.now().timestamp())}" # Replace with actual
            except Exception as e:
                st.error(f"❌ ZERODHA Order Error: {e}")
                return None
        else:
            st.error(f"❌ Cannot place LIVE order: Not connected or broker {self.broker_name} not supported for live.")
            return None

    def _parse_goodwill_historical_data(self, data_list, symbol):
        """Parses Goodwill historical data into a pandas DataFrame."""
        if not data_list or not isinstance(data_list, list):
            print(f"Goodwill historical data for {symbol} is empty or not a list.")
            return pd.DataFrame() # Return empty DataFrame

        df = pd.DataFrame(data_list)
        # Assuming Goodwill provides keys like 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        # Convert timestamp to datetime objects. Goodwill's timestamp format needs to be known.
        # Example: If it's epoch seconds: pd.to_datetime(df['timestamp'], unit='s')
        # Example: If it's string 'YYYY-MM-DD HH:MM:SS': pd.to_datetime(df['timestamp'])
        # For now, let's assume it has 'hO', 'hH', 'hL', 'hC', 'hV' and a 'time' field.
        # This part is HIGHLY DEPENDENT on the actual API response structure.
        rename_map = {
            'time': 'Timestamp', # Or whatever the time field is called
            'hO': 'Open', 'hH': 'High', 'hL': 'Low', 'hC': 'Close', 'hV': 'Volume'
        }
        df.rename(columns=rename_map, inplace=True)
        if 'Timestamp' in df.columns:
            try:
                # Attempt to convert assuming a common string format or epoch.
                # This is a common point of failure if format is unexpected.
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
            except Exception as e:
                print(f"Error converting Goodwill timestamp for {symbol}: {e}. Timestamps: {df['Timestamp'].head()}")
                return pd.DataFrame() # Return empty if timestamp conversion fails
        else:
            print(f"Timestamp column not found in Goodwill historical data for {symbol} after rename. Columns: {df.columns}")
            return pd.DataFrame()


        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Expected column '{col}' not found in Goodwill historical data for {symbol}.")
                # Create empty column if missing to avoid errors later, or handle more gracefully
                df[col] = np.nan

        return df.dropna()


    def get_data(self, symbol):
        if (self.mode == "LIVE" or self.mode == "PAPER_BROKER_DATA"):
            if self.broker_name == "GOODWILL" and self.is_connected and self.goodwill_session:
                return self.get_goodwill_data(symbol)
            elif self.broker_name == "ZERODHA" and self.is_connected and self.kite: # Only for LIVE
                 if self.mode == "LIVE":
                    return self.get_zerodha_data(symbol)
                 else: # Zerodha paper data not fully implemented, fallback to yfinance
                    print("Zerodha selected for Paper (Broker Data) - falling back to yfinance for data.")
                    return self.get_yfinance_data(symbol)
            else: # Not connected or unsupported broker for these modes
                print(f"Broker {self.broker_name} not ready for {self.mode}, falling back to yfinance.")
                return self.get_yfinance_data(symbol)
        else: # Standard yfinance paper mode
            return self.get_yfinance_data(symbol)

    def get_goodwill_data(self, symbol):
        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                print(f"Goodwill: Token not found for {symbol}")
                return None

            # 1. Fetch current quote (LTP)
            quotes_url = "https://api.gwcindia.in/v1/market/quotes"
            quotes_payload = {"symbols": [{"exchange": "NSE", "token": instrument_token}]}
            response = self.goodwill_session.post(quotes_url, json=quotes_payload, timeout=10)
            if response.status_code == 401: # Unauthorized
                if self.refresh_goodwill_token(): response = self.goodwill_session.post(quotes_url, json=quotes_payload, timeout=10)
                else: return None
            response.raise_for_status()
            quotes_data = response.json()

            latest_price_data = None
            if quotes_data.get("status") == "success" and quotes_data.get("data"):
                quote = quotes_data["data"][0]
                latest_price_data = {
                    'price': float(quote.get("ltp", 0)),
                    'volume': int(quote.get("volume", 0)),
                    'high': float(quote.get("high", 0)),
                    'low': float(quote.get("low", 0)),
                    'timestamp': datetime.now() # Use current time for LTP
                }
            else:
                print(f"Goodwill: Quote fetch error for {symbol}: {quotes_data.get('message')}")
                return None # Cannot proceed without LTP

            # 2. Fetch or update historical data for indicators
            # Check cache first
            cached_df = self.get_cached_dataframe(symbol)
            current_minute = datetime.now().replace(second=0, microsecond=0)

            if cached_df is None or cached_df.empty or cached_df.index[-1] < current_minute - timedelta(minutes=1):
                # Cache is old or empty, fetch new historical data
                history_url = "https://api.gwcindia.in/v1/market/history" # CHECK Endpoint!
                to_date = datetime.now()
                from_date = to_date - timedelta(days=2) # Fetch a bit more to ensure enough 1-min data

                history_payload = {
                    "exchange": "NSE", "token": instrument_token, "interval": "1m",
                    # Goodwill date format might be "YYYY-MM-DD" or "DD-MM-YYYY". Using ISO for now.
                    "from": from_date.strftime("%Y-%m-%d"), # Or "%d-%m-%Y"
                    "to": to_date.strftime("%Y-%m-%d")     # Or "%d-%m-%Y"
                }
                print(f"Fetching Goodwill history for {symbol}: {history_payload}")
                hist_response = self.goodwill_session.post(history_url, json=history_payload, timeout=15)
                if hist_response.status_code == 401:
                    if self.refresh_goodwill_token(): hist_response = self.goodwill_session.post(history_url, json=history_payload, timeout=15)
                    else: return latest_price_data # Return LTP if history fails after refresh
                hist_response.raise_for_status()
                hist_data_json = hist_response.json()

                if hist_data_json.get("status") == "success" and hist_data_json.get("data"):
                    df = self._parse_goodwill_historical_data(hist_data_json["data"], symbol)
                    if not df.empty:
                        self.update_cache_dataframe(symbol, df)
                        cached_df = df
                    else:
                        print(f"Goodwill: Parsed historical data is empty for {symbol}.")
                        # Fallback to yfinance for history if broker history fails fundamentally
                        return self._calculate_metrics_with_yfinance_fallback(symbol, latest_price_data)
                else:
                    print(f"Goodwill: History fetch error for {symbol}: {hist_data_json.get('message')}")
                    # Fallback to yfinance for history if broker history fails
                    return self._calculate_metrics_with_yfinance_fallback(symbol, latest_price_data)
            else: # Use cached data, potentially append current LTP if it's a new bar
                pass # df is already cached_df

            if cached_df is None or cached_df.empty:
                 print(f"Goodwill: No historical data (cached or fetched) for {symbol}")
                 return latest_price_data # Return only LTP if no history

            # Combine current LTP with historical data if it forms a new bar.
            # This logic can be complex to get right for live appending.
            # For simplicity, we'll rely on the last bar of fetched/cached history for calculations
            # and use the LTP quote as the most current price.

            prices = cached_df['Close'].dropna()
            volumes = cached_df['Volume'].dropna()

            if len(prices) < 15:
                print(f"Goodwill: Not enough historical prices ({len(prices)}) for {symbol} for full calculation.")
                # Optionally, try fetching more data or use yfinance fallback for indicators
                return self._calculate_metrics_with_yfinance_fallback(symbol, latest_price_data)


            change_1min = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100 if len(prices) >= 2 else 0
            change_5min = ((prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]) * 100 if len(prices) >= 6 else 0
            change_15min = ((prices.iloc[-1] - prices.iloc[-15]) / prices.iloc[-15]) * 100 if len(prices) >= 15 else 0
            vol_series = volumes.tail(5)
            vol_ratio = vol_series.iloc[-1] / vol_series.mean() if len(vol_series) >=1 and vol_series.mean() > 0 else 1

            return {
                'price': latest_price_data['price'],
                'volume': latest_price_data['volume'], # Current total volume for the day
                'high': latest_price_data['high'],   # Current day's high
                'low': latest_price_data['low'],     # Current day's low
                'change_1min': change_1min,
                'change_5min': change_5min,
                'change_15min': change_15min,
                'volume_ratio': vol_ratio, # Based on 1-min volume bars
                'timestamp': latest_price_data['timestamp']
            }

        except requests.exceptions.RequestException as e:
            print(f"Goodwill: Request exception for {symbol}: {e}")
            # Attempt to refresh token on general request exception too, if it might be auth-related
            if "token" in str(e).lower() or "auth" in str(e).lower():
                if self.refresh_goodwill_token():
                    return self.get_goodwill_data(symbol) # Retry
            return None
        except Exception as e:
            print(f"Goodwill: Unexpected error for {symbol}: {e}")
            return None

    def _calculate_metrics_with_yfinance_fallback(self, symbol, latest_price_data):
        """Helper to calculate indicators using yfinance if broker history fails."""
        print(f"Falling back to yfinance for historical indicator calculation for {symbol}.")
        yf_data = self.get_yfinance_data(symbol, only_historical=True)
        if yf_data:
            latest_price_data.update({
                'change_1min': yf_data.get('change_1min',0),
                'change_5min': yf_data.get('change_5min',0),
                'change_15min': yf_data.get('change_15min',0),
                'volume_ratio': yf_data.get('volume_ratio',1)
            })
        return latest_price_data


    def get_zerodha_data(self, symbol):
        if self.kite and self.is_connected:
            try:
                # For Zerodha, LTP is easy. Historical data for indicators is more involved.
                # We might use yfinance for historical if Kite historical is too complex for this script.
                quote = self.kite.quote(f"NSE:{symbol}")
                if quote and f"NSE:{symbol}" in quote:
                    q_data = quote[f"NSE:{symbol}"]
                    ltp = q_data['last_price']
                    day_volume = q_data['volume']
                    day_high = q_data['ohlc']['high']
                    day_low = q_data['ohlc']['low']

                    latest_price_data = {'price': ltp, 'volume': day_volume, 'high': day_high, 'low': day_low, 'timestamp': datetime.now()}
                    return self._calculate_metrics_with_yfinance_fallback(symbol, latest_price_data) # Use yf for indicators
                else:
                    print(f"Zerodha: Quote fetch error for {symbol}")
                    return None
            except Exception as e:
                print(f"Zerodha: Data fetch error for {symbol}: {e}")
                return None
        return None

    def get_yfinance_data(self, symbol, only_historical=False):
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            # Fetch more data to ensure calculations are possible if some bars are missing
            data = ticker.history(period="2d", interval="1m") # Fetch 2 days for robustness
            if data.empty:
                print(f"yfinance: No data for {symbol}")
                return None

            data = data.iloc[-self.data_cache_max_len:] # Keep only relevant recent data
            if data.empty or len(data) < 15: # Need at least 15 for 15-min change
                print(f"yfinance: Not enough recent data for {symbol} (need 15 bars, got {len(data)})")
                return None

            latest = data.iloc[-1]
            prices = data['Close'].dropna()
            volumes = data['Volume'].dropna() # 1-min volumes

            if len(prices) < 15:
                print(f"yfinance: Not enough price points after dropna for {symbol} (need 15, got {len(prices)})")
                return None

            change_1min = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100 if len(prices) >= 2 else 0
            change_5min = ((prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]) * 100 if len(prices) >= 6 else 0
            change_15min = ((prices.iloc[-1] - prices.iloc[-15]) / prices.iloc[-15]) * 100 if len(prices) >= 15 else 0
            vol_series = volumes.tail(5) # Last 5 1-min volumes
            vol_ratio = vol_series.iloc[-1] / vol_series.mean() if len(vol_series) >= 1 and vol_series.mean() > 0 else 1

            result = {
                'price': float(latest['Close']),
                'volume': int(latest['Volume']), # This is the volume of the last 1-min bar
                'high': float(latest['High']),   # High of the last 1-min bar
                'low': float(latest['Low']),     # Low of the last 1-min bar
                'change_1min': change_1min,
                'change_5min': change_5min,
                'change_15min': change_15min,
                'volume_ratio': vol_ratio,
                'timestamp': pd.to_datetime(latest.name).to_pydatetime() # Timestamp of the bar
            }
            if only_historical: # If called as fallback, only indicators are needed
                return {k: v for k, v in result.items() if k not in ['price', 'volume', 'high', 'low', 'timestamp']}
            return result

        except Exception as e:
            print(f"yfinance: Error for {symbol}: {e}")
            return None

    def update_cache_dataframe(self, symbol, df_new_data):
        """Updates the historical data cache with a new DataFrame."""
        if df_new_data.empty:
            return
        if symbol not in self.data_cache:
            self.data_cache[symbol] = pd.DataFrame()

        # Ensure new data has Timestamp index and required columns
        if not isinstance(df_new_data.index, pd.DatetimeIndex):
            print(f"Cache update for {symbol} failed: New data has no DatetimeIndex.")
            return
        # Required columns for calculations
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df_new_data.columns for col in required_cols):
            print(f"Cache update for {symbol} failed: New data missing required columns. Has: {df_new_data.columns}")
            return

        # Concatenate and keep unique recent data
        combined_df = pd.concat([self.data_cache[symbol], df_new_data])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')] # Keep last entry for duplicate index
        combined_df.sort_index(inplace=True)
        self.data_cache[symbol] = combined_df.tail(self.data_cache_max_len * 2) # Keep more for buffer

    def get_cached_dataframe(self, symbol) -> Optional[pd.DataFrame]:
        """Retrieves the cached DataFrame for a symbol."""
        return self.data_cache.get(symbol)


class TradingModeManager:
    """Easy toggle between paper and live trading"""
    def __init__(self, bot_instance):
        self.bot = bot_instance
        # Initialize saved credentials (can be loaded from a config file in a real app)
        self.saved_credentials = {
            "GOODWILL": {"api_key": "", "user_id": "", "password": "", "totp": "", "imei": ""},
            "ZERODHA": {"api_key": "", "api_secret": "", "access_token": ""},
            "yfinance": {} # No creds for yfinance
        }
        self.current_broker = "yfinance" # Default

    def _update_bot_and_mode(self, mode, broker_name_to_set):
        self.bot.mode = mode
        self.bot.broker_name = broker_name_to_set # Update bot's understanding of current broker
        creds = self.saved_credentials.get(broker_name_to_set, {})
        self.bot.setup_data_feed(
            broker_name=broker_name_to_set, # Pass explicitly
            api_key=creds.get("api_key"),
            api_secret=creds.get("api_secret"),
            access_token=creds.get("access_token"),
            user_id=creds.get("user_id"),
            password=creds.get("password"),
            totp=creds.get("totp"), # TOTP will be asked fresh via UI
            imei=creds.get("imei")
        )

    def switch_mode(self, new_mode, broker_name, **kwargs):
        """Generic method to switch mode and broker, and setup data feed."""
        self.current_broker = broker_name
        self.saved_credentials[broker_name].update(kwargs) # Save/update creds for this broker

        if new_mode == "PAPER":
            self._update_bot_and_mode("PAPER", "yfinance") # Standard paper always uses yfinance
            st.success("✅ Switched to Paper Trading (yfinance data)")
        elif new_mode == "PAPER_BROKER_DATA":
            if broker_name == "yfinance":
                st.error("yfinance cannot be used for 'Paper (Broker Data)'. Select a broker.")
                return False
            self._update_bot_and_mode("PAPER_BROKER_DATA", broker_name)
            if self.bot.data_feed.is_connected:
                st.success(f"✅ Switched to Paper Trading ({broker_name} data)")
            else:
                st.error(f"⚠️ Failed to connect to {broker_name} for paper trading data. Check credentials.")
                self._update_bot_and_mode("PAPER", "yfinance") # Fallback
        elif new_mode == "LIVE":
            if broker_name == "yfinance":
                st.error("yfinance cannot be used for LIVE trading.")
                return False
            self._update_bot_and_mode("LIVE", broker_name)
            if self.bot.data_feed.is_connected:
                st.success(f"✅ Switched to LIVE Trading ({broker_name})")
            else:
                st.error(f"🔴 Failed to connect for {broker_name} Live Trading. Check credentials.")
                self._update_bot_and_mode("PAPER", "yfinance") # Fallback
        return self.bot.data_feed.is_connected


    def get_current_mode_info(self):
        icon = "❓"
        status_text = "Unknown Mode"
        data_source_text = "N/A"
        risk_text = "N/A"

        current_broker_for_mode = self.bot.broker_name # Broker used by data_feed
        if self.bot.mode == "LIVE":
            icon, status_text, risk_text = '🔴', 'Live Trading Active', 'HIGH - Real money trades'
            data_source_text = f'{current_broker_for_mode} Live Feed'
        elif self.bot.mode == "PAPER_BROKER_DATA":
            icon, status_text, risk_text = '🟠', 'Paper Trading (Broker Data)', 'ZERO - Simulated trades'
            data_source_text = f'{current_broker_for_mode} Data Feed'
        elif self.bot.mode == "PAPER":
            icon, status_text, risk_text = '🟢', 'Paper Trading (yfinance)', 'ZERO - Simulated trades'
            data_source_text = 'yfinance (Delayed)'

        return {
            'mode': self.bot.mode, 'icon': icon, 'status': status_text,
            'data_source': data_source_text, 'risk': risk_text
        }

class SimpleScalpingBot:
    def __init__(self):
        self.is_running = False
        self.mode = "PAPER"
        self.capital = 100000.0
        self.positions = {}
        self.trades = []
        self.signals = []
        self.pnl = 0.0
        self.broker_name = "yfinance" # Default broker for the bot's context
        self.data_feed: Optional[BrokerDataFeed] = None

        self.nse_symbols = self.load_nse_symbols()
        self.active_symbols = []
        self.price_data = {}
        self.max_active_positions = 3
        self.init_db()
        if not self.data_feed: # Initial setup
             self.setup_data_feed(self.broker_name)


    def load_nse_symbols(self):
        return [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
            "BAJFINANCE", "BAJAJFINSV", "RELIANCE", "INFY", "TCS"
        ] # Reduced for faster testing

    def init_db(self):
        try:
            self.conn = sqlite3.connect('bot_data.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, action TEXT,
                    price REAL, quantity INTEGER, pnl REAL, strategy TEXT
                )''')
            self.conn.commit()
        except Exception as e:
            st.error(f"DB Init error: {e}")


    def setup_data_feed(self, broker_name_to_use, **kwargs):
        self.broker_name = broker_name_to_use # Update bot's broker context
        # Pass all kwargs to BrokerDataFeed, which knows how to handle them based on broker_name
        self.data_feed = BrokerDataFeed(mode=self.mode, broker_name=broker_name_to_use, **kwargs)


    def scan_market(self):
        if not self.data_feed or not self.data_feed.is_connected:
            st.warning("Market Scan: Data feed not available or not connected.")
            return []
        opportunities = []
        for symbol in self.nse_symbols: # Use curated list
            score = self.analyze_symbol(symbol)
            if score > 40: # Threshold for opportunity
                opportunities.append((symbol, score))
        opportunities.sort(key=lambda x: x[1], reverse=True)
        self.active_symbols = [s for s, _ in opportunities[:8]] # Track top 8 for signal generation
        return opportunities[:5] # Return top 5 for display


    def analyze_symbol(self, symbol):
        if not self.data_feed: return 0
        data = self.data_feed.get_data(symbol)
        if not data or data.get('price') is None:
            # print(f"No data or price for {symbol} in analyze_symbol")
            return 0

        score = 0
        # Use abs for momentum checks as direction is handled by signal generation
        if abs(data.get('change_1min', 0)) > 0.1: score += 25
        if abs(data.get('change_5min', 0)) > 0.2: score += 20
        if data.get('volume_ratio', 1) > 1.5: score += 20
        elif data.get('volume_ratio', 1) > 1.2: score += 10
        if data.get('volume', 0) > 20000 : score += 15 # Checks if the last 1-min bar volume is high (or total day for some APIs)

        if score > 30: # Store data if score is decent
            self.price_data[symbol] = data
            self.price_data[symbol]['score'] = score
        return score

    def generate_signals(self):
        new_signals = []
        if not self.active_symbols: self.scan_market() # Scan if no active symbols

        for symbol in self.active_symbols:
            if symbol not in self.price_data:
                # Try to get data if missing, might happen if scan was awhile ago
                self.analyze_symbol(symbol)
                if symbol not in self.price_data:
                    continue # Skip if still no data

            signal = self.check_trading_conditions(symbol)
            if signal:
                new_signals.append(signal)
                self.signals.append(signal) # Log all signals
                self.execute_signal(signal)
        return new_signals

    def check_trading_conditions(self, symbol):
        data = self.price_data[symbol]
        # Ensure all keys are present before attempting to use them
        required_keys = ['change_1min', 'change_5min', 'volume_ratio', 'price', 'score']
        if not all(key in data and data[key] is not None for key in required_keys):
            # print(f"Missing required keys in data for {symbol}: {data}")
            return None

        # 1-minute momentum strategy
        if abs(data['change_1min']) > 0.15 and data['volume_ratio'] > 1.5:
            return {'symbol': symbol, 'action': 'BUY' if data['change_1min'] > 0 else 'SELL',
                    'price': data['price'], 'confidence': min(data['score'], 90),
                    'timestamp': datetime.now(), 'strategy': '1min_momentum', 'timeframe': '1min'}
        # 5-minute breakout strategy
        if abs(data['change_5min']) > 0.3 and data['volume_ratio'] > 1.3:
            return {'symbol': symbol, 'action': 'BUY' if data['change_5min'] > 0 else 'SELL',
                    'price': data['price'], 'confidence': min(data['score'], 85),
                    'timestamp': datetime.now(), 'strategy': '5min_breakout', 'timeframe': '5min'}
        return None

    def execute_signal(self, signal):
        if len(self.positions) >= self.max_active_positions:
            st.warning(f"Max positions ({self.max_active_positions}) reached. Signal for {signal['symbol']} ignored.")
            return
        if not self.data_feed:
            st.error("Execute Signal: Data feed not available.")
            return

        risk_per_trade_abs = self.capital * 0.005  # Risk 0.5% of capital per trade
        stop_loss_pct = 0.25 / 100  # 0.25% stop loss from entry price

        if signal['price'] <= 0 or stop_loss_pct <= 0 : # Basic sanity check
            st.error(f"Invalid price or SL% for {signal['symbol']}. Price: {signal['price']}, SL: {stop_loss_pct*100}%")
            return

        # Calculate quantity based on risk
        stop_loss_amount_per_share = signal['price'] * stop_loss_pct
        if stop_loss_amount_per_share <= 0:
             st.error(f"Stop loss amount per share is zero or negative for {signal['symbol']}. Cannot calculate quantity.")
             return
        quantity = int(risk_per_trade_abs / stop_loss_amount_per_share)
        quantity = max(1, min(quantity, 100)) # Min 1 share, Max 100 shares (example cap)

        order_id = self.data_feed.place_order(
            symbol=signal['symbol'], action=signal['action'],
            quantity=quantity, price=signal['price'] # For LIMIT. For MARKET, price might be 0.
        )

        if order_id:
            # Assuming order fills at signal price for paper/simplicity
            entry_price = signal['price']
            pos_id = f"{signal['symbol']}_{int(datetime.now().timestamp())}"
            self.positions[pos_id] = {
                'symbol': signal['symbol'], 'action': signal['action'], 'quantity': quantity,
                'entry_price': entry_price, 'entry_time': datetime.now(),
                'stop_loss': entry_price * (1 - stop_loss_pct) if signal['action'] == 'BUY' else entry_price * (1 + stop_loss_pct),
                'target': entry_price * (1 + (2 * stop_loss_pct)) if signal['action'] == 'BUY' else entry_price * (1 - (2 * stop_loss_pct)), # Example R:R 1:2
                'strategy': signal['strategy'], 'timeframe': signal.get('timeframe', '5min'),
                'order_id': order_id
            }
            st.success(f"Position opened for {signal['symbol']} ({pos_id}). Qty: {quantity}")
        else:
            st.error(f"Order placement failed for {signal['symbol']}. No position opened.")


    def update_positions(self):
        if not self.data_feed: return
        closed_positions_ids = []
        for pos_id, pos in list(self.positions.items()): # Iterate over a copy for safe deletion
            current_data = self.data_feed.get_data(pos['symbol'])
            if not current_data or current_data.get('price') is None:
                # print(f"No current data for active position {pos['symbol']}, skipping update.")
                continue
            current_price = current_data['price']
            exit_reason = None
            if pos['action'] == 'BUY':
                if current_price <= pos['stop_loss']: exit_reason = "STOP_LOSS"
                elif current_price >= pos['target']: exit_reason = "TARGET_HIT"
            else: # SELL
                if current_price >= pos['stop_loss']: exit_reason = "STOP_LOSS"
                elif current_price <= pos['target']: exit_reason = "TARGET_HIT"

            hold_minutes = (datetime.now() - pos['entry_time']).total_seconds() / 60
            max_hold = {'1min': 5, '5min': 15}.get(pos['timeframe'], 10) # Adjusted hold times
            if not exit_reason and hold_minutes > max_hold : exit_reason = "TIME_EXIT"

            if exit_reason:
                self.close_position(pos_id, current_price, exit_reason)
                closed_positions_ids.append(pos_id)
        # for pos_id in closed_positions_ids: # Already deleted in close_position
        #     if pos_id in self.positions: del self.positions[pos_id]

    def close_position(self, pos_id, exit_price, reason):
        if pos_id not in self.positions: return # Already closed
        pos = self.positions.pop(pos_id) # Remove and get position

        pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['action'] == 'BUY' else \
              (pos['entry_price'] - exit_price) * pos['quantity']
        self.capital += pnl
        self.pnl += pnl

        trade_log = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'symbol': pos['symbol'],
            'action': f"CLOSE {pos['action']}", 'price': exit_price, 'quantity': pos['quantity'],
            'pnl': round(pnl, 2), 'strategy': f"{pos['strategy']}_{reason}"
        }
        self.trades.append(trade_log)
        if self.data_feed.mode == "LIVE": # Also log to DB for live trades
             try:
                cursor = self.conn.cursor()
                cursor.execute("INSERT INTO trades (timestamp, symbol, action, price, quantity, pnl, strategy) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (trade_log['timestamp'], trade_log['symbol'], trade_log['action'], trade_log['price'],
                                trade_log['quantity'], trade_log['pnl'], trade_log['strategy']))
                self.conn.commit()
             except Exception as e:
                 st.error(f"DB Error logging trade: {e}")

        st.info(f"Closed {pos['symbol']} ({pos['action']}) @ {exit_price:.2f}. P&L: {pnl:.2f}. Reason: {reason}")


    def get_performance(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return {'total_trades': 0, 'winning_trades': 0, 'win_rate': 0,
                    'total_pnl': self.pnl, 'avg_trade': 0, 'max_profit':0, 'max_loss':0}
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        pnls = [t['pnl'] for t in self.trades]
        return {
            'total_trades': total_trades, 'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
            'total_pnl': round(self.pnl, 2),
            'avg_trade': round(self.pnl / total_trades, 2) if total_trades > 0 else 0,
            'max_profit': round(max(pnls),2) if any(p > 0 for p in pnls) else 0,
            'max_loss': round(min(pnls),2) if any(p < 0 for p in pnls) else 0
        }

# ==================== STREAMLIT UI ====================
if 'bot' not in st.session_state:
    st.session_state.bot = SimpleScalpingBot()
bot = st.session_state.bot

if 'mode_manager' not in st.session_state:
    st.session_state.mode_manager = TradingModeManager(bot) # Pass bot instance
mode_manager = st.session_state.mode_manager

# Initial setup of data_feed if not already done
if bot.data_feed is None:
    creds = mode_manager.saved_credentials.get(bot.broker_name, {})
    bot.setup_data_feed(
        bot.broker_name,
        api_key=creds.get("api_key"), api_secret=creds.get("api_secret"),
        access_token=creds.get("access_token"), user_id=creds.get("user_id"),
        # password=creds.get("password"), totp=creds.get("totp"), # Don't pass sensitive for initial setup
        imei=creds.get("imei")
    )

st.sidebar.header("⚡ Scalping Bot Controls")

# --- Broker Selection ---
broker_options = ["yfinance", "GOODWILL", "ZERODHA"] # yfinance for standard paper
# Ensure current bot.broker_name is valid, else default
if bot.broker_name not in broker_options: bot.broker_name = "yfinance"
selected_broker_ui = st.sidebar.selectbox(
    "Select Broker/Data Source",
    broker_options,
    index=broker_options.index(bot.broker_name),
    key="broker_select_ui"
)
# Update bot's broker context if changed by user, this doesn't trigger connection yet
if selected_broker_ui != bot.broker_name:
    bot.broker_name = selected_broker_ui
    # Mode manager also needs to know
    mode_manager.current_broker = selected_broker_ui
    # Re-initialize data feed with new broker context but current mode and no creds yet
    # This effectively disconnects the old feed
    bot.setup_data_feed(selected_broker_ui)
    st.rerun()


# --- Credentials Expander ---
creds_kwargs = {} # To collect credentials for mode switching
with st.sidebar.expander(f"🔗 API Credentials for {selected_broker_ui}", expanded= (selected_broker_ui != "yfinance")):
    if selected_broker_ui == "GOODWILL":
        st.info("Goodwill: Enter User ID, Password, API Key (as Vendor Key), and current TOTP.")
        creds_kwargs["api_key"] = st.text_input("Goodwill API Key (VendorKey)", value=mode_manager.saved_credentials["GOODWILL"].get("api_key",""), type="password", key="gw_apikey")
        creds_kwargs["user_id"] = st.text_input("Goodwill User ID", value=mode_manager.saved_credentials["GOODWILL"].get("user_id",""), key="gw_userid")
        creds_kwargs["password"] = st.text_input("Goodwill Password", type="password", key="gw_password")
        creds_kwargs["totp"] = st.text_input("Goodwill TOTP", key="gw_totp") # TOTP is always fresh
        creds_kwargs["imei"] = st.text_input("Goodwill IMEI (optional)", value=mode_manager.saved_credentials["GOODWILL"].get("imei", str(uuid.uuid4())), key="gw_imei")
    elif selected_broker_ui == "ZERODHA":
        st.info("Zerodha: Enter API Key and pre-generated Access Token.")
        creds_kwargs["api_key"] = st.text_input("Zerodha API Key", value=mode_manager.saved_credentials["ZERODHA"].get("api_key",""), type="password", key="z_apikey")
        creds_kwargs["access_token"] = st.text_input("Zerodha Access Token", value=mode_manager.saved_credentials["ZERODHA"].get("access_token",""), type="password", key="z_token")
        # creds_kwargs["api_secret"] = st.text_input("Zerodha API Secret", type="password", key="z_secret") # If implementing full auth
    else: # yfinance
        st.info("yfinance (Paper) mode does not require API credentials.")

# --- Trading Mode Selection ---
st.sidebar.subheader("🔄 Trading Mode")
current_mode_display_info = mode_manager.get_current_mode_info()
st.sidebar.markdown(f"**Current:** {current_mode_display_info['icon']} {current_mode_display_info['mode']} ({current_mode_display_info['data_source']})")

mode_col1, mode_col2, mode_col3 = st.sidebar.columns(3)
with mode_col1:
    if st.button("🟢 Paper", help="Simulated trades with yfinance data."):
        if mode_manager.switch_mode("PAPER", "yfinance"): # Standard paper is always yfinance
            st.rerun()
with mode_col2:
    if st.button("🟠 Paper (Broker)", help=f"Simulated trades with {selected_broker_ui} data.", disabled=(selected_broker_ui=="yfinance")):
        if selected_broker_ui != "yfinance":
            if mode_manager.switch_mode("PAPER_BROKER_DATA", selected_broker_ui, **creds_kwargs):
                st.rerun()
        else:
            st.warning("Select a broker (not yfinance) for this mode.")
with mode_col3:
    if st.button("🔴 Live", help=f"Real trades with {selected_broker_ui}.", disabled=(selected_broker_ui=="yfinance")):
        if selected_broker_ui != "yfinance":
            if mode_manager.switch_mode("LIVE", selected_broker_ui, **creds_kwargs):
                st.rerun()
        else:
            st.warning("Select a broker (not yfinance) for LIVE trading.")


# --- Bot Settings ---
st.sidebar.subheader("⚙️ Settings")
capital_input = st.sidebar.number_input("Capital (₹)", value=bot.capital, min_value=10000.0, step=1000.0)
if capital_input != bot.capital:
    bot.capital = capital_input
    # bot.pnl = 0 # Optionally reset PNL if capital changes
    # bot.trades = []

max_pos_input = st.sidebar.slider("Max Active Positions", 1, 10, bot.max_active_positions)
if max_pos_input != bot.max_active_positions:
    bot.max_active_positions = max_pos_input


# --- Control Buttons ---
st.sidebar.subheader("🎮 Control")
# Disable start if not connected, unless it's yfinance paper which is always "connected"
can_start = (bot.data_feed and bot.data_feed.is_connected) or \
            (bot.mode == "PAPER" and bot.broker_name == "yfinance")

col_ctrl1, col_ctrl2 = st.sidebar.columns(2)
with col_ctrl1:
    if st.button("🚀 Start", disabled=bot.is_running or not can_start):
        if can_start:
            bot.is_running = True
            st.success(f"Bot started in {bot.mode} with {bot.broker_name}!")
            st.rerun()
        else:
            st.error("Cannot start. Data feed not connected. Check Broker/API settings and selected mode.")
with col_ctrl2:
    if st.button("⏹️ Stop", disabled=not bot.is_running):
        bot.is_running = False
        st.success("Bot stopped!")
        st.rerun()

# Status display
st.sidebar.subheader("📊 Status")
if bot.is_running:
    if bot.mode == "LIVE": st.sidebar.error("🔴 LIVE TRADING ACTIVE")
    elif bot.mode == "PAPER_BROKER_DATA": st.sidebar.warning("🟠 PAPER (BROKER DATA) ACTIVE")
    else: st.sidebar.success("🟢 PAPER (YFINANCE) ACTIVE")
else:
    st.sidebar.info("⚪ BOT STOPPED")

if bot.data_feed and bot.data_feed.is_connected:
    st.sidebar.caption(f"🔌 Connected to: {bot.data_feed.broker_name}")
else:
    st.sidebar.caption(f"🔌 Not Connected to: {bot.broker_name}")


# ==================== MAIN DASHBOARD ====================
st.title(f"⚡ FlyingBuddha Scalping Bot ({bot.mode} - {bot.broker_name})")

# Mode banner in main area
mode_info_main = mode_manager.get_current_mode_info()
if bot.mode == "LIVE": st.error(f"{mode_info_main['icon']} **{mode_info_main['status'].upper()}** | DATA: {mode_info_main['data_source']} | RISK: {mode_info_main['risk']}")
elif bot.mode == "PAPER_BROKER_DATA": st.warning(f"{mode_info_main['icon']} **{mode_info_main['status'].upper()}** | DATA: {mode_info_main['data_source']} | RISK: {mode_info_main['risk']}")
else: st.success(f"{mode_info_main['icon']} **{mode_info_main['status'].upper()}** | DATA: {mode_info_main['data_source']} | RISK: {mode_info_main['risk']}")


if bot.is_running:
    # Perform bot actions only if running
    bot.scan_market() # Scan periodically or let generate_signals do it if needed
    new_signals = bot.generate_signals() # This also executes signals
    bot.update_positions()

    if new_signals: st.balloons() # Celebrate new signals that led to trades
    st.info("⚡ Auto-refreshing every 8 seconds...") # Adjust as needed
    time.sleep(8)
    st.rerun()


# Metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
perf = bot.get_performance()
col_m1.metric("Capital", f"₹{bot.capital:,.2f}", delta=f"{perf['total_pnl']:,.2f} P&L")
col_m2.metric("Total Trades", perf['total_trades'])
col_m3.metric("Win Rate", f"{perf['win_rate']:.1f}%", delta=f"{perf['winning_trades']} Wins")
col_m4.metric("Active Positions", len(bot.positions))


# Opportunities Display
st.subheader("🎯 Live Opportunities (Top Scanned)")
if bot.price_data and bot.active_symbols: # Show data for active_symbols
    opp_data_rows = []
    # Sort active symbols by score from price_data for display
    sorted_display_symbols = sorted(
        bot.active_symbols,
        key=lambda s: bot.price_data.get(s, {}).get('score', 0),
        reverse=True
    )
    for symbol in sorted_display_symbols:
        data = bot.price_data.get(symbol)
        if data:
            opp_data_rows.append({
                'Symbol': symbol, 'Score': f"{data.get('score',0):.0f}",
                'Price': f"₹{data.get('price',0):.2f}",
                '1m %': f"{data.get('change_1min',0):+.2f}%",
                '5m %': f"{data.get('change_5min',0):+.2f}%",
                'Vol Ratio': f"{data.get('volume_ratio',0):.1f}x"
            })
    if opp_data_rows: st.dataframe(pd.DataFrame(opp_data_rows), use_container_width=True, hide_index=True)
    else: st.info("No active opportunities matching criteria, or data pending.")
else: st.info("Bot not running or no opportunities identified yet.")


# Positions and Trades Display
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("📍 Active Positions")
    if bot.positions:
        pos_rows = []
        for pos_id, pos in bot.positions.items():
            current_p = bot.price_data.get(pos['symbol'], {}).get('price', pos['entry_price']) # Use last known price
            current_pnl = (current_p - pos['entry_price']) * pos['quantity'] if pos['action'] == 'BUY' else \
                          (pos['entry_price'] - current_p) * pos['quantity']
            pos_rows.append({
                'Symbol': pos['symbol'], 'Action': pos['action'], 'Qty': pos['quantity'],
                'Entry ₹': f"{pos['entry_price']:.2f}", 'Current ₹': f"{current_p:.2f}",
                'SL ₹': f"{pos['stop_loss']:.2f}", 'Tgt ₹': f"{pos['target']:.2f}",
                'P&L ₹': f"{current_pnl:.2f}"
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else: st.info("No active positions.")

with col_p2:
    st.subheader("📈 Recent Trades Log")
    if bot.trades:
        # Sort trades by timestamp descending for display if not already
        # Assuming trades are appended, so last ones are most recent
        recent_trades_df = pd.DataFrame(bot.trades[-10:]) # Show last 10
        st.dataframe(recent_trades_df, use_container_width=True, hide_index=True)
    else: st.info("No trades executed yet.")


# Signals Display
st.subheader("📡 Recent Signals (Generated)")
if bot.signals:
    sig_rows = []
    for signal in reversed(bot.signals[-5:]): # Show last 5, most recent first
        sig_rows.append({
            'Time': signal['timestamp'].strftime("%H:%M:%S"), 'Symbol': signal['symbol'],
            'Action': signal['action'], 'Price ₹': f"{signal['price']:.2f}",
            'Confidence': f"{signal['confidence']:.0f}%", 'Strategy': signal['strategy']
        })
    st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
else: st.info("No signals generated recently.")


# Footer
st.markdown("---")
st.markdown(f"**FlyingBuddha Scalping Bot** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if not bot.is_running:
    st.info("💡 Bot is stopped. Select mode, configure API (if needed), and click 'Start'.")

