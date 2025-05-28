#!/usr/bin/env python3
"""
FlyingBuddha Scalping Bot - COMPLETE FIXED PRODUCTION VERSION
Real-time scalping with 8 advanced strategies and FIXED Goodwill API integration
- FIXED gwcmodel detection with enhanced class scanning
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

# ==================== ENHANCED GWCMODEL DETECTION ====================

# Enhanced gwcmodel detection with comprehensive class scanning
try:
    import gwcmodel
    GWCMODEL_AVAILABLE = True
    
    # Get all available attributes in gwcmodel
    gwcmodel_attributes = [attr for attr in dir(gwcmodel) if not attr.startswith('_')]
    
    # Priority order for class detection - comprehensive list
    class_candidates = [
        'GWCModel', 'GWCApi', 'GWC', 'Api', 'Client', 'GoodwillApi', 
        'GWCClient', 'Trading', 'Session', 'Connection', 'Broker',
        'GWCConnect', 'GWCSession', 'GWCTrade', 'NorenApi'
    ]
    
    GWC_CLASS = None
    GWC_CLASS_NAME = None
    
    # Method 1: Try priority candidates
    for candidate in class_candidates:
        if hasattr(gwcmodel, candidate):
            try:
                cls = getattr(gwcmodel, candidate)
                # Check if it's a class with __init__ method
                if hasattr(cls, '__init__') and hasattr(cls, '__call__'):
                    GWC_CLASS = cls
                    GWC_CLASS_NAME = candidate
                    break
            except Exception:
                continue
    
    # Method 2: Scan all attributes for class-like objects
    if not GWC_CLASS:
        for attr_name in gwcmodel_attributes:
            try:
                attr = getattr(gwcmodel, attr_name)
                # Check if it looks like a class
                if (hasattr(attr, '__init__') and 
                    hasattr(attr, '__class__') and 
                    callable(attr) and
                    attr_name[0].isupper()):  # Class names usually start with capital
                    GWC_CLASS = attr
                    GWC_CLASS_NAME = attr_name
                    break
            except Exception:
                continue
    
    # Method 3: Try to find any callable with 'login' or 'connect' methods
    if not GWC_CLASS:
        for attr_name in gwcmodel_attributes:
            try:
                attr = getattr(gwcmodel, attr_name)
                if (callable(attr) and 
                    (hasattr(attr, 'login') or hasattr(attr, 'connect') or hasattr(attr, 'authenticate'))):
                    GWC_CLASS = attr
                    GWC_CLASS_NAME = attr_name
                    break
            except Exception:
                continue
                
except ImportError:
    GWCMODEL_AVAILABLE = False
    GWC_CLASS = None
    GWC_CLASS_NAME = None
    gwcmodel_attributes = []

# Set page config
st.set_page_config(
    page_title="⚡ FlyingBuddha Scalping Bot - FIXED",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== COMPLETE FIXED GOODWILL INTEGRATION ====================

class CompleteGoodwillIntegration:
    """
    COMPLETE FIXED Goodwill Integration with robust authentication
    - Enhanced gwcmodel detection and initialization
    - Proper request token flow per official documentation
    - Multiple fallback authentication methods
    - Production-ready error handling
    """
    
    def __init__(self):
        self.gwc_client = None
        self.access_token = None
        self.api_key = None
        self.api_secret = None
        self.client_id = None
        self.user_session_id = None
        self.is_connected = False
        self.connection_method = None
        self.base_url = "https://api.gwcindia.in/v1"
        self.last_login_time = None
        self.user_profile = None
        
        # Store credentials for session management
        self.stored_credentials = {}
    
    def get_gwcmodel_status(self) -> Dict:
        """Get detailed gwcmodel status"""
        return {
            'available': GWCMODEL_AVAILABLE,
            'class_found': GWC_CLASS is not None,
            'class_name': GWC_CLASS_NAME,
            'attributes': gwcmodel_attributes[:10] if gwcmodel_attributes else [],
            'total_attributes': len(gwcmodel_attributes)
        }
    
    def initialize_gwcmodel(self) -> Tuple[bool, str]:
        """Enhanced gwcmodel initialization with detailed diagnostics"""
        if not GWCMODEL_AVAILABLE:
            return False, "❌ gwcmodel library not installed. Install with: pip install gwcmodel"
        
        if not GWC_CLASS:
            available_attrs = ', '.join(gwcmodel_attributes[:8])
            return False, f"❌ No valid gwcmodel class found. Available: {available_attrs}..."
        
        try:
            # Try different initialization methods
            init_methods = [
                lambda: GWC_CLASS(),
                lambda: GWC_CLASS(debug=False),
                lambda: GWC_CLASS(host="api.gwcindia.in"),
                lambda: GWC_CLASS(userid="", password="", twoFA="", vendor_code="", api_secret="", imei="abc1234")
            ]
            
            for i, init_method in enumerate(init_methods):
                try:
                    self.gwc_client = init_method()
                    return True, f"✅ Successfully initialized gwcmodel.{GWC_CLASS_NAME} (method {i+1})"
                except Exception as e:
                    if i == len(init_methods) - 1:  # Last attempt
                        return False, f"❌ All initialization methods failed. Last error: {str(e)}"
                    continue
            
        except Exception as e:
            return False, f"❌ Initialization error: {str(e)}"
    
    def generate_login_url(self, api_key: str) -> str:
        """Generate login URL for request token flow"""
        return f"https://api.gwcindia.in/v1/login?api_key={api_key}"
    
    def parse_request_token_from_url(self, redirect_url: str) -> Optional[str]:
        """Parse request token from redirect URL with enhanced validation"""
        try:
            if not redirect_url or not isinstance(redirect_url, str):
                return None
                
            redirect_url = redirect_url.strip()
            
            # Multiple patterns to catch request_token
            patterns = [
                "request_token=",
                "requestToken=", 
                "token=",
                "rt="
            ]
            
            for pattern in patterns:
                if pattern in redirect_url:
                    token_part = redirect_url.split(pattern)[1]
                    # Handle parameters after token
                    request_token = token_part.split("&")[0].split("#")[0]
                    
                    # Validate token format (typically 24-32 characters)
                    if request_token and len(request_token) >= 20:
                        return request_token
            
            return None
            
        except Exception as e:
            st.error(f"❌ Error parsing request token: {e}")
            return None
    
    def create_signature(self, api_key: str, request_token: str, api_secret: str) -> str:
        """Create signature exactly as per Goodwill API documentation"""
        try:
            # Exactly as per documentation: checksum = api_key + request_token + api_secret
            checksum = f"{api_key}{request_token}{api_secret}"
            # Create SHA-256 hash
            signature = hashlib.sha256(checksum.encode('utf-8')).hexdigest()
            return signature
        except Exception as e:
            st.error(f"❌ Signature creation error: {e}")
            return ""
    
    def login_with_request_token(self, api_key: str, request_token: str, api_secret: str) -> bool:
        """
        Complete login using request token - PRIMARY METHOD
        Implements exact flow from developer.gwcindia.in/api/
        """
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Validate inputs
            if not all([api_key, request_token, api_secret]):
                st.error("❌ Missing required fields for authentication")
                return False
            
            # Create signature exactly as per documentation
            signature = self.create_signature(api_key, request_token, api_secret)
            
            if not signature:
                st.error("❌ Failed to create signature")
                return False
            
            # Prepare login-response payload as per documentation
            payload = {
                "api_key": api_key,
                "request_token": request_token,
                "signature": signature
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "FlyingBuddha-ScalpingBot-Fixed/2.0"
            }
            
            st.info("🔄 Authenticating with Goodwill API...")
            
            # Make the login-response API call
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
                    
                    # Extract session details as per documentation
                    self.client_id = user_data.get('clnt_id')
                    self.access_token = user_data.get('access_token')
                    self.user_session_id = user_data.get('usersessionid')
                    self.user_profile = user_data
                    
                    if self.access_token and self.client_id:
                        self.is_connected = True
                        self.connection_method = "direct_api_token"
                        self.last_login_time = datetime.now()
                        
                        # Store in session state
                        st.session_state["gw_logged_in"] = True
                        st.session_state["gw_access_token"] = self.access_token
                        st.session_state["gw_client_id"] = self.client_id
                        st.session_state["gw_connection"] = self.connection_method
                        st.session_state["gw_user_session_id"] = self.user_session_id
                        st.session_state["gw_user_profile"] = self.user_profile
                        
                        # Display success information
                        user_name = user_data.get('name', 'Unknown')
                        user_email = user_data.get('email', 'Unknown')
                        exchanges = user_data.get('exarr', [])
                        
                        st.success(f"✅ Connected Successfully via Request Token!")
                        st.info(f"👤 **User:** {user_name}")
                        st.info(f"📧 **Email:** {user_email}")
                        st.info(f"🏦 **Client ID:** {self.client_id}")
                        st.info(f"📊 **Exchanges:** {', '.join(exchanges[:5])}{'...' if len(exchanges) > 5 else ''}")
                        
                        return True
                    else:
                        st.error("❌ Missing access_token or client_id in response")
                        return False
                else:
                    error_msg = data.get('error_msg', 'Authentication failed')
                    error_type = data.get('error_type', 'Unknown')
                    st.error(f"❌ API Error: {error_msg} (Type: {error_type})")
                    
                    # Show specific error solutions
                    if "Invalid Signature" in error_msg:
                        st.error("🔧 **Solution:** Verify your API Secret is correct")
                    elif "Invalid Request Token" in error_msg:
                        st.error("🔧 **Solution:** Get a fresh request token from the login URL")
                    elif "Invalid API key" in error_msg:
                        st.error("🔧 **Solution:** Check your API Key from developer.gwcindia.in")
                    
                    return False
            else:
                st.error(f"❌ HTTP Error {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. Please try again.")
            return False
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error. Check your internet connection.")
            return False
        except Exception as e:
            st.error(f"❌ Login error: {str(e)}")
            return False
    
    def try_gwcmodel_login(self, api_key: str, user_id: str, password: str, totp: str = None) -> bool:
        """
        Enhanced gwcmodel login with multiple method attempts
        """
        success, message = self.initialize_gwcmodel()
        if not success:
            st.warning(f"⚠️ gwcmodel initialization: {message}")
            return False
        
        try:
            st.info(f"🔄 Trying gwcmodel.{GWC_CLASS_NAME} authentication...")
            
            # Store credentials
            self.stored_credentials = {
                'api_key': api_key,
                'user_id': user_id,
                'password': password,
                'totp': totp
            }
            
            # Multiple login parameter formats for different gwcmodel versions
            login_params_formats = [
                # Format 1: Standard gwcmodel format
                {
                    'userid': user_id,
                    'password': password,
                    'twoFA': totp,
                    'vendor_code': api_key,
                    'api_secret': self.api_secret or '',
                    'imei': 'abc1234'
                },
                # Format 2: Alternative format
                {
                    'api_key': api_key,
                    'user_id': user_id,
                    'password': password,
                    'totp': totp
                },
                # Format 3: Simple format
                {
                    'uid': user_id,
                    'pwd': password,
                    'factor2': totp,
                    'vc': api_key
                },
                # Format 4: NorenApi format
                {
                    'userid': user_id,
                    'password': password,
                    'twoFA': totp,
                    'vendor_code': api_key,
                    'api_secret': self.api_secret or '',
                    'imei': 'abc1234'
                }
            ]
            
            # Try different login method names
            login_methods = [
                'login', 'authenticate', 'connect', 'session_login', 
                'authorize', 'start_session', 'create_session'
            ]
            
            for method_name in login_methods:
                if hasattr(self.gwc_client, method_name):
                    method = getattr(self.gwc_client, method_name)
                    
                    for i, params in enumerate(login_params_formats):
                        try:
                            # Filter out None/empty values
                            filtered_params = {k: v for k, v in params.items() if v}
                            
                            st.info(f"🔄 Trying {method_name}() format {i+1}: {list(filtered_params.keys())}")
                            
                            # Try the login method
                            response = method(**filtered_params)
                            
                            if self._process_gwcmodel_response(response, user_id):
                                self.connection_method = f"gwcmodel_{method_name}"
                                st.success(f"✅ Connected via gwcmodel.{method_name}()!")
                                return True
                                
                        except Exception as e:
                            st.warning(f"⚠️ {method_name}() format {i+1} failed: {str(e)}")
                            continue
            
            st.warning("⚠️ All gwcmodel login methods failed. Try request token method.")
            return False
                
        except Exception as e:
            st.warning(f"⚠️ gwcmodel error: {str(e)}")
            return False
    
    def _process_gwcmodel_response(self, response, user_id: str) -> bool:
        """Enhanced gwcmodel response processing"""
        try:
            if not response:
                return False
            
            # Handle different response types
            if isinstance(response, dict):
                # Check for success indicators
                success_indicators = ['success', 'Success', 'OK', 'Ok']
                status_fields = ['status', 'stat', 'Status', 'Stat']
                
                is_success = False
                for status_field in status_fields:
                    if response.get(status_field) in success_indicators:
                        is_success = True
                        break
                
                if is_success or response.get('access_token') or response.get('susertoken'):
                    # Extract data section
                    data = response.get('data', response.get('Data', response))
                    
                    # Extract tokens with multiple possible field names
                    self.access_token = (
                        data.get('access_token') or data.get('AccessToken') or
                        data.get('session_token') or data.get('SessionToken') or
                        data.get('susertoken') or data.get('token') or
                        response.get('access_token') or response.get('susertoken')
                    )
                    
                    self.client_id = (
                        data.get('client_id') or data.get('clnt_id') or
                        data.get('ClientId') or data.get('actid') or
                        data.get('uid') or data.get('userid') or
                        response.get('actid') or user_id
                    )
                    
                    self.user_session_id = (
                        data.get('usersessionid') or data.get('session_id') or
                        response.get('usersessionid')
                    )
                    
                    if self.access_token:
                        self.is_connected = True
                        self.last_login_time = datetime.now()
                        self.user_profile = data if isinstance(data, dict) else response
                        
                        # Store in session state
                        st.session_state["gw_logged_in"] = True
                        st.session_state["gw_access_token"] = self.access_token
                        st.session_state["gw_client_id"] = self.client_id
                        st.session_state["gw_connection"] = self.connection_method
                        st.session_state["gw_user_profile"] = self.user_profile
                        
                        return True
                else:
                    error_msg = (
                        response.get('error_msg') or response.get('emsg') or 
                        response.get('message') or 'Unknown error'
                    )
                    st.warning(f"⚠️ gwcmodel response error: {error_msg}")
            
            elif isinstance(response, str):
                # Handle string responses
                if any(word in response.lower() for word in ['success', 'ok', 'login']):
                    self.access_token = 'CONNECTED_VIA_GWCMODEL'
                    self.client_id = user_id
                    self.is_connected = True
                    self.last_login_time = datetime.now()
                    
                    st.session_state["gw_logged_in"] = True
                    st.session_state["gw_access_token"] = self.access_token
                    st.session_state["gw_client_id"] = self.client_id
                    st.session_state["gw_connection"] = self.connection_method
                    
                    return True
            
            return False
            
        except Exception as e:
            st.warning(f"⚠️ Response processing error: {str(e)}")
            return False
    
    def get_headers(self) -> Dict:
        """Get authenticated headers for API calls"""
        if not self.access_token or not self.api_key:
            return {}
        
        return {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "FlyingBuddha-ScalpingBot-Fixed/2.0"
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection with detailed diagnostics"""
        if not self.is_connected:
            return False, "Not connected"
        
        try:
            # Test 1: Try gwcmodel methods if available
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                test_methods = ['profile', 'get_profile', 'user_profile', 'user_details', 'get_balance']
                
                for method_name in test_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            response = method()
                            if response:
                                return True, f"gwcmodel.{method_name}() successful"
                        except Exception:
                            continue
            
            # Test 2: Direct API test
            headers = self.get_headers()
            if headers:
                response = requests.get(f"{self.base_url}/profile", headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return True, "Direct API profile call successful"
                    else:
                        return False, f"API returned error: {data.get('error_msg', 'Unknown')}"
                else:
                    return False, f"HTTP {response.status_code}: {response.text[:100]}"
            
            return False, "No valid headers for API call"
            
        except Exception as e:
            return False, f"Connection test error: {str(e)}"
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Enhanced quote fetching with multiple methods"""
        if not self.is_connected:
            return None
        
        try:
            # Method 1: Direct API call (most reliable)
            token = self._get_symbol_token(symbol, exchange)
            if token:
                headers = self.get_headers()
                payload = {"exchange": exchange, "token": token}
                
                response = requests.post(
                    f"{self.base_url}/getquote",
                    json=payload,
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
                            'change': float(quote_data.get('change', 0)),
                            'change_per': float(quote_data.get('change_per', 0)),
                            'timestamp': datetime.now()
                        }
            
            # Method 2: Try gwcmodel if available
            if self.gwc_client and self.connection_method.startswith('gwcmodel'):
                quote_methods = ['quote', 'get_quote', 'ltp', 'get_ltp', 'get_quotes']
                
                for method_name in quote_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            
                            # Try different parameter formats
                            param_formats = [
                                {'exchange': exchange, 'tradingsymbol': f"{symbol}-EQ"},
                                {'exchange': exchange, 'symbol': f"{symbol}-EQ"},
                                {'exch': exchange, 'tsym': f"{symbol}-EQ"},
                                {'token': token, 'exchange': exchange} if token else None
                            ]
                            
                            for params in param_formats:
                                if params:
                                    try:
                                        response = method(**params)
                                        if response and isinstance(response, dict):
                                            data = response.get('data', response)
                                            price = data.get('ltp') or data.get('last_price') or data.get('price')
                                            if price and float(price) > 0:
                                                return {
                                                    'price': float(price),
                                                    'volume': int(data.get('volume', 0)),
                                                    'high': float(data.get('high', price)),
                                                    'low': float(data.get('low', price)),
                                                    'open': float(data.get('open', price)),
                                                    'timestamp': datetime.now()
                                                }
                                    except Exception:
                                        continue
                        except Exception:
                            continue
            
            return None
            
        except Exception as e:
            print(f"Quote error for {symbol}: {e}")
            return None
    
    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """Get symbol token using fetchsymbol API with caching"""
        try:
            # Use session state for caching tokens
            cache_key = f"token_{symbol}_{exchange}"
            if cache_key in st.session_state:
                return st.session_state[cache_key]
            
            headers = self.get_headers()
            if not headers:
                return None
            
            payload = {"s": symbol}
            
            response = requests.post(
                f"{self.base_url}/fetchsymbol",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    results = data.get('data', [])
                    for result in results:
                        if (result.get('exchange') == exchange and 
                            result.get('symbol') == f"{symbol}-EQ"):
                            token = result.get('token')
                            # Cache the token
                            st.session_state[cache_key] = token
                            return token
            
            return None
            
        except Exception:
            return None
    
    def place_order(self, symbol: str, action: str, quantity: int, price: float, 
                   order_type: str = "MKT", product: str = "MIS") -> Optional[str]:
        """Enhanced order placement with validation"""
        if not self.is_connected:
            st.error("❌ Not connected to Goodwill")
            return None
        
        try:
            # Validate inputs
            if quantity <= 0:
                st.error("❌ Invalid quantity")
                return None
            
            if price < 0:
                st.error("❌ Invalid price")
                return None
            
            headers = self.get_headers()
            if not headers:
                st.error("❌ Authentication headers not available")
                return None
            
            # Prepare order payload as per documentation
            order_payload = {
                "tsym": f"{symbol}-EQ",
                "exchange": "NSE",
                "trantype": action.upper(),  # B or S
                "validity": "DAY",
                "pricetype": order_type,  # MKT, L, SL-L, SL-M
                "qty": str(quantity),
                "discqty": "0",
                "price": str(price) if order_type != "MKT" else "0",
                "trgprc": "0",
                "product": product,  # MIS, CNC, NRML
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
                    st.success(f"🎯 Order Placed: {action} {quantity} {symbol} @ ₹{price:.2f} | ID: {order_id}")
                    return order_id
                else:
                    error_msg = data.get('error_msg', 'Order failed')
                    st.error(f"❌ Order Failed: {error_msg}")
            else:
                st.error(f"❌ Order HTTP Error: {response.status_code}")
            
            return None
            
        except Exception as e:
            st.error(f"❌ Order error: {str(e)}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.is_connected:
            return []
        
        try:
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
    
    def get_profile(self) -> Optional[Dict]:
        """Get user profile with enhanced error handling"""
        if not self.is_connected:
            return None
        
        try:
            headers = self.get_headers()
            if headers:
                response = requests.get(f"{self.base_url}/profile", headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return data.get('data', {})
                    else:
                        st.warning(f"Profile API error: {data.get('error_msg', 'Unknown error')}")
            
            return self.user_profile  # Return cached profile if API fails
            
        except Exception as e:
            st.warning(f"Profile fetch error: {e}")
            return self.user_profile
    
    def logout(self) -> bool:
        """Enhanced logout with complete cleanup"""
        try:
            # API logout
            headers = self.get_headers()
            if headers:
                try:
                    response = requests.get(f"{self.base_url}/logout", headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            st.info("✅ API logout successful")
                except Exception:
                    pass  # Continue with cleanup even if API logout fails
            
            # gwcmodel logout
            if self.gwc_client:
                logout_methods = ['logout', 'disconnect', 'close', 'end_session']
                for method_name in logout_methods:
                    if hasattr(self.gwc_client, method_name):
                        try:
                            method = getattr(self.gwc_client, method_name)
                            method()
                            break
                        except Exception:
                            continue
            
            # Clear all data
            self.gwc_client = None
            self.access_token = None
            self.is_connected = False
            self.client_id = None
            self.connection_method = None
            self.user_profile = None
            self.stored_credentials = {}
            
            # Clear session state
            session_keys = [
                "gw_logged_in", "gw_access_token", "gw_client_id", 
                "gw_connection", "gw_user_session_id", "gw_user_profile"
            ]
            for key in session_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear token cache
            token_keys = [k for k in st.session_state.keys() if k.startswith("token_")]
            for key in token_keys:
                del st.session_state[key]
            
            return True
            
        except Exception as e:
            st.warning(f"Logout error: {e}")
            return False

# ==================== ENHANCED 8-STRATEGY SYSTEM ====================

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
        
        # Enhanced caching system
        self.analysis_cache = {}
        self.cache_timeout = 30  # seconds
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Enhanced market condition analysis with error handling"""
        
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
            
            # Enhanced technical indicators
            sma_5 = closes.rolling(5).mean()
            sma_20 = closes.rolling(20).mean()
            rsi = self._calculate_rsi(closes, 14)
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(closes, 20)
            vwap = self._calculate_vwap(data)
            
            current_price = closes.iloc[-1]
            
            # Enhanced market condition analysis
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
            if len(self.analysis_cache) > 50:
                old_keys = [k for k in self.analysis_cache.keys() 
                           if int(k.split('_')[-1]) < int(time.time() // self.cache_timeout) - 10]
                for key in old_keys:
                    del self.analysis_cache[key]
            
            return conditions
            
        except Exception as e:
            print(f"Market analysis error for {symbol}: {e}")
            return self._get_default_conditions()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI with error handling"""
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
        """Calculate Bollinger Bands with error handling"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.fillna(sma), lower.fillna(sma), sma.fillna(prices)
        except:
            return prices, prices, prices
    
    def _calculate_vwap(self, data):
        """Calculate VWAP with error handling"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap.fillna(data['Close'])
        except:
            return data['Close']
    
    def _check_support_resistance(self, data, level_type='support'):
        """Enhanced support/resistance detection"""
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
        """Return safe default conditions"""
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
        """Enhanced strategy selection with improved scoring"""
        
        strategy_scores = {}
        
        # Strategy 1: Momentum Breakout
        strategy_scores[1] = (
            (60 if market_conditions['trending'] else 10) +
            (30 if market_conditions['volume_surge'] else 5) +
            (20 if market_conditions['volatility'] > 1.0 else 0) +
            self._get_performance_bonus(1)
        )
        
        # Strategy 2: Mean Reversion
        strategy_scores[2] = (
            (70 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 15) +
            (25 if market_conditions['consolidating'] else 5) +
            (15 if not market_conditions['trending'] else 0) +
            self._get_performance_bonus(2)
        )
        
        # Strategy 3: Volume Spike
        strategy_scores[3] = (
            (80 if market_conditions['volume_surge'] else 10) +
            (15 if market_conditions['volatile'] else 5) +
            (10 if market_conditions['news_driven'] else 0) +
            self._get_performance_bonus(3)
        )
        
        # Strategy 4: Bollinger Squeeze
        strategy_scores[4] = (
            (70 if market_conditions['bb_squeeze'] else 10) +
            (20 if market_conditions['consolidating'] else 5) +
            (10 if market_conditions['volatility'] < 1.0 else 0) +
            self._get_performance_bonus(4)
        )
        
        # Strategy 5: RSI Divergence
        strategy_scores[5] = (
            (60 if market_conditions['rsi_oversold'] or market_conditions['rsi_overbought'] else 20) +
            (25 if market_conditions['trending'] else 10) +
            (15 if 1.0 < market_conditions['volatility'] < 2.5 else 5) +
            self._get_performance_bonus(5)
        )
        
        # Strategy 6: VWAP Touch
        strategy_scores[6] = (
            (60 if market_conditions['vwap_distance'] < 0.003 else 15) +
            (25 if market_conditions['volume_surge'] else 10) +
            (15 if market_conditions['trending'] else 5) +
            self._get_performance_bonus(6)
        )
        
        # Strategy 7: Support/Resistance
        strategy_scores[7] = (
            (70 if market_conditions['near_support'] or market_conditions['near_resistance'] else 15) +
            (20 if market_conditions['consolidating'] else 10) +
            (10 if not market_conditions['trending'] else 5) +
            self._get_performance_bonus(7)
        )
        
        # Strategy 8: News Momentum
        strategy_scores[8] = (
            (90 if market_conditions['news_driven'] else 10) +
            (10 if market_conditions['volatile'] else 5) +
            (5 if market_conditions['volume_surge'] else 0) +
            self._get_performance_bonus(8)
        )
        
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        return best_strategy, strategy_scores
    
    def _get_performance_bonus(self, strategy_id: int) -> float:
        """Enhanced performance bonus calculation"""
        stats = self.strategy_stats[strategy_id]
        
        if stats['trades'] < 3:
            return 5  # Neutral bonus for new strategies
        
        # Performance bonus (0-25 points)
        success_bonus = (stats['success_rate'] / 100) * 20
        
        # Recent usage penalty for diversity
        if stats['last_used']:
            hours_since = (datetime.now() - stats['last_used']).total_seconds() / 3600
            recency_bonus = min(hours_since * 0.5, 5)
        else:
            recency_bonus = 5
        
        return success_bonus + recency_bonus
    
    def get_strategy_signals(self, strategy_id: int, symbol: str, market_conditions: Dict) -> Dict:
        """Enhanced strategy signal generation"""
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 10:
                return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Insufficient data'}
            
            current_price = data['Close'].iloc[-1]
            
            # Route to specific strategy methods
            strategy_methods = {
                1: self._momentum_breakout_signals,
                2: self._mean_reversion_signals,
                3: self._volume_spike_signals,
                4: self._bollinger_squeeze_signals,
                5: self._rsi_divergence_signals,
                6: self._vwap_touch_signals,
                7: self._support_resistance_signals,
                8: self._news_momentum_signals
            }
            
            if strategy_id in strategy_methods:
                return strategy_methods[strategy_id](data, market_conditions)
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'Invalid strategy'}
            
        except Exception as e:
            print(f"Strategy signal error: {e}")
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    # Individual strategy signal methods (keeping original logic but with enhanced error handling)
    def _momentum_breakout_signals(self, data, conditions):
        """Enhanced momentum breakout signals"""
        try:
            current_price = data['Close'].iloc[-1]
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            breakout_up = current_price > high_20 * 1.002
            breakout_down = current_price < low_20 * 0.998
            
            if breakout_up and conditions['volume_surge']:
                return {
                    'signal': 1, 'confidence': 0.85, 'entry_price': current_price,
                    'stop_pct': 0.008, 'reason': 'Upward breakout with volume'
                }
            elif breakout_down and conditions['volume_surge'] and conditions['trend_direction'] < 0:
                return {
                    'signal': -1, 'confidence': 0.85, 'entry_price': current_price,
                    'stop_pct': 0.008, 'reason': 'Downward breakout with volume'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No breakout'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _mean_reversion_signals(self, data, conditions):
        """Enhanced mean reversion signals"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = conditions['rsi']
            
            if rsi < 25 and conditions['bb_lower_touch']:
                return {
                    'signal': 1, 'confidence': 0.80, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'Oversold mean reversion'
                }
            elif rsi > 75 and conditions['bb_upper_touch']:
                return {
                    'signal': -1, 'confidence': 0.80, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'Overbought mean reversion'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No reversion signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _volume_spike_signals(self, data, conditions):
        """Enhanced volume spike signals"""
        try:
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].pct_change().iloc[-1] * 100
            
            if conditions['volume_surge'] and abs(price_change) > 0.4:
                signal = 1 if price_change > 0 else -1
                confidence = min(0.90, conditions['volume_ratio'] * 0.3)
                return {
                    'signal': signal, 'confidence': confidence, 'entry_price': current_price,
                    'stop_pct': 0.007, 'reason': f'Volume spike with {abs(price_change):.2f}% move'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No volume spike'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _bollinger_squeeze_signals(self, data, conditions):
        """Enhanced Bollinger squeeze signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['bb_squeeze'] and conditions['consolidating']:
                signal = conditions['trend_direction']
                return {
                    'signal': signal, 'confidence': 0.75, 'entry_price': current_price,
                    'stop_pct': 0.005, 'reason': 'Bollinger squeeze breakout'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No squeeze'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _rsi_divergence_signals(self, data, conditions):
        """Enhanced RSI divergence signals"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = conditions['rsi']
            
            if rsi < 30 and conditions['near_support']:
                return {
                    'signal': 1, 'confidence': 0.70, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'RSI oversold at support'
                }
            elif rsi > 70 and conditions['near_resistance']:
                return {
                    'signal': -1, 'confidence': 0.70, 'entry_price': current_price,
                    'stop_pct': 0.006, 'reason': 'RSI overbought at resistance'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No RSI signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _vwap_touch_signals(self, data, conditions):
        """Enhanced VWAP touch signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['vwap_distance'] < 0.003:
                if conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': 1, 'confidence': 0.75, 'entry_price': current_price,
                        'stop_pct': 0.005, 'reason': 'VWAP support with volume'
                    }
                elif not conditions['above_vwap'] and conditions['volume_surge']:
                    return {
                        'signal': -1, 'confidence': 0.75, 'entry_price': current_price,
                        'stop_pct': 0.005, 'reason': 'VWAP resistance with volume'
                    }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No VWAP signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _support_resistance_signals(self, data, conditions):
        """Enhanced support/resistance signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['near_support'] and conditions['rsi'] < 45:
                return {
                    'signal': 1, 'confidence': 0.80, 'entry_price': current_price,
                    'stop_pct': 0.004, 'reason': 'Support level bounce'
                }
            elif conditions['near_resistance'] and conditions['rsi'] > 55:
                return {
                    'signal': -1, 'confidence': 0.80, 'entry_price': current_price,
                    'stop_pct': 0.004, 'reason': 'Resistance level rejection'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No S/R signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def _news_momentum_signals(self, data, conditions):
        """Enhanced news momentum signals"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if conditions['news_driven']:
                signal = conditions['trend_direction']
                confidence = min(0.95, conditions['volume_ratio'] * 0.35)
                
                return {
                    'signal': signal, 'confidence': confidence, 'entry_price': current_price,
                    'stop_pct': 0.01, 'reason': f'News momentum (Vol: {conditions["volume_ratio"]:.1f}x)'
                }
            
            return {'signal': 0, 'confidence': 0, 'entry_price': current_price, 'stop_pct': 0.005, 'reason': 'No news signal'}
        except:
            return {'signal': 0, 'confidence': 0, 'entry_price': 0, 'stop_pct': 0.005, 'reason': 'Error'}
    
    def update_strategy_performance(self, strategy_id: int, trade_result: Dict):
        """Enhanced strategy performance tracking"""
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
    """Enhanced production-ready scalping bot"""
    
    def __init__(self):
        self.api = CompleteGoodwillIntegration()
        self.strategies = EnhancedScalpingStrategies()
        self.is_running = False
        self.capital = 100000.0
        self.positions = {}
        self.trades = []
        self.signals = []
        self.pnl = 0.0
        self.mode = "PAPER"
        
        # Enhanced trading parameters
        self.max_positions = 4
        self.risk_per_trade = 0.15  # 15% per trade
        self.max_hold_time = 180  # 3 minutes
        self.risk_reward_ratio = 2.0  # 1:2
        
        # Enhanced risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_drawdown_limit = 0.10  # 10% max drawdown
        self.daily_pnl = 0.0
        self.peak_capital = self.capital
        self.current_drawdown = 0.0
        
        # Enhanced symbol list
        self.symbols = [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS",
            "WIPRO", "LT", "TITAN", "MARUTI", "BHARTIARTL"
        ]
        
        self.init_database()
    
    def init_database(self):
        """Enhanced database initialization"""
        try:
            self.conn = sqlite3.connect('production_scalping_trades.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Enhanced trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, action TEXT,
                    entry_price REAL, exit_price REAL, quantity INTEGER,
                    pnl REAL, hold_time REAL, strategy_id INTEGER,
                    strategy_name TEXT, exit_reason TEXT, confidence REAL,
                    order_id TEXT, mode TEXT, connection_method TEXT,
                    market_conditions TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER, winning_trades INTEGER,
                    total_pnl REAL, max_drawdown REAL, 
                    capital_used REAL, best_strategy TEXT,
                    connection_method TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_id INTEGER PRIMARY KEY,
                    strategy_name TEXT, total_trades INTEGER,
                    wins INTEGER, total_pnl REAL, success_rate REAL,
                    avg_profit REAL, avg_loss REAL, avg_hold_time REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
    
    def check_risk_limits(self) -> bool:
        """Enhanced risk limit checking"""
        try:
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
        except Exception as e:
            st.error(f"Risk check error: {e}")
            return False
    
    def scan_for_opportunities(self):
        """Enhanced opportunity scanning with better error handling"""
        opportunities = []
        
        for symbol in self.symbols[:8]:  # Scan first 8 symbols for performance
            try:
                # Get live data with multiple sources
                live_data = None
                data_source = "yfinance"
                
                if self.api.is_connected:
                    live_data = self.api.get_quote(symbol)
                    if live_data and live_data['price'] > 0:
                        data_source = f"Goodwill ({self.api.connection_method})"
                
                # Fallback to yfinance if needed
                if not live_data or live_data['price'] <= 0:
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
                            data_source = "yfinance (fallback)"
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
                
                # Enhanced signal filtering
                if (signals['signal'] != 0 and 
                    signals['confidence'] > 0.75 and 
                    len(self.positions) < self.max_positions):
                    
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
        
        # Sort by confidence and return top opportunities
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:3]  # Top 3 opportunities
    
    def calculate_position_size(self, symbol, entry_price, stop_pct):
        """Enhanced position sizing with improved risk management"""
        try:
            # Base risk amount
            risk_amount = self.capital * self.risk_per_trade
            
            # Adjust for current drawdown
            if self.current_drawdown > 0:
                risk_reduction = min(0.5, self.current_drawdown / (self.capital * 0.05))
                risk_amount *= (1 - risk_reduction)
            
            # Adjust for market volatility
            if stop_pct > 0.008:  # High volatility
                risk_amount *= 0.8
            elif stop_pct < 0.004:  # Low volatility
                risk_amount *= 1.2
            
            # Calculate quantity based on stop loss
            stop_loss_amount = entry_price * stop_pct
            
            if stop_loss_amount > 0:
                quantity = int(risk_amount / stop_loss_amount)
                
                # Apply position size limits
                max_quantity = min(2000, int(self.capital * 0.12 / entry_price))
                quantity = max(1, min(quantity, max_quantity))
                
                return quantity
            
            return 1
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 1
    
    def execute_entry(self, opportunity):
        """Enhanced entry execution with better validation"""
        try:
            # Pre-execution checks
            if not self.check_risk_limits():
                return None
            
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
            action = "B" if signal > 0 else "S"
            
            # Enhanced stop loss and target calculation
            if action == "B":
                stop_loss = entry_price * (1 - stop_pct)
                target = entry_price * (1 + (stop_pct * self.risk_reward_ratio))
            else:
                stop_loss = entry_price * (1 + stop_pct)
                target = entry_price * (1 - (stop_pct * self.risk_reward_ratio))
            
            # Place order
            order_id = None
            
            if self.mode == "LIVE" and self.api.is_connected:
                order_id = self.api.place_order(symbol, action, quantity, entry_price, "MKT", "MIS")
                if not order_id:
                    st.error(f"❌ Failed to place live order for {symbol}")
                    return None
            else:
                order_id = f"PAPER_{int(datetime.now().timestamp())}"
                st.info(f"📝 PAPER: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name} | Confidence: {confidence:.1%}")
            
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
                    'data_source': opportunity['data_source'],
                    'market_conditions': opportunity['market_conditions']
                }
                
                st.success(f"🚀 Position Opened: {action} {quantity} {symbol} @ ₹{entry_price:.2f} | Strategy: {strategy_name}")
                return position_id
            
            return None
            
        except Exception as e:
            st.error(f"Entry execution error: {e}")
            return None
    
    def update_position_tracking(self, position_id, current_price):
        """Enhanced position tracking with advanced trailing stop"""
        try:
            position = self.positions[position_id]
            
            # Calculate current P&L
            if position['action'] == 'B':
                current_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                current_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Update profit tracking
            position['highest_profit'] = max(position['highest_profit'], current_pnl)
            position['lowest_profit'] = min(position['lowest_profit'], current_pnl)
            
            # Enhanced dynamic trailing stop
            if current_pnl > 0:
                profit_ratio = current_pnl / (position['entry_price'] * position['quantity'] * position['stop_pct'])
                
                # Progressive trail tightening
                if profit_ratio > 2:  # 2x risk achieved
                    trail_factor = 0.3  # Tighten to 30% of original stop
                elif profit_ratio > 1:  # 1x risk achieved
                    trail_factor = 0.5  # Tighten to 50% of original stop
                else:
                    trail_factor = 0.8  # Keep 80% of original stop
                
                if position['action'] == 'B':
                    new_stop = current_price * (1 - position['stop_pct'] * trail_factor)
                    position['trailing_stop'] = max(position['trailing_stop'], new_stop)
                else:
                    new_stop = current_price * (1 + position['stop_pct'] * trail_factor)
                    position['trailing_stop'] = min(position['trailing_stop'], new_stop)
                    
        except Exception as e:
            print(f"Position tracking error: {e}")
    
    def check_exit_conditions(self, position_id, current_price):
        """Enhanced exit condition checking"""
        try:
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
            
            # Enhanced risk management exits
            current_pnl = (
                (current_price - position['entry_price']) * position['quantity'] 
                if position['action'] == 'B' 
                else (position['entry_price'] - current_price) * position['quantity']
            )
            
            position_risk = abs(current_pnl) / self.capital
            
            if position_risk > self.risk_per_trade * 1.8:  # 180% of intended risk
                return "RISK_LIMIT"
            
            # Profit protection exit
            if (position['highest_profit'] > 0 and 
                current_pnl < position['highest_profit'] * 0.3 and 
                hold_time > 60):  # Protect 70% of peak profit after 1 minute
                return "PROFIT_PROTECTION"
            
            return None
            
        except Exception as e:
            print(f"Exit condition check error: {e}")
            return None
    
    def close_position(self, position_id, exit_price, exit_reason):
        """Enhanced position closing with comprehensive logging"""
        if position_id not in self.positions:
            return
        
        try:
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
            
            # Create comprehensive trade record
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
                'connection_method': self.api.connection_method or 'yfinance',
                'market_conditions': json.dumps(position.get('market_conditions', {})),
                'data_source': position['data_source']
            }
            
            self.trades.append(trade_record)
            
            # Enhanced database logging
            try:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO trades VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
                    )
                ''', (
                    trade_record['timestamp'], trade_record['symbol'], trade_record['action'],
                    trade_record['entry_price'], trade_record['exit_price'], trade_record['quantity'],
                    trade_record['pnl'], trade_record['hold_time'], trade_record['strategy_id'],
                    trade_record['strategy_name'], trade_record['exit_reason'], trade_record['confidence'],
                    trade_record['order_id'], trade_record['mode'], trade_record['connection_method'],
                    trade_record['market_conditions']
                ))
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
                st.info(f"📝 PAPER EXIT: {exit_action} {position['quantity']} {position['symbol']} @ ₹{exit_price:.2f}")
            
            pnl_color = "🟢" if pnl >= 0 else "🔴"
            st.info(f"✅ Position Closed: {position['symbol']} | {pnl_color} P&L: ₹{pnl:.2f} | Strategy: {position['strategy_name']} | Reason: {exit_reason}")
            
        except Exception as e:
            st.error(f"Position closing error: {e}")
    
    def run_trading_cycle(self):
        """Enhanced main trading cycle with better error handling"""
        try:
            # Check risk limits before any trading
            if not self.check_risk_limits():
                self.is_running = False
                st.error("🛑 Trading stopped due to risk limits")
                return
            
            # Check exit conditions for existing positions first
            positions_to_close = []
            
            for pos_id in list(self.positions.keys()):
                position = self.positions[pos_id]
                
                # Get current price from best available source
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
                except Exception as e:
                    print(f"Price fetch error for {position['symbol']}: {e}")
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(pos_id, current_price)
                if exit_reason:
                    positions_to_close.append((pos_id, current_price, exit_reason))
            
            # Close positions that need to be closed
            for pos_id, exit_price, exit_reason in positions_to_close:
                self.close_position(pos_id, exit_price, exit_reason)
                time.sleep(0.5)  # Brief delay between closes
            
            # Scan for new opportunities if we have room
            if len(self.positions) < self.max_positions:
                opportunities = self.scan_for_opportunities()
                
                for opp in opportunities:
                    if len(self.positions) < self.max_positions:
                        self.execute_entry(opp)
                        time.sleep(1)  # Prevent rapid-fire orders
                        break  # Only take one position per cycle
                        
        except Exception as e:
            st.error(f"Trading cycle error: {e}")
    
    def get_performance_metrics(self):
        """Enhanced performance metrics calculation"""
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_trade': 0, 'max_profit': 0, 'max_loss': 0, 'avg_hold_time': 0,
                'best_strategy': 'N/A', 'strategy_breakdown': {}, 'sharpe_ratio': 0,
                'max_drawdown': round(self.current_drawdown, 2), 'profit_factor': 0, 
                'daily_pnl': round(self.daily_pnl, 2), 'winning_trades': 0, 'losing_trades': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0
            }
        
        pnls = [trade['pnl'] for trade in self.trades]
        hold_times = [trade['hold_time'] for trade in self.trades]
        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p < 0])
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
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
        total_profits = sum(wins) if wins else 1
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_profits / total_losses
        
        # Sharpe ratio (simplified)
        sharpe_ratio = 0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)  # Annualized
        
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
            'losing_trades': losing_trades,
            'avg_win': round(np.mean(wins), 2) if wins else 0,
            'avg_loss': round(np.mean(losses), 2) if losses else 0,
            'largest_win': round(max(wins), 2) if wins else 0,
            'largest_loss': round(min(losses), 2) if losses else 0
        }

# ==================== STREAMLIT APPLICATION ====================

def show_enhanced_disclaimer():
    """Show enhanced production disclaimer"""
    st.markdown("""
    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin: 10px 0;">
        <h4 style="color: #2e7d32; margin-top: 0;">✅ COMPLETE FIXED VERSION - Production Ready</h4>
        <p style="margin-bottom: 0;"><strong>This version includes all major fixes and enhancements:</strong></p>
        <ul style="margin: 10px 0;">
            <li>🔧 <strong>Enhanced gwcmodel Detection:</strong> Comprehensive class scanning and initialization</li>
            <li>🎫 <strong>Proper Request Token Flow:</strong> Complete implementation per official documentation</li>
            <li>🔗 <strong>URL Parser:</strong> Automatic request_token extraction from redirect URLs</li>
            <li>🔐 <strong>Multiple Auth Methods:</strong> gwcmodel + Direct API with robust fallbacks</li>
            <li>🎯 <strong>8 Advanced Strategies:</strong> Enhanced with improved signal generation</li>
            <li>🛡️ <strong>Production Risk Management:</strong> 5% daily loss, 10% max drawdown limits</li>
            <li>📊 <strong>Enhanced Performance Tracking:</strong> Comprehensive metrics and analytics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_gwcmodel_diagnostics():
    """Show comprehensive gwcmodel diagnostics"""
    st.sidebar.markdown("### 🔧 gwcmodel Diagnostics")
    
    status = bot.api.get_gwcmodel_status()
    
    if status['available']:
        if status['class_found']:
            st.sidebar.success(f"✅ {status['class_name']} available")
        else:
            st.sidebar.error("❌ No valid class found")
            if status['attributes']:
                st.sidebar.code(f"Found: {', '.join(status['attributes'][:3])}...")
                st.sidebar.text(f"Total: {status['total_attributes']} attributes")
    else:
        st.sidebar.error("❌ gwcmodel not installed")
        st.sidebar.code("pip install gwcmodel")

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = ProductionScalpingBot()

if 'gw_logged_in' not in st.session_state:
    st.session_state.gw_logged_in = False

bot = st.session_state.bot

# ==================== MAIN APPLICATION UI ====================

# Title and disclaimer
st.title("⚡ FlyingBuddha Scalping Bot - COMPLETE FIXED VERSION")
show_enhanced_disclaimer()

# Show diagnostics
show_gwcmodel_diagnostics()

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.header("🚀 Enhanced 8-Strategy Bot")

# Enhanced status display
connection_status = "🟢 Connected" if bot.api.is_connected else "🔴 Disconnected"
connection_method = f" ({bot.api.connection_method})" if bot.api.connection_method else ""

st.sidebar.markdown(f"""
**📊 Enhanced Status:**
- **API Status:** {connection_status}{connection_method}
- **Trading Mode:** {'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}
- **Capital:** ₹{bot.capital:,.2f}
- **Daily P&L:** ₹{bot.daily_pnl:,.2f}
- **Positions:** {len(bot.positions)}/{bot.max_positions}
- **Bot Status:** {'🟢 Running' if bot.is_running else '🔴 Stopped'}
- **Drawdown:** ₹{bot.current_drawdown:,.2f}
""")

# ==================== ENHANCED AUTHENTICATION ====================

st.subheader("🔐 Enhanced Goodwill Authentication")

# Comprehensive setup instructions
with st.expander("📋 Complete Setup Guide - FIXED VERSION", expanded=not st.session_state.gw_logged_in):
    st.markdown("""
    ### 🎯 FIXED Authentication Flow
    
    **✅ Method 1: Request Token Flow (Primary - Most Reliable)**
    1. Enter your API Key and API Secret below
    2. Click "🔗 Generate Login URL"
    3. **Open the URL in NEW TAB** and login with your Goodwill credentials
    4. After successful login, **copy the ENTIRE redirect URL** from browser
    5. Paste the complete URL in "Redirect URL" field
    6. Click "🎫 Login with Request Token"
    
    **✅ Method 2: gwcmodel (Secondary - If Available)**
    - System automatically detects available gwcmodel classes
    - Uses your trading account credentials
    - Multiple initialization methods attempted
    
    ### 📊 Your API Details
    - **API Key:** `9c155c1fff651d01513b455396af2449` 
    - **User ID:** `GWC100643`
    - **API Docs:** [developer.gwcindia.in/api](https://developer.gwcindia.in/api/)
    
    ### 🔧 What's Fixed:
    - ✅ Enhanced gwcmodel class detection
    - ✅ Proper SHA-256 signature generation  
    - ✅ Automatic request_token parsing
    - ✅ Multiple fallback authentication methods
    - ✅ Comprehensive error handling with solutions
    """)

if not st.session_state.gw_logged_in:
    # Enhanced Request Token Method
    st.markdown("### 🎫 Method 1: Request Token Authentication")
    
    auth_col1, auth_col2 = st.columns(2)
    
    with auth_col1:
        api_key = st.text_input("API Key", value="9c155c1fff651d01513b455396af2449", type="password")
        api_secret = st.text_input("API Secret", type="password", 
                                  help="Required for SHA-256 signature generation")
        
        if api_key and st.button("🔗 Generate Login URL", use_container_width=True):
            login_url = bot.api.generate_login_url(api_key)
            st.success("✅ Login URL Generated!")
            st.markdown(f"**[🔗 Click Here to Login to Goodwill]({login_url})**", unsafe_allow_html=True)
            st.code(login_url, language="text")
            st.info("👆 **IMPORTANT:** Open this URL in a NEW TAB, login, then copy the redirect URL below")
    
    with auth_col2:
        redirect_url = st.text_area(
            "Complete Redirect URL (after login)", 
            height=120,
            placeholder="After login, paste the COMPLETE URL from your browser here:\nExample: https://your-redirect-url.com?request_token=abc123xyz...",
            help="Copy the entire URL from browser address bar after successful Goodwill login"
        )
        
        if st.button("🎫 Login with Request Token", type="primary", use_container_width=True):
            if redirect_url and api_secret:
                request_token = bot.api.parse_request_token_from_url(redirect_url)
                
                if request_token:
                    st.info(f"🔍 Found Request Token: {request_token[:15]}...{request_token[-5:]}")
                    
                    with st.spinner("🔄 Authenticating with Goodwill API..."):
                        success = bot.api.login_with_request_token(api_key, request_token, api_secret)
                        if success:
                            st.session_state.gw_logged_in = True
                            st.rerun()
                else:
                    st.error("❌ Could not find request_token in URL. Please check the URL format.")
                    st.info("💡 Make sure you copied the COMPLETE URL after successful login")
            else:
                st.error("❌ Please provide both Redirect URL and API Secret")
    
    # Enhanced gwcmodel Method
    st.markdown("---")
    st.markdown("### 🔑 Method 2: gwcmodel Authentication")
    
    if GWCMODEL_AVAILABLE and GWC_CLASS:
        st.success(f"✅ gwcmodel.{GWC_CLASS_NAME} detected")
        
        gwc_col1, gwc_col2 = st.columns(2)
        
        with gwc_col1:
            gwc_api_key = st.text_input("gwcmodel API Key", value="9c155c1fff651d01513b455396af2449", type="password")
            gwc_user_id = st.text_input("User ID", value="GWC100643")
        
        with gwc_col2:
            gwc_password = st.text_input("Password", type="password")
            gwc_totp = st.text_input("TOTP/2FA Code", help="Leave empty if 2FA not enabled")
        
        if st.button("🔑 Login via gwcmodel", use_container_width=True):
            if gwc_api_key and gwc_user_id and gwc_password:
                bot.api.api_secret = api_secret if api_secret else ""
                with st.spinner("🔄 Trying gwcmodel authentication methods..."):
                    success = bot.api.try_gwcmodel_login(gwc_api_key, gwc_user_id, gwc_password, gwc_totp)
                    if success:
                        st.session_state.gw_logged_in = True
                        st.rerun()
            else:
                st.error("❌ Please fill in API Key, User ID, and Password")
    else:
        st.warning("⚠️ gwcmodel not available or no valid class found. Use Request Token method above.")
        if gwcmodel_attributes:
            st.code(f"Available attributes: {', '.join(gwcmodel_attributes[:5])}")

else:
    # Successfully connected - show enhanced connection info
    st.success(f"✅ Successfully Connected via {bot.api.connection_method}")
    
    # Enhanced connection details
    conn_col1, conn_col2, conn_col3, conn_col4 = st.columns(4)
    
    with conn_col1:
        st.metric("Client ID", bot.api.client_id or "Unknown")
    
    with conn_col2:
        connection_time = bot.api.last_login_time.strftime("%H:%M:%S") if bot.api.last_login_time else "Unknown"
        st.metric("Connected At", connection_time)
    
    with conn_col3:
        session_duration = ""
        if bot.api.last_login_time:
            duration = datetime.now() - bot.api.last_login_time
            session_duration = f"{int(duration.total_seconds() / 60)}m"
        st.metric("Session Duration", session_duration)
    
    with conn_col4:
        if st.button("🔄 Test Connection"):
            test_result, message = bot.api.test_connection()
            if test_result:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    # Enhanced connection actions
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("👤 Profile", use_container_width=True):
            profile = bot.api.get_profile()
            if profile:
                st.json(profile)
            else:
                st.warning("Could not fetch profile")
    
    with action_col2:
        if st.button("📊 Positions", use_container_width=True):
            positions = bot.api.get_positions()
            if positions:
                st.json(positions[:3])  # Show first 3
            else:
                st.info("No positions found")
    
    with action_col3:
        if st.button("💰 Balance", use_container_width=True):
            # Try to get balance via API
            try:
                headers = bot.api.get_headers()
                if headers:
                    response = requests.get(f"{bot.api.base_url}/balance", headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            st.json(data.get('data', {}))
                        else:
                            st.warning("Balance API not available")
                    else:
                        st.warning("Balance API call failed")
                else:
                    st.warning("No authentication headers")
            except Exception as e:
                st.warning(f"Balance fetch error: {e}")
    
    with action_col4:
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            if bot.api.logout():
                st.session_state.gw_logged_in = False
                st.success("✅ Logged out successfully")
                st.rerun()
            else:
                st.warning("Logout completed with warnings")
                st.session_state.gw_logged_in = False
                st.rerun()

# ==================== ENHANCED TRADING CONTROLS ====================

st.subheader("🎮 Enhanced Trading Controls")

# Mode and bot control
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if st.button("🟠 Paper Mode", disabled=(bot.mode == "PAPER"), use_container_width=True):
        bot.mode = "PAPER"
        st.success("✅ Switched to Paper Mode")
        st.rerun()

with control_col2:
    if st.button("🔴 Live Mode", disabled=(bot.mode == "LIVE"), use_container_width=True):
        if not bot.api.is_connected:
            st.error("❌ Connect to Goodwill API first")
        else:
            bot.mode = "LIVE"
            st.warning("⚠️ Live Mode - Real orders will be placed!")
            st.rerun()

with control_col3:
    if not bot.is_running:
        if st.button("🚀 START BOT", type="primary", use_container_width=True):
            if bot.mode == "LIVE" and not bot.api.is_connected:
                st.error("❌ Connect to Goodwill API for Live Mode")
            else:
                bot.is_running = True
                st.success("✅ Bot Started!")
                st.rerun()
    else:
        if st.button("⏹️ STOP BOT", type="secondary", use_container_width=True):
            bot.is_running = False
            st.warning("⏸️ Bot Stopped!")
            st.rerun()

with control_col4:
    if st.button("🔄 Manual Cycle", use_container_width=True):
        with st.spinner("Running trading cycle..."):
            bot.run_trading_cycle()
            st.rerun()

# Enhanced trading parameters
st.subheader("⚙️ Enhanced Trading Parameters")

param_col1, param_col2, param_col3, param_col4 = st.columns(4)

with param_col1:
    new_risk = st.slider("Risk per Trade (%)", 5, 25, int(bot.risk_per_trade * 100), 1)
    bot.risk_per_trade = new_risk / 100

with param_col2:
    bot.max_positions = st.slider("Max Positions", 1, 6, bot.max_positions, 1)

with param_col3:
    bot.max_hold_time = st.slider("Max Hold Time (sec)", 60, 300, bot.max_hold_time, 30)

with param_col4:
    bot.risk_reward_ratio = st.slider("Risk:Reward", 1.5, 3.0, bot.risk_reward_ratio, 0.1)

# Capital management and symbol selection
st.subheader("💰 Capital & Symbol Management")

capital_col1, capital_col2 = st.columns(2)

with capital_col1:
    new_capital = st.number_input("Capital (₹)", 10000, 10000000, int(bot.capital), 10000)
    if new_capital != bot.capital:
        bot.capital = float(new_capital)
        bot.peak_capital = max(bot.peak_capital, bot.capital)

with capital_col2:
    all_symbols = [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", 
        "BAJFINANCE", "RELIANCE", "INFY", "TCS", "ADANIPORTS",
        "WIPRO", "LT", "TITAN", "MARUTI", "BHARTIARTL", "HINDUNILVR",
        "ASIANPAINT", "NESTLEIND", "POWERGRID", "NTPC"
    ]
    
    selected_symbols = st.multiselect(
        "Trading Symbols:",
        options=all_symbols,
        default=bot.symbols,
        help="Select symbols for trading (max 15 recommended)"
    )
    
    if selected_symbols:
        bot.symbols = selected_symbols[:15]  # Limit to 15 for performance

# ==================== ENHANCED STATUS DASHBOARD ====================

st.subheader("📊 Enhanced Status Dashboard")

# Main metrics
main_col1, main_col2, main_col3, main_col4, main_col5, main_col6 = st.columns(6)

with main_col1:
    bot_status = f"{'🟢 Running' if bot.is_running else '🔴 Stopped'}"
    st.metric("Bot Status", bot_status)

with main_col2:
    api_status = f"{'🟢 Connected' if bot.api.is_connected else '🔴 Offline'}"
    st.metric("API Status", api_status)

with main_col3:
    mode_display = f"{'🔴 LIVE' if bot.mode == 'LIVE' else '🟠 PAPER'}"
    st.metric("Trading Mode", mode_display)

with main_col4:
    st.metric("Positions", f"{len(bot.positions)}/{bot.max_positions}")

with main_col5:
    pnl_color = "🟢" if bot.pnl >= 0 else "🔴"
    st.metric("Total P&L", f"₹{bot.pnl:.2f}", delta=f"{pnl_color}")

with main_col6:
    daily_color = "🟢" if bot.daily_pnl >= 0 else "🔴"
    st.metric("Daily P&L", f"₹{bot.daily_pnl:.2f}", delta=f"{daily_color}")

# ==================== ENHANCED RISK MONITORING ====================

st.subheader("🛡️ Enhanced Risk Monitoring")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    drawdown_pct = (bot.current_drawdown / bot.peak_capital) * 100 if bot.peak_capital > 0 else 0
    drawdown_color = "🟢" if drawdown_pct < 5 else "🟡" if drawdown_pct < 8 else "🔴"
    st.metric("Drawdown", f"₹{bot.current_drawdown:.2f}", delta=f"{drawdown_color} {drawdown_pct:.1f}%")

with risk_col2:
    daily_loss_pct = (abs(bot.daily_pnl) / bot.capital) * 100 if bot.daily_pnl < 0 else 0
    loss_color = "🟢" if daily_loss_pct < 2 else "🟡" if daily_loss_pct < 4 else "🔴"
    st.metric("Daily Loss %", f"{loss_color} {daily_loss_pct:.1f}%", delta="Limit: 5.0%")

with risk_col3:
    position_risk = len(bot.positions) * bot.risk_per_trade * 100
    risk_color = "🟢" if position_risk < 40 else "🟡" if position_risk < 60 else "🔴"
    st.metric("Position Risk", f"{risk_color} {position_risk:.1f}%")

with risk_col4:
    capital_utilization = (sum([p['quantity'] * p['entry_price'] for p in bot.positions.values()]) / bot.capital) * 100 if bot.positions else 0
    util_color = "🟢" if capital_utilization < 50 else "🟡" if capital_utilization < 75 else "🔴"
    st.metric("Capital Used", f"{util_color} {capital_utilization:.1f}%")

# Risk warnings
if drawdown_pct > 8:
    st.error("⚠️ **RISK ALERT:** Approaching maximum drawdown limit (10%)")
if daily_loss_pct > 4:
    st.error("⚠️ **RISK ALERT:** Approaching daily loss limit (5%)")
if position_risk > 60:
    st.warning("⚠️ **WARNING:** High position risk exposure")

# ==================== ENHANCED PERFORMANCE METRICS ====================

st.subheader("📈 Enhanced Performance Analytics")

metrics = bot.get_performance_metrics()

if metrics['total_trades'] > 0:
    # Main performance metrics
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric("Total Trades", metrics['total_trades'])
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    with perf_col2:
        st.metric("Total P&L", f"₹{metrics['total_pnl']:.2f}")
        st.metric("Avg Trade", f"₹{metrics['avg_trade']:.2f}")
    
    with perf_col3:
        st.metric("Avg Win", f"₹{metrics['avg_win']:.2f}")
        st.metric("Avg Loss", f"₹{metrics['avg_loss']:.2f}")
    
    with perf_col4:
        st.metric("Largest Win", f"₹{metrics['largest_win']:.2f}")
        st.metric("Largest Loss", f"₹{metrics['largest_loss']:.2f}")
    
    with perf_col5:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")

else:
    st.info("📊 No performance data yet. Start trading to see comprehensive analytics.")

# ==================== ENHANCED 8-STRATEGY DASHBOARD ====================

st.subheader("🎯 Enhanced 8-Strategy Performance Dashboard")

if any(stats['trades'] > 0 for stats in bot.strategies.strategy_stats.values()):
    strategy_data = []
    for strategy_id, stats in bot.strategies.strategy_stats.items():
        if stats['trades'] > 0:
            strategy_info = bot.strategies.strategies[strategy_id]
            strategy_data.append({
                'ID': strategy_id,
                'Strategy': strategy_info['name'],
                'Description': strategy_info['desc'],
                'Trades': stats['trades'],
                'Win Rate': f"{stats['success_rate']:.1f}%",
                'Total P&L': f"₹{stats['total_pnl']:.2f}",
                'Avg Profit': f"₹{stats['avg_profit']:.2f}",
                'Avg Loss': f"₹{stats['avg_loss']:.2f}",
                'Avg Hold': f"{stats['avg_hold_time']:.1f}s",
                'Last Used': stats['last_used'].strftime("%H:%M") if stats['last_used'] else "Never"
            })
    
    if strategy_data:
        df_strategies = pd.DataFrame(strategy_data)
        
        # Enhanced styling
        def style_performance(val):
            if '₹' in str(val):
                try:
                    num_val = float(str(val).replace('₹', '').replace(',', ''))
                    return 'background-color: #c8e6c9' if num_val >= 0 else 'background-color: #ffcdd2'
                except:
                    return ''
            elif '%' in str(val):
                try:
                    num_val = float(str(val).replace('%', ''))
                    return 'background-color: #c8e6c9' if num_val >= 50 else 'background-color: #fff3e0'
                except:
                    return ''
            return ''
        
        styled_strategies = df_strategies.style.applymap(style_performance, 
                                                        subset=['Win Rate', 'Total P&L', 'Avg Profit', 'Avg Loss'])
        st.dataframe(styled_strategies, use_container_width=True)
        
        # Enhanced strategy analytics
        if len(strategy_data) > 2:
            st.subheader("📊 Strategy Performance Charts")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Strategy P&L comparison
                fig_pnl = go.Figure()
                
                strategy_names = [d['Strategy'] for d in strategy_data]
                strategy_pnls = [float(d['Total P&L'].replace('₹', '')) for d in strategy_data]
                colors = [bot.strategies.strategies[d['ID']]['color'] for d in strategy_data]
                
                fig_pnl.add_trace(go.Bar(
                    x=strategy_names,
                    y=strategy_pnls,
                    marker_color=colors,
                    text=[f"₹{p:.0f}" for p in strategy_pnls],
                    textposition='auto'
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
                # Strategy win rate comparison
                fig_winrate = go.Figure()
                
                win_rates = [float(d['Win Rate'].replace('%', '')) for d in strategy_data]
                
                fig_winrate.add_trace(go.Bar(
                    x=strategy_names,
                    y=win_rates,
                    marker_color=colors,
                    text=[f"{w:.1f}%" for w in win_rates],
                    textposition='auto'
                ))
                
                fig_winrate.update_layout(
                    title="Strategy Win Rates",
                    xaxis_title="Strategy",
                    yaxis_title="Win Rate (%)",
                    height=400,
                    xaxis={'tickangle': -45}
                )
                st.plotly_chart(fig_winrate, use_container_width=True)

else:
    st.info("🎯 No strategy performance data yet. Run the bot to see detailed analytics.")
    
    # Show available strategies
    st.subheader("🎯 Available Trading Strategies")
    
    strategy_info_data = []
    for strategy_id, strategy_info in bot.strategies.strategies.items():
        strategy_info_data.append({
            'ID': strategy_id,
            'Strategy Name': strategy_info['name'],
            'Description': strategy_info['desc']
        })
    
    df_strategy_info = pd.DataFrame(strategy_info_data)
    st.dataframe(df_strategy_info, use_container_width=True)

# ==================== ENHANCED ACTIVE POSITIONS ====================

st.subheader("🎯 Enhanced Active Positions")

if bot.positions:
    positions_data = []
    
    for pos_id, position in bot.positions.items():
        # Get current price for P&L calculation
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
            'Action': f"{'🟢 LONG' if position['action'] == 'B' else '🔴 SHORT'}",
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
    
    # Enhanced position styling
    def style_positions(val):
        if 'P&L' in str(val) or '₹' in str(val):
            try:
                if 'P&L' in str(val):
                    num_val = float(str(val).replace('₹', '').replace(',', ''))
                    return 'background-color: #c8e6c9; font-weight: bold' if num_val >= 0 else 'background-color: #ffcdd2; font-weight: bold'
            except:
                return ''
        elif 'LONG' in str(val):
            return 'color: green; font-weight: bold'
        elif 'SHORT' in str(val):
            return 'color: red; font-weight: bold'
        return ''
    
    styled_positions = df_positions.style.applymap(style_positions, subset=['Action', 'P&L'])
    st.dataframe(styled_positions, use_container_width=True)
    
    # Enhanced position management
    pos_mgmt_col1, pos_mgmt_col2 = st.columns(2)
    
    with pos_mgmt_col1:
        if st.button("🛑 Emergency Close All", type="secondary", use_container_width=True):
            for pos_id in list(bot.positions.keys()):
                position = bot.positions[pos_id]
                try:
                    current_price = position['entry_price']
                    if bot.api.is_connected:
                        live_data = bot.api.get_quote(position['symbol'])
                        if live_data and live_data['price'] > 0:
                            current_price = live_data['price']
                    bot.close_position(pos_id, current_price, "EMERGENCY_CLOSE")
                except Exception as e:
                    st.error(f"Error closing {position['symbol']}: {e}")
            
            st.warning("🛑 All positions closed via emergency!")
            st.rerun()
    
    with pos_mgmt_col2:
        # Select individual position to close
        position_options = [f"{pos['Symbol']} ({pos['Strategy']}) - {pos['Action']}" for pos in positions_data]
        selected_position = st.selectbox("Select Position to Close:", [""] + position_options)
        
        if selected_position and st.button("🔒 Close Selected", use_container_width=True):
            for pos_data in positions_data:
                display_name = f"{pos_data['Symbol']} ({pos_data['Strategy']}) - {pos_data['Action']}"
                if display_name == selected_position:
                    pos_id = pos_data['Position ID']
                    position = bot.positions[pos_id]
                    
                    try:
                        current_price = position['entry_price']
                        if bot.api.is_connected:
                            live_data = bot.api.get_quote(position['symbol'])
                            if live_data and live_data['price'] > 0:
                                current_price = live_data['price']
                        
                        bot.close_position(pos_id, current_price, "MANUAL_CLOSE")
                        st.success(f"✅ Closed: {position['symbol']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error closing position: {e}")
                    break

else:
    st.info("📭 No active positions. Start the bot to begin trading.")

# ==================== ENHANCED MARKET OPPORTUNITIES ====================

st.subheader("🔍 Enhanced Market Opportunities")

opp_col1, opp_col2 = st.columns(2)

with opp_col1:
    if st.button("🔍 Scan for Opportunities", use_container_width=True):
        with st.spinner("🔄 Scanning market conditions..."):
            opportunities = bot.scan_for_opportunities()
            st.session_state['latest_opportunities'] = opportunities

with opp_col2:
    auto_scan = st.checkbox("🔄 Auto-scan every 30 seconds", value=False)

# Display opportunities
if 'latest_opportunities' in st.session_state and st.session_state['latest_opportunities']:
    opportunities = st.session_state['latest_opportunities']
    
    opp_data = []
    for opp in opportunities:
        opp_data.append({
            'Symbol': opp['symbol'],
            'Strategy': opp['strategy_name'],
            'Signal': f"{'🟢 LONG' if opp['signal'] > 0 else '🔴 SHORT'}",
            'Confidence': f"{opp['confidence']:.1%}",
            'Price': f"₹{opp['price']:.2f}",
            'Stop %': f"{opp['stop_pct']:.2%}",
            'Reason': opp['reason'],
            'Volume': f"{opp['volume']:,}",
            'Data Source': opp['data_source']
        })
    
    df_opportunities = pd.DataFrame(opp_data)
    st.dataframe(df_opportunities, use_container_width=True)
    
    # Manual execution option
    if st.button("🚀 Execute Top Opportunity", type="primary"):
        if opportunities:
            with st.spinner("Executing top opportunity..."):
                result = bot.execute_entry(opportunities[0])
                if result:
                    st.success(f"✅ Executed: {opportunities[0]['symbol']} - {opportunities[0]['strategy_name']}")
                else:
                    st.error("❌ Failed to execute entry")
                st.rerun()

else:
    st.info("🔍 Click 'Scan for Opportunities' to find trading signals")

# ==================== ENHANCED TRADE HISTORY ====================

st.subheader("📋 Enhanced Trade History & Analytics")

if bot.trades:
    # Trade summary
    total_trades = len(bot.trades)
    recent_trades = bot.trades[-20:]  # Show last 20 trades
    
    st.markdown(f"**📊 Showing last 20 trades (Total: {total_trades})**")
    
    df_trades = pd.DataFrame(recent_trades)
    
    # Enhanced trade styling
    def style_trades(val):
        if isinstance(val, (int, float)):
            return 'background-color: #c8e6c9; font-weight: bold' if val >= 0 else 'background-color: #ffcdd2; font-weight: bold'
        return ''
    
    trade_columns = ['timestamp', 'symbol', 'strategy_name', 'action', 'quantity', 'pnl', 'hold_time', 'exit_reason', 'confidence', 'mode']
    
    if all(col in df_trades.columns for col in trade_columns):
        styled_trades = df_trades[trade_columns].style.applymap(style_trades, subset=['pnl'])
        st.dataframe(styled_trades, use_container_width=True)
    else:
        st.dataframe(df_trades, use_container_width=True)
    
    # Enhanced trade analytics
    if len(bot.trades) > 5:
        st.subheader("📊 Trade Analytics & Visualizations")
        
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
                line=dict(color='#2E86C1', width=3),
                marker=dict(size=6, color='#2E86C1'),
                fill='tonexty' if any(p >= 0 for p in cumulative_pnl) else None,
                fillcolor='rgba(46, 134, 193, 0.1)'
            ))
            
            # Add zero line
            fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig_cumulative.update_layout(
                title="📈 Cumulative P&L Progression",
                xaxis_title="Trade Number",
                yaxis_title="P&L (₹)",
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with viz_col2:
            # P&L distribution histogram
            pnls = [trade['pnl'] for trade in bot.trades]
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=pnls,
                nbinsx=15,
                name="P&L Distribution",
                marker_color='#58D68D',
                opacity=0.8,
                text=[f"Count: {len([p for p in pnls if bin_start <= p < bin_end])}" 
                      for bin_start, bin_end in zip(
                          np.histogram(pnls, bins=15)[1][:-1], 
                          np.histogram(pnls, bins=15)[1][1:]
                      )],
                texttemplate='%{text}',
                textposition='auto'
            ))
            
            # Add average line
            avg_pnl = np.mean(pnls)
            fig_dist.add_vline(x=avg_pnl, line_dash="dash", line_color="red", 
                              annotation_text=f"Avg: ₹{avg_pnl:.2f}")
            
            fig_dist.update_layout(
                title="📊 P&L Distribution",
                xaxis_title="P&L (₹)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Additional analytics
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            # Strategy performance over time
            strategy_performance = {}
            for trade in bot.trades:
                strategy = trade.get('strategy_name', 'Unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': [], 'pnls': []}
                strategy_performance[strategy]['trades'].append(len(strategy_performance[strategy]['trades']) + 1)
                strategy_performance[strategy]['pnls'].append(trade['pnl'])
            
            fig_strategy_perf = go.Figure()
            
            for strategy, data in strategy_performance.items():
                if len(data['pnls']) > 1:
                    cumulative_strategy_pnl = np.cumsum(data['pnls'])
                    fig_strategy_perf.add_trace(go.Scatter(
                        x=data['trades'],
                        y=cumulative_strategy_pnl,
                        mode='lines+markers',
                        name=strategy,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
            
            fig_strategy_perf.update_layout(
                title="📈 Strategy Performance Over Time",
                xaxis_title="Trade Number (per strategy)",
                yaxis_title="Cumulative P&L (₹)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_strategy_perf, use_container_width=True)
        
        with analytics_col2:
            # Hold time vs P&L scatter
            hold_times = [trade['hold_time'] for trade in bot.trades]
            trade_pnls = [trade['pnl'] for trade in bot.trades]
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=hold_times,
                y=trade_pnls,
                mode='markers',
                marker=dict(
                    size=8,
                    color=trade_pnls,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="P&L (₹)")
                ),
                text=[f"{trade['symbol']}<br>{trade['strategy_name']}" for trade in bot.trades],
                hovertemplate='<b>%{text}</b><br>Hold Time: %{x:.1f}s<br>P&L: ₹%{y:.2f}<extra></extra>'
            ))
            
            fig_scatter.update_layout(
                title="⏱️ Hold Time vs P&L Analysis",
                xaxis_title="Hold Time (seconds)",
                yaxis_title="P&L (₹)",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Enhanced export functionality
    st.subheader("💾 Enhanced Export & Reports")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export All Trades", use_container_width=True):
            csv_data = df_trades.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Complete Trade History",
                data=csv_data,
                file_name=f"flyingbuddha_all_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("📈 Export Performance Report", use_container_width=True):
            performance_report = {
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trading_summary': {
                    'total_trades': len(bot.trades),
                    'total_pnl': sum([t['pnl'] for t in bot.trades]),
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio']
                },
                'risk_metrics': {
                    'max_drawdown': bot.current_drawdown,
                    'daily_pnl': bot.daily_pnl,
                    'capital': bot.capital,
                    'risk_per_trade': bot.risk_per_trade
                },
                'strategy_performance': {
                    str(sid): {
                        'name': bot.strategies.strategies[sid]['name'],
                        'trades': stats['trades'],
                        'success_rate': stats['success_rate'],
                        'total_pnl': stats['total_pnl']
                    } for sid, stats in bot.strategies.strategy_stats.items() if stats['trades'] > 0
                },
                'connection_details': {
                    'method': bot.api.connection_method,
                    'client_id': bot.api.client_id,
                    'last_login': bot.api.last_login_time.isoformat() if bot.api.last_login_time else None
                }
            }
            
            report_json = json.dumps(performance_report, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Performance Report",
                data=report_json,
                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with export_col3:
        if st.button("🎯 Export Strategy Analysis", use_container_width=True):
            strategy_analysis = {}
            for strategy_id, stats in bot.strategies.strategy_stats.items():
                if stats['trades'] > 0:
                    strategy_trades = [t for t in bot.trades if t.get('strategy_id') == strategy_id]
                    strategy_analysis[bot.strategies.strategies[strategy_id]['name']] = {
                        'total_trades': stats['trades'],
                        'success_rate': stats['success_rate'],
                        'total_pnl': stats['total_pnl'],
                        'avg_profit': stats['avg_profit'],
                        'avg_loss': stats['avg_loss'],
                        'avg_hold_time': stats['avg_hold_time'],
                        'trade_details': strategy_trades
                    }
            
            analysis_json = json.dumps(strategy_analysis, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Strategy Analysis",
                data=analysis_json,
                file_name=f"strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    st.info("📭 No trades executed yet. Start the bot to see comprehensive trade history and analytics.")

# ==================== ENHANCED LIVE MARKET DATA ====================

st.subheader("📊 Enhanced Live Market Data")

data_source_info = f"🟢 {bot.api.connection_method}" if bot.api.is_connected else "🟠 yfinance (Fallback)"
st.markdown(f"**Data Source:** {data_source_info}")

market_col1, market_col2 = st.columns(2)

with market_col1:
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.session_state['refresh_market'] = True

with market_col2:
    auto_refresh_market = st.checkbox("🔄 Auto-refresh market data", value=False)

# Display market data
if st.session_state.get('refresh_market', False) or auto_refresh_market:
    market_data_cols = st.columns(5)
    
    for i, symbol in enumerate(bot.symbols[:15]):  # Show first 15 symbols
        with market_data_cols[i % 5]:
            try:
                if bot.api.is_connected:
                    live_data = bot.api.get_quote(symbol)
                    if live_data and live_data['price'] > 0:
                        change_color = "🟢" if live_data.get('change', 0) >= 0 else "🔴"
                        change_pct = live_data.get('change_per', 0)
                        st.metric(
                            symbol, 
                            f"₹{live_data['price']:.2f}",
                            delta=f"{change_color} {change_pct:.2f}%",
                            help=f"Volume: {live_data['volume']:,}\nHigh: ₹{live_data.get('high', 0):.2f}\nLow: ₹{live_data.get('low', 0):.2f}"
                        )
                    else:
                        st.metric(symbol, "No data", help="API data unavailable")
                else:
                    # Fallback to yfinance
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period="1d", interval="1m")
                    if len(hist) > 0:
                        current_price = hist.iloc[-1]['Close']
                        prev_price = hist.iloc[-2]['Close'] if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        change_color = "🟢" if change_pct >= 0 else "🔴"
                        st.metric(
                            symbol, 
                            f"₹{current_price:.2f}",
                            delta=f"{change_color} {change_pct:.2f}%",
                            help="Data from yfinance"
                        )
                    else:
                        st.metric(symbol, "Error", help="Data fetch failed")
            except Exception as e:
                st.metric(symbol, "Error", help=f"Error: {str(e)[:50]}")
    
    st.session_state['refresh_market'] = False

# ==================== SYSTEM DIAGNOSTICS & SETTINGS ====================

with st.expander("⚙️ Enhanced System Diagnostics & Advanced Settings", expanded=False):
    st.subheader("🔧 System Status & Diagnostics")
    
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    
    with diag_col1:
        st.markdown("**🔗 API & Connection Status:**")
        st.write(f"gwcmodel Available: {'✅' if GWCMODEL_AVAILABLE else '❌'}")
        st.write(f"gwcmodel Class: {'✅' if GWC_CLASS else '❌'} {GWC_CLASS_NAME or 'None'}")
        st.write(f"Goodwill API: {'✅ Connected' if bot.api.is_connected else '❌ Disconnected'}")
        st.write(f"Connection Method: {bot.api.connection_method or 'None'}")
        st.write(f"Database: {'✅ Active' if hasattr(bot, 'conn') else '❌ Failed'}")
        st.write(f"Session Duration: {(datetime.now() - bot.api.last_login_time).total_seconds() / 60:.1f}m" if bot.api.last_login_time else "N/A")
    
    with diag_col2:
        st.markdown("**📊 Performance Metrics:**")
        st.write(f"Active Strategies: {len([s for s in bot.strategies.strategy_stats.values() if s['trades'] > 0])}/8")
        st.write(f"Cache Size: {len(bot.strategies.analysis_cache)} items")
        st.write(f"Database Records: {len(bot.trades)} trades")
        st.write(f"Memory Usage: {len(str(bot.positions)) + len(str(bot.trades))} bytes")
        st.write(f"Symbols Tracked: {len(bot.symbols)}")
        st.write(f"Peak Capital: ₹{bot.peak_capital:,.2f}")
    
    with diag_col3:
        st.markdown("**🛡️ Risk & Safety Status:**")
        risk_status = "✅ Within Limits" if bot.check_risk_limits() else "⚠️ Breached"
        position_status = "✅ OK" if len(bot.positions) <= bot.max_positions else "⚠️ Exceeded"
        
        st.write(f"Risk Limits: {risk_status}")
        st.write(f"Position Limits: {position_status}")
        st.write(f"Daily P&L: ₹{bot.daily_pnl:.2f}")
        st.write(f"Max Drawdown: ₹{bot.current_drawdown:.2f}")
        st.write(f"Risk per Trade: {bot.risk_per_trade:.1%}")
        st.write(f"Total Risk Exposure: {len(bot.positions) * bot.risk_per_trade:.1%}")
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.markdown("**⚙️ Trading Parameters:**")
        new_daily_limit = st.slider("Daily Loss Limit (%)", 1, 10, int(bot.daily_loss_limit * 100))
        bot.daily_loss_limit = new_daily_limit / 100
        
        new_drawdown_limit = st.slider("Max Drawdown Limit (%)", 5, 20, int(bot.max_drawdown_limit * 100))
        bot.max_drawdown_limit = new_drawdown_limit / 100
        
        strategy_timeout = st.slider("Strategy Cache Timeout (sec)", 10, 120, bot.strategies.cache_timeout)
        bot.strategies.cache_timeout = strategy_timeout
    
    with settings_col2:
        st.markdown("**🔄 System Actions:**")
        
        if st.button("🧹 Clear Strategy Cache"):
            bot.strategies.analysis_cache.clear()
            st.success("✅ Strategy cache cleared")
        
        if st.button("🗃️ Clear Token Cache"):
            token_keys = [k for k in st.session_state.keys() if k.startswith("token_")]
            for key in token_keys:
                del st.session_state[key]
            st.success("✅ Token cache cleared")
        
        if st.button("💾 Save Strategy Performance"):
            try:
                cursor = bot.conn.cursor()
                for strategy_id, stats in bot.strategies.strategy_stats.items():
                    if stats['trades'] > 0:
                        cursor.execute('''
                            INSERT OR REPLACE INTO strategy_performance VALUES 
                            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            strategy_id, bot.strategies.strategies[strategy_id]['name'],
                            stats['trades'], stats['wins'], stats['total_pnl'],
                            stats['success_rate'], stats['avg_profit'], stats['avg_loss'],
                            stats['avg_hold_time'], datetime.now()
                        ))
                bot.conn.commit()
                st.success("✅ Strategy performance saved to database")
            except Exception as e:
                st.error(f"❌ Database save error: {e}")
        
        if st.button("🔄 Reset Daily P&L"):
            bot.daily_pnl = 0.0
            st.success("✅ Daily P&L reset")

# ==================== AUTO-REFRESH LOGIC ====================

# Auto-refresh when bot is running
if bot.is_running:
    bot.run_trading_cycle()
    time.sleep(3)  # 3-second cycle for production
    st.rerun()

# Auto-scan for opportunities
if auto_scan and not bot.is_running:
    current_time = time.time()
    if 'last_auto_scan' not in st.session_state:
        st.session_state['last_auto_scan'] = 0
    
    if current_time - st.session_state['last_auto_scan'] > 30:  # 30 seconds
        with st.spinner("🔄 Auto-scanning opportunities..."):
            opportunities = bot.scan_for_opportunities()
            st.session_state['latest_opportunities'] = opportunities
            st.session_state['last_auto_scan'] = current_time
        st.rerun()

# ==================== ENHANCED FOOTER ====================

st.markdown("---")

st.markdown("""
### 🚀 FlyingBuddha Complete Fixed Production Scalping Bot

**🎯 Complete 8-Strategy System with FIXED Goodwill Integration:**

1. **🔥 Momentum Breakout** - Price breakouts with volume confirmation and trend validation
2. **🎯 Mean Reversion** - RSI oversold/overbought signals with Bollinger Band touches  
3. **📊 Volume Spike** - High volume momentum moves (>1.8x average) with price acceleration
4. **🎪 Bollinger Squeeze** - Low volatility breakouts from consolidation patterns
5. **⚡ RSI Divergence** - RSI extremes at key support/resistance levels
6. **📈 VWAP Touch** - Price reactions near Volume Weighted Average Price
7. **🏗️ Support/Resistance** - Bounces and rejections at identified key levels  
8. **📰 News Momentum** - High volume + volatility events indicating news-driven moves

**🔧 COMPLETE FIXES IMPLEMENTED:**

✅ **Enhanced gwcmodel Detection** - Comprehensive class scanning with multiple initialization methods  
✅ **Proper Request Token Flow** - Complete implementation per official Goodwill API documentation  
✅ **Automatic URL Parser** - Extracts request_token from any redirect URL format  
✅ **Robust Signature Generation** - Correct SHA-256 checksum as per API specifications  
✅ **Multiple Authentication Methods** - gwcmodel + Direct API with intelligent fallbacks  
✅ **Production Error Handling** - Detailed error messages with specific solutions  
✅ **Enhanced Risk Management** - 5% daily loss limit, 10% max drawdown protection  
✅ **Advanced Performance Analytics** - Comprehensive metrics, charts, and strategy tracking  
✅ **Real-time Data Integration** - Live quotes via Goodwill API with yfinance fallback  
✅ **Production Database Logging** - Complete trade history and performance persistence  

**📋 Complete Authentication Flow:**
1. **Generate Login URL** with your API key
2. **Login via Goodwill Website** using your trading credentials  
3. **Copy Complete Redirect URL** from browser after successful login
4. **Automatic Token Extraction** and signature generation
5. **Secure API Authentication** with full session management

**🛡️ Production Risk Management:**
- Dynamic position sizing based on volatility and drawdown
- Advanced trailing stops with profit protection
- Real-time risk monitoring with automatic circuit breakers
- Comprehensive performance tracking across all strategies
- Emergency position management with manual override controls

**🔗 Technical Foundation:**
- **API Documentation:** https://developer.gwcindia.in/api/
- **Request Token Flow:** Implemented exactly as per official specifications
- **gwcmodel Integration:** Enhanced detection and initialization  
- **Error Handling:** All documented API errors mapped to user solutions
- **Fallback Systems:** Multiple layers of redundancy for reliability

**⚙️ Advanced Features:**
- Real-time strategy performance comparison and selection
- Dynamic market condition analysis with 15+ technical indicators  
- Advanced order management with multiple order types
- Comprehensive trade analytics with visual performance tracking
- Export capabilities for trade history and performance reports
- Auto-refresh market data with connection status monitoring

⚠️ **Production Trading Warning:** This bot places real orders in Live Mode. Always test thoroughly in Paper Mode and never risk more than you can afford to lose. The 8-strategy system is designed for experienced traders who understand scalping risks.

🎯 **For Support:** This is a complete production implementation. All major authentication and integration issues have been resolved with this fixed version.
""")

# Enhanced status indicator
if bot.is_running:
    current_time = datetime.now().strftime('%H:%M:%S')
    connection_info = f" | {bot.api.connection_method}" if bot.api.is_connected else " | Offline"
    position_info = f" | {len(bot.positions)}/{bot.max_positions} positions"
    pnl_info = f" | Daily P&L: ₹{bot.daily_pnl:.0f}"
    
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; right: 20px; background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 12px 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); z-index: 1000; font-weight: bold;">
        🔄 <strong>LIVE TRADING ACTIVE</strong><br>
        <small>⏰ {current_time} | 🎯 8 Strategies{connection_info}{position_info}{pnl_info}</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; right: 20px; background: linear-gradient(90deg, #FF9800, #F57C00); color: white; padding: 12px 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); z-index: 1000; font-weight: bold;">
        ⏸️ <strong>BOT STOPPED</strong><br>
        <small>Ready to trade | {len(bot.symbols)} symbols | {'🟢 Connected' if bot.api.is_connected else '🔴 Offline'}</small>
    </div>
    """, unsafe_allow_html=True)

# Final application footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
    ⚡ <strong>FlyingBuddha Production Scalping Bot v2.0 - COMPLETE FIXED VERSION</strong><br>
    🎯 <strong>8 Advanced Strategies | 🔧 Enhanced gwcmodel Integration | 🎫 Proper Request Token Flow | 🛡️ Production Risk Management</strong><br>
    <em>Built with Streamlit | Powered by Goodwill API | Real-time Trading System</em><br>
    <small>⚠️ For educational and trading purposes. Trade responsibly and never risk more than you can afford to lose.</small>
</div>
""", unsafe_allow_html=True)

# ==================== END OF COMPLETE APPLICATION ====================
