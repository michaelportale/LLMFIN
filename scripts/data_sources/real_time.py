"""
Real-time data streaming implementation.
"""
import pandas as pd
import websocket
import json
import threading
import time
from typing import List, Dict, Any, Optional, Callable
import queue
from datetime import datetime
from .base import DataSource


class RealTimeDataStream:
    """
    Base class for real-time data streaming.
    """
    
    def __init__(self, on_data: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the real-time data stream.
        
        Args:
            on_data: Callback function to handle incoming data
        """
        self.on_data = on_data
        self.is_running = False
        self.ws = None
        self.data_queue = queue.Queue()
        self.thread = None
    
    def start(self):
        """
        Start the data stream.
        """
        if self.is_running:
            print("Stream is already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """
        Stop the data stream.
        """
        self.is_running = False
        if self.ws:
            self.ws.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
    
    def _run(self):
        """
        Run the data stream. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _process_data(self, data: Dict[str, Any]):
        """
        Process incoming data.
        
        Args:
            data: Data received from the stream
        """
        self.data_queue.put(data)
        if self.on_data:
            self.on_data(data)


class AlphaVantageRealTime(RealTimeDataStream):
    """
    Real-time data streaming from Alpha Vantage.
    Note: Alpha Vantage doesn't offer true WebSocket streaming, so this simulates
    streaming by making frequent API calls.
    """
    
    def __init__(self, api_key: str, symbols: List[str], interval: int = 60, 
                 on_data: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the Alpha Vantage real-time stream.
        
        Args:
            api_key: Alpha Vantage API key
            symbols: List of symbols to stream
            interval: Polling interval in seconds
            on_data: Callback function to handle incoming data
        """
        super().__init__(on_data)
        self.api_key = api_key
        self.symbols = symbols
        self.interval = interval
        self.base_url = "https://www.alphavantage.co/query"
        self.last_values = {}
    
    def _run(self):
        """
        Run the polling loop to simulate streaming.
        """
        while self.is_running:
            for symbol in self.symbols:
                try:
                    # Use the intraday API with 1min interval for latest data
                    import requests
                    params = {
                        "function": "TIME_SERIES_INTRADAY",
                        "symbol": symbol,
                        "interval": "1min",
                        "apikey": self.api_key,
                        "outputsize": "compact"
                    }
                    
                    response = requests.get(self.base_url, params=params)
                    data = response.json()
                    
                    # Check for errors
                    if "Error Message" in data:
                        print(f"Error for {symbol}: {data['Error Message']}")
                        continue
                    
                    if "Note" in data:
                        print(f"API limit reached: {data['Note']}")
                        time.sleep(60)  # Wait longer if we hit rate limit
                        continue
                    
                    # Extract the latest data point
                    time_series_key = "Time Series (1min)"
                    if time_series_key in data:
                        time_series = data[time_series_key]
                        latest_time = max(time_series.keys())
                        latest_data = time_series[latest_time]
                        
                        # Format the data
                        formatted_data = {
                            "symbol": symbol,
                            "timestamp": latest_time,
                            "open": float(latest_data["1. open"]),
                            "high": float(latest_data["2. high"]),
                            "low": float(latest_data["3. low"]),
                            "close": float(latest_data["4. close"]),
                            "volume": int(latest_data["5. volume"])
                        }
                        
                        # Only process if this is new data
                        if symbol not in self.last_values or self.last_values[symbol] != formatted_data:
                            self.last_values[symbol] = formatted_data
                            self._process_data(formatted_data)
                    
                except Exception as e:
                    print(f"Error fetching real-time data for {symbol}: {str(e)}")
                
                # Small delay between symbols to avoid hitting rate limits
                time.sleep(1)
            
            # Wait until next polling interval
            time.sleep(self.interval)


class WebSocketDataStream(RealTimeDataStream):
    """
    WebSocket-based real-time data streaming.
    """
    
    def __init__(self, url: str, subscription_message: Dict[str, Any], 
                 message_parser: Callable[[str], Dict[str, Any]],
                 on_data: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the WebSocket data stream.
        
        Args:
            url: WebSocket URL
            subscription_message: Message to send to subscribe to the stream
            message_parser: Function to parse incoming messages
            on_data: Callback function to handle incoming data
        """
        super().__init__(on_data)
        self.url = url
        self.subscription_message = subscription_message
        self.message_parser = message_parser
    
    def _run(self):
        """
        Run the WebSocket connection.
        """
        def on_message(ws, message):
            try:
                data = self.message_parser(message)
                self._process_data(data)
            except Exception as e:
                print(f"Error parsing message: {str(e)}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            if self.is_running:
                print("Attempting to reconnect in 5 seconds...")
                time.sleep(5)
                self._connect()
        
        def on_open(ws):
            print("WebSocket connection opened")
            if self.subscription_message:
                ws.send(json.dumps(self.subscription_message))
        
        self._connect(on_message, on_error, on_close, on_open)
    
    def _connect(self, on_message=None, on_error=None, on_close=None, on_open=None):
        """
        Establish WebSocket connection.
        """
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        self.ws.run_forever()


class CryptoRealTimeStream(WebSocketDataStream):
    """
    Real-time cryptocurrency data streaming.
    Uses Binance WebSocket API as an example.
    """
    
    def __init__(self, symbols: List[str], 
                 on_data: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the cryptocurrency real-time stream.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g. ["btcusdt", "ethusdt"])
            on_data: Callback function to handle incoming data
        """
        # Format symbols for Binance WebSocket
        formatted_symbols = [symbol.lower() + "@trade" for symbol in symbols]
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(formatted_symbols)}"
        
        # No subscription message needed for this endpoint
        subscription_message = None
        
        # Define parser for Binance messages
        def parse_binance_message(message: str) -> Dict[str, Any]:
            data = json.loads(message)
            return {
                "symbol": data["s"],
                "timestamp": datetime.fromtimestamp(data["T"] / 1000).isoformat(),
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "trade_id": data["t"],
                "is_buyer_maker": data["m"]
            }
        
        super().__init__(url, subscription_message, parse_binance_message, on_data)


class ForexRealTimeStream(WebSocketDataStream):
    """
    Real-time forex data streaming.
    Uses OANDA v20 API as an example.
    """
    
    def __init__(self, api_key: str, account_id: str, pairs: List[str],
                 on_data: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the forex real-time stream.
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            pairs: List of forex pairs (e.g. ["EUR_USD", "GBP_USD"])
            on_data: Callback function to handle incoming data
        """
        # OANDA streaming URL
        url = f"wss://stream-fxtrade.oanda.com/v3/accounts/{account_id}/pricing/stream"
        
        # Headers for authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Subscription message
        subscription_message = {
            "instruments": ",".join(pairs)
        }
        
        # Define parser for OANDA messages
        def parse_oanda_message(message: str) -> Dict[str, Any]:
            data = json.loads(message)
            if data.get("type") == "PRICE":
                return {
                    "symbol": data["instrument"],
                    "timestamp": data["time"],
                    "bid": float(data["bids"][0]["price"]),
                    "ask": float(data["asks"][0]["price"]),
                    "spread": float(data["asks"][0]["price"]) - float(data["bids"][0]["price"])
                }
            return None
        
        super().__init__(url, subscription_message, parse_oanda_message, on_data)
        
        # Add headers to WebSocket connection
        self.headers = headers
    
    def _connect(self, on_message=None, on_error=None, on_close=None, on_open=None):
        """
        Establish WebSocket connection with headers.
        """
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        self.ws.run_forever()


class DataStreamManager:
    """
    Manager for multiple real-time data streams.
    """
    
    def __init__(self):
        """
        Initialize the data stream manager.
        """
        self.streams = {}
        self.callbacks = []
    
    def add_stream(self, name: str, stream: RealTimeDataStream):
        """
        Add a data stream.
        
        Args:
            name: Name of the stream
            stream: RealTimeDataStream instance
        """
        self.streams[name] = stream
        
        # Set callback to forward data to all registered callbacks
        def on_data(data):
            data["stream"] = name
            for callback in self.callbacks:
                callback(data)
        
        stream.on_data = on_data
    
    def remove_stream(self, name: str):
        """
        Remove a data stream.
        
        Args:
            name: Name of the stream to remove
        """
        if name in self.streams:
            self.streams[name].stop()
            del self.streams[name]
    
    def start_all(self):
        """
        Start all data streams.
        """
        for name, stream in self.streams.items():
            print(f"Starting stream: {name}")
            stream.start()
    
    def stop_all(self):
        """
        Stop all data streams.
        """
        for name, stream in self.streams.items():
            print(f"Stopping stream: {name}")
            stream.stop()
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback to receive data from all streams.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unregister a callback.
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback) 