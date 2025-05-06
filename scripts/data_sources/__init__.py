"""
Data sources package for fetching financial data from various providers.
"""
from .base import DataSource
from .yahoo_finance import YahooFinanceDataSource
from .alpha_vantage import AlphaVantageDataSource
from .quandl import QuandlDataSource
from .crypto import CryptoDataSource
from .forex import ForexDataSource
from .real_time import (
    RealTimeDataStream,
    AlphaVantageRealTime,
    WebSocketDataStream,
    CryptoRealTimeStream,
    ForexRealTimeStream,
    DataStreamManager
)

__all__ = [
    'DataSource',
    'YahooFinanceDataSource',
    'AlphaVantageDataSource',
    'QuandlDataSource',
    'CryptoDataSource',
    'ForexDataSource',
    'RealTimeDataStream',
    'AlphaVantageRealTime',
    'WebSocketDataStream',
    'CryptoRealTimeStream',
    'ForexRealTimeStream',
    'DataStreamManager'
] 