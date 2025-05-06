# envs/portfolio.py

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime
import os


class PortfolioEnv(gym.Env):
    """
    A more realistic portfolio environment that:
    - Observes daily OHLCV data and sentiment
    - Has discrete actions: 0 = HOLD, 1 = BUY, 2 = SELL
    - Accounts for transaction costs, holds cash, and handles position sizing
    """

    def __init__(self, csv_path, sentiment_func, initial_cash=10000, transaction_cost=0.001, 
                 lookback_window=10, position_size=0.1):
        super(PortfolioEnv, self).__init__()

        # Load price data (daily OHLC)

        self.ticker = os.path.basename(csv_path).split('.')[0]

        self.df = pd.read_csv(csv_path)

        if self.df.columns[0] not in ["Date", "date"]:
            self.df.rename(columns={self.df.columns[0]: "Date"}, inplace=True)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.sort_values("Date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Calculate additional features (returns, volatility)
        self._calculate_features()

        self.sentiment_func = sentiment_func
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.position_size = position_size  # Percentage of cash to use per trade

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # HOLD, BUY, SELL
        
        # Observation space includes price data, technical indicators, and portfolio state
        # [normalized price, returns, volatility, RSI, sentiment, cash_ratio, position_size]
        self.observation_space = gym.spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(7,), 
            dtype=np.float32
        )

        self._max_steps = len(self.df) - self.lookback_window - 1
        
        # For tracking performance
        self.portfolio_values = []
        self.actions_taken = []
        self.trade_history = []
        
    def _calculate_features(self):
        """Calculate technical indicators and features from price data"""
        # Price returns (daily)
        self.df['Returns'] = self.df['Close'].pct_change().fillna(0)
        
        # 14-day volatility
        self.df['Volatility'] = self.df['Returns'].rolling(window=14).std().fillna(0)
        
        # Simple RSI calculation
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'].fillna(50, inplace=True)  # Fill initial NaN values
        self.df['RSI'] = self.df['RSI'].fillna(50)  # Default RSI value for first 14 days

    def reset(self, seed=None, options=None):
        #Seed the random number generator
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_cash
        self.num_shares = 0
        self.portfolio_values = [self.initial_cash]
        self.actions_taken = []
        self.trade_history = []
        return self._get_observation(), {}

    def step(self, action):
        # Get current price data
        current_price = self._get_current_price()
        
        # Store pre-action portfolio value for reward calculation
        pre_action_value = self.cash + self.num_shares * current_price
        
        # Execute trade based on action
        order_size = int(self.cash * self.position_size / current_price) if action == 1 else self.num_shares
        
        if action == 1 and self.cash >= current_price:  # BUY
            shares_to_buy = max(1, order_size)  # Buy at least 1 share
            cost = shares_to_buy * current_price
            transaction_fee = cost * self.transaction_cost
            
            if self.cash >= (cost + transaction_fee):
                self.num_shares += shares_to_buy
                self.cash -= (cost + transaction_fee)
                self.trade_history.append({
                    'date': self.df.loc[self.current_step, 'Date'],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'cost': cost + transaction_fee
                })
            
        elif action == 2 and self.num_shares > 0:  # SELL
            shares_to_sell = max(1, min(order_size, self.num_shares))
            revenue = shares_to_sell * current_price
            transaction_fee = revenue * self.transaction_cost
            
            self.num_shares -= shares_to_sell
            self.cash += (revenue - transaction_fee)
            self.trade_history.append({
                'date': self.df.loc[self.current_step, 'Date'],
                'action': 'SELL',
                'price': current_price,
                'shares': shares_to_sell,
                'revenue': revenue - transaction_fee
            })
        
        # Store the action taken
        self.actions_taken.append(action)
        
        # Move to next day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate new portfolio value and reward
        new_price = self._get_current_price()
        total_value = self.cash + self.num_shares * new_price
        self.portfolio_values.append(total_value)
        
        # Calculate reward as daily return
        reward = (total_value - pre_action_value) / pre_action_value if pre_action_value > 0 else 0
        
        # Get new observation
        obs = self._get_observation()
        
        # Provide additional info for analysis
        info = {
            "portfolio_value": total_value,
            "cash": self.cash,
            "shares": self.num_shares,
            "date": self.df.loc[self.current_step, "Date"].strftime("%Y-%m-%d"),
            "daily_return": reward
        }
        
        return obs, reward, done, False, info

    def _get_current_price(self):
        return float(self.df.loc[self.current_step, "Close"])

    def _get_observation(self):
        # Price features
        price = self._get_current_price()
        normalized_price = price / 100.0  # Simple normalization
        
        # Get technical features
        returns = float(self.df.loc[self.current_step, "Returns"])
        volatility = float(self.df.loc[self.current_step, "Volatility"])
        rsi = float(self.df.loc[self.current_step, "RSI"]) / 100.0  # Normalize RSI to 0-1
        
        # Get sentiment for the current date
        date = self.df.loc[self.current_step, "Date"]
        ticker = self.ticker
        sentiment_score = self.sentiment_func(date, ticker=ticker)
        
        # Portfolio state features
        total_value = self.cash + self.num_shares * price
        cash_ratio = self.cash / total_value if total_value > 0 else 0
        position_ratio = (self.num_shares * price) / total_value if total_value > 0 else 0
        
        # Construct observation vector
        obs = np.array([
            normalized_price, 
            returns,
            volatility,
            rsi,
            sentiment_score,
            cash_ratio,
            position_ratio
        ], dtype=np.float32)
        
        obs = np.clip(obs, -10.0, 10.0)  # Ensure obs is within bounds
        return obs
        
    def get_performance_metrics(self):
        """Calculate performance metrics for the strategy"""
        if not self.portfolio_values:
            return {}
        
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        
        # Calculate returns
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate max drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Annualize assuming 252 trading days
        
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": annualized_sharpe,
            "num_trades": len(self.trade_history)
        }