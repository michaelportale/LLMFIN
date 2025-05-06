# envs/multi_asset_portfolio.py

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict, Tuple, Union, Optional


class MultiAssetPortfolioEnv(gym.Env):
    """
    A portfolio environment that handles multiple assets:
    - Observes daily OHLCV data and sentiment for multiple assets
    - Has a more complex action space for portfolio allocation
    - Accounts for transaction costs, holds cash, and handles position sizing
    - Supports market, limit, and stop-loss orders
    """

    def __init__(
        self, 
        csv_paths: List[str], 
        sentiment_func,
        initial_cash: float = 10000, 
        transaction_cost: float = 0.001,
        lookback_window: int = 10, 
        position_size: float = 0.2,
        max_allocation: float = 0.5,  # Maximum allocation to any single asset
        support_advanced_orders: bool = True  # Whether to support limit and stop-loss orders
    ):
        super(MultiAssetPortfolioEnv, self).__init__()

        # Store asset information
        self.num_assets = len(csv_paths)
        self.tickers = [os.path.basename(path).split('.')[0] for path in csv_paths]
        self.dataframes = []
        self.aligned_dates = None
        
        # Load price data for all assets and align them to common dates
        self._load_and_align_data(csv_paths)
        
        # Store parameters
        self.sentiment_func = sentiment_func
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.position_size = position_size
        self.max_allocation = max_allocation
        self.support_advanced_orders = support_advanced_orders
        
        # Define action space
        if support_advanced_orders:
            # Extended action space: For each asset:
            # [0=HOLD, 1=MARKET_BUY, 2=MARKET_SELL, 3=LIMIT_BUY, 4=LIMIT_SELL, 5=STOP_LOSS]
            # Plus intensity and price ratio for limit/stop orders
            # Action format: [[action_type, intensity, price_ratio], ...]
            self.action_space = spaces.Box(
                low=np.array([[0, 0, 0.5]] * self.num_assets),  # price_ratio minimum 0.5 (50% of current price)
                high=np.array([[5, 1, 1.5]] * self.num_assets),  # price_ratio maximum 1.5 (150% of current price)
                dtype=np.float32
            )
            # Pending orders structure: list of dicts with order details
            # Each order: {ticker, order_type, price, shares, expiration_step}
            self.pending_orders = []
            # Maximum order expiration (in days)
            self.max_order_expiration = 10
        else:
            # Original action space: [0=HOLD, 1=BUY, 2=SELL] plus intensity
            self.action_space = spaces.Box(
                low=np.array([[0, 0]] * self.num_assets),
                high=np.array([[2, 1]] * self.num_assets),
                dtype=np.float32
            )
        
        # Observation space
        # For each asset: [normalized_price, returns, volatility, RSI, sentiment]
        # Plus portfolio state: [cash_ratio, position_ratios...]
        features_per_asset = 5  # price, returns, volatility, RSI, sentiment
        portfolio_features = 1 + self.num_assets  # cash_ratio + position_ratio for each asset
        
        # Add features for pending orders if using advanced orders
        if support_advanced_orders:
            # For each asset: [has_pending_limit_buy, has_pending_limit_sell, has_pending_stop_loss]
            portfolio_features += 3 * self.num_assets
            
        total_features = (features_per_asset * self.num_assets) + portfolio_features
        
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # Calculate maximum steps based on available data
        self._max_steps = len(self.aligned_dates) - self.lookback_window - 1
        
        # Performance tracking
        self.portfolio_values = []
        self.asset_allocations = []  # Track allocations over time
        self.actions_taken = []
        self.trade_history = []
    
    def _load_and_align_data(self, csv_paths: List[str]):
        """Load data from all CSVs and align them to common dates"""
        all_dfs = []
        
        for path in csv_paths:
            df = pd.read_csv(path)
            
            # Ensure Date column is properly formatted
            if df.columns[0] not in ["Date", "date"]:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            
            # Calculate features for this asset
            self._calculate_features(df)
            
            # Store DataFrame with ticker info for identification
            ticker = os.path.basename(path).split('.')[0]
            df["_ticker"] = ticker
            all_dfs.append(df)
        
        # Find common dates across all assets
        common_dates = set(all_dfs[0]["Date"])
        for df in all_dfs[1:]:
            common_dates &= set(df["Date"])
        
        # Sort dates chronologically
        common_dates = sorted(list(common_dates))
        self.aligned_dates = common_dates
        
        # Filter DataFrames to only include common dates
        for i, df in enumerate(all_dfs):
            filtered_df = df[df["Date"].isin(common_dates)].copy()
            filtered_df.reset_index(drop=True, inplace=True)
            self.dataframes.append(filtered_df)
    
    def _calculate_features(self, df: pd.DataFrame):
        """Calculate technical indicators for a single asset DataFrame"""
        # Price returns (daily)
        df['Returns'] = df['Close'].pct_change().fillna(0)
        
        # 14-day volatility
        df['Volatility'] = df['Returns'].rolling(window=14).std().fillna(0)
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'].fillna(50, inplace=True)
        df['RSI'] = df['RSI'].fillna(50)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = self.lookback_window
        self.cash = self.initial_cash
        
        # Initialize holdings for each asset
        self.shares = {ticker: 0 for ticker in self.tickers}
        
        # Initialize tracking variables
        self.portfolio_values = [self.initial_cash]
        self.asset_allocations = [{ticker: 0.0 for ticker in self.tickers}]
        self.actions_taken = []
        self.trade_history = []
        
        # Reset pending orders if using advanced orders
        if self.support_advanced_orders:
            self.pending_orders = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        """
        Take a step in the environment based on the action
        """
        # Process pending orders first (if using advanced orders)
        if self.support_advanced_orders:
            self._process_pending_orders()
            
            # Parse extended action format
            # Round the first column to get discrete action types
            discrete_actions = np.round(action[:, 0]).astype(int)
            # Second column is intensity (0-1)
            intensities = np.clip(action[:, 1], 0, 1)
            # Third column is price ratio for limit/stop orders (0.5-1.5)
            price_ratios = np.clip(action[:, 2], 0.5, 1.5)
        else:
            # Parse original action format
            discrete_actions = np.round(action[:, 0]).astype(int)
            intensities = np.clip(action[:, 1], 0, 1)
            
        # Store current portfolio value for reward calculation
        current_prices = self._get_current_prices()
        portfolio_value_before = self._calculate_portfolio_value(current_prices)
        
        # Process each asset's action
        for i, ticker in enumerate(self.tickers):
            asset_action = discrete_actions[i]
            intensity = intensities[i]
            
            if asset_action == 0:  # HOLD
                continue
                
            elif asset_action == 1:  # MARKET BUY
                self._execute_buy(ticker, intensity)
                
            elif asset_action == 2:  # MARKET SELL
                self._execute_sell(ticker, intensity)
            
            elif self.support_advanced_orders and asset_action == 3:  # LIMIT BUY
                price_ratio = price_ratios[i]
                self._place_limit_buy(ticker, intensity, price_ratio)
                
            elif self.support_advanced_orders and asset_action == 4:  # LIMIT SELL
                price_ratio = price_ratios[i]
                self._place_limit_sell(ticker, intensity, price_ratio)
                
            elif self.support_advanced_orders and asset_action == 5:  # STOP LOSS
                price_ratio = price_ratios[i]
                self._place_stop_loss(ticker, intensity, price_ratio)
        
        # Store actions
        self.actions_taken.append(discrete_actions)
        
        # Move to next day
        self.current_step += 1
        done = self.current_step >= len(self.aligned_dates) - 1
        
        # Calculate new portfolio value and reward
        new_prices = self._get_current_prices()
        portfolio_value_after = self._calculate_portfolio_value(new_prices)
        
        # Store new portfolio value
        self.portfolio_values.append(portfolio_value_after)
        
        # Store current allocations
        allocations = self._calculate_allocations(new_prices)
        self.asset_allocations.append(allocations)
        
        # Calculate reward as daily return
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before if portfolio_value_before > 0 else 0
        
        # Get new observation
        obs = self._get_observation()
        
        # Provide additional info for analysis
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "allocations": allocations,
            "date": self.aligned_dates[self.current_step].strftime("%Y-%m-%d"),
            "daily_return": reward,
            "pending_orders": len(self.pending_orders) if self.support_advanced_orders else 0
        }
        
        return obs, reward, done, False, info
    
    def _execute_buy(self, ticker: str, intensity: float):
        """Execute a buy order for the specified asset with given intensity"""
        idx = self.tickers.index(ticker)
        current_price = self._get_current_prices()[idx]
        
        # Calculate available cash for this trade based on intensity and position_size
        available_cash = self.cash * self.position_size * intensity
        
        # Don't proceed if we don't have enough cash for even 1 share
        if available_cash < current_price:
            return
        
        # Calculate shares to buy, ensuring we respect max allocation if specified
        max_shares_by_cash = int(available_cash / current_price)
        
        # Check if buying would exceed max allocation
        if self.max_allocation < 1.0:
            current_portfolio_value = self._calculate_portfolio_value(self._get_current_prices())
            current_position_value = self.shares[ticker] * current_price
            max_target_value = current_portfolio_value * self.max_allocation
            
            # How much more we can allocate to this asset
            remaining_allocation = max_target_value - current_position_value
            max_shares_by_allocation = int(remaining_allocation / current_price)
            
            # Take the minimum to respect both constraints
            shares_to_buy = min(max_shares_by_cash, max_shares_by_allocation)
            shares_to_buy = max(0, shares_to_buy)  # Ensure non-negative
        else:
            shares_to_buy = max_shares_by_cash
        
        # Execute trade if we're buying at least 1 share
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price
            transaction_fee = cost * self.transaction_cost
            total_cost = cost + transaction_fee
            
            # Update state
            self.shares[ticker] += shares_to_buy
            self.cash -= total_cost
            
            # Record trade
            self.trade_history.append({
                'date': self.aligned_dates[self.current_step],
                'ticker': ticker,
                'action': 'BUY',
                'price': current_price,
                'shares': shares_to_buy,
                'cost': total_cost,
                'intensity': intensity
            })
    
    def _execute_sell(self, ticker: str, intensity: float):
        """Execute a sell order for the specified asset with given intensity"""
        idx = self.tickers.index(ticker)
        current_price = self._get_current_prices()[idx]
        current_shares = self.shares[ticker]
        
        # Don't proceed if we don't have any shares
        if current_shares <= 0:
            return
        
        # Calculate shares to sell based on intensity
        shares_to_sell = int(current_shares * intensity)
        shares_to_sell = max(1, shares_to_sell)  # Sell at least 1 share if intensity > 0
        shares_to_sell = min(shares_to_sell, current_shares)  # Can't sell more than we have
        
        # Execute trade
        revenue = shares_to_sell * current_price
        transaction_fee = revenue * self.transaction_cost
        net_revenue = revenue - transaction_fee
        
        # Update state
        self.shares[ticker] -= shares_to_sell
        self.cash += net_revenue
        
        # Record trade
        self.trade_history.append({
            'date': self.aligned_dates[self.current_step],
            'ticker': ticker,
            'action': 'SELL',
            'price': current_price,
            'shares': shares_to_sell,
            'revenue': net_revenue,
            'intensity': intensity
        })
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets"""
        prices = []
        for i, df in enumerate(self.dataframes):
            prices.append(float(df.loc[self.current_step, "Close"]))
        return np.array(prices)
    
    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """Calculate total portfolio value given current prices"""
        # Cash plus sum of (shares * price) for each asset
        asset_values = sum(self.shares[ticker] * prices[i] for i, ticker in enumerate(self.tickers))
        return self.cash + asset_values
    
    def _calculate_allocations(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate current allocation percentages for each asset"""
        total_value = self._calculate_portfolio_value(prices)
        allocations = {ticker: 0.0 for ticker in self.tickers}
        
        if total_value > 0:
            # Calculate allocation percentage for each asset
            for i, ticker in enumerate(self.tickers):
                asset_value = self.shares[ticker] * prices[i]
                allocations[ticker] = asset_value / total_value
        
        return allocations
    
    def _process_pending_orders(self):
        """Process all pending limit and stop-loss orders"""
        if not self.support_advanced_orders:
            return
            
        current_prices = self._get_current_prices()
        executed_orders = []
        
        # Process each pending order
        for order in self.pending_orders:
            ticker = order['ticker']
            idx = self.tickers.index(ticker)
            current_price = current_prices[idx]
            
            # Check if order should be executed based on type and price
            executed = False
            
            if order['order_type'] == 'LIMIT_BUY' and current_price <= order['price']:
                # Execute limit buy if current price falls below limit price
                shares_to_buy = order['shares']
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee
                
                # Check if we still have enough cash
                if self.cash >= total_cost:
                    self.shares[ticker] += shares_to_buy
                    self.cash -= total_cost
                    
                    # Record trade
                    self.trade_history.append({
                        'date': self.aligned_dates[self.current_step],
                        'ticker': ticker,
                        'action': 'LIMIT_BUY',
                        'price': current_price,
                        'limit_price': order['price'],
                        'shares': shares_to_buy,
                        'cost': total_cost
                    })
                    
                    executed = True
                
            elif order['order_type'] == 'LIMIT_SELL' and current_price >= order['price']:
                # Execute limit sell if current price rises above limit price
                shares_to_sell = order['shares']
                current_shares = self.shares[ticker]
                
                # Check if we still have shares to sell
                if current_shares >= shares_to_sell:
                    revenue = shares_to_sell * current_price
                    transaction_fee = revenue * self.transaction_cost
                    net_revenue = revenue - transaction_fee
                    
                    self.shares[ticker] -= shares_to_sell
                    self.cash += net_revenue
                    
                    # Record trade
                    self.trade_history.append({
                        'date': self.aligned_dates[self.current_step],
                        'ticker': ticker,
                        'action': 'LIMIT_SELL',
                        'price': current_price,
                        'limit_price': order['price'],
                        'shares': shares_to_sell,
                        'revenue': net_revenue
                    })
                    
                    executed = True
                
            elif order['order_type'] == 'STOP_LOSS' and current_price <= order['price']:
                # Execute stop-loss if current price falls below stop price
                shares_to_sell = order['shares']
                current_shares = self.shares[ticker]
                
                # Check if we still have shares to sell
                if current_shares >= shares_to_sell:
                    revenue = shares_to_sell * current_price
                    transaction_fee = revenue * self.transaction_cost
                    net_revenue = revenue - transaction_fee
                    
                    self.shares[ticker] -= shares_to_sell
                    self.cash += net_revenue
                    
                    # Record trade
                    self.trade_history.append({
                        'date': self.aligned_dates[self.current_step],
                        'ticker': ticker,
                        'action': 'STOP_LOSS',
                        'price': current_price,
                        'stop_price': order['price'],
                        'shares': shares_to_sell,
                        'revenue': net_revenue
                    })
                    
                    executed = True
            
            # Remove executed orders or expired orders
            if executed or self.current_step >= order['expiration_step']:
                executed_orders.append(order)
                
                # Record canceled order
                if not executed and self.current_step >= order['expiration_step']:
                    self.trade_history.append({
                        'date': self.aligned_dates[self.current_step],
                        'ticker': ticker,
                        'action': f"CANCEL_{order['order_type']}",
                        'reason': 'EXPIRED'
                    })
        
        # Remove executed/expired orders
        for order in executed_orders:
            self.pending_orders.remove(order)
    
    def _place_limit_buy(self, ticker: str, intensity: float, price_ratio: float):
        """Place a limit buy order"""
        if not self.support_advanced_orders or intensity <= 0:
            return
            
        idx = self.tickers.index(ticker)
        current_price = self._get_current_prices()[idx]
        
        # Set limit price below current price (price_ratio < 1) for buys
        limit_price = current_price * price_ratio
        
        # Calculate available cash for this trade
        available_cash = self.cash * self.position_size * intensity
        
        # Don't proceed if we don't have enough cash for even 1 share
        if available_cash < limit_price:
            return
            
        # Calculate shares to buy, respecting max allocation
        shares_to_buy = int(available_cash / limit_price)
        
        # Check if buying would exceed max allocation
        if self.max_allocation < 1.0:
            current_portfolio_value = self._calculate_portfolio_value(self._get_current_prices())
            current_position_value = self.shares[ticker] * current_price
            max_target_value = current_portfolio_value * self.max_allocation
            
            # How much more we can allocate to this asset
            remaining_allocation = max_target_value - current_position_value
            max_shares_by_allocation = int(remaining_allocation / limit_price)
            
            # Take the minimum to respect both constraints
            shares_to_buy = min(shares_to_buy, max_shares_by_allocation)
            shares_to_buy = max(0, shares_to_buy)  # Ensure non-negative
            
        # Add order to pending list
        if shares_to_buy > 0:
            # Set aside cash for this order
            reserved_cash = shares_to_buy * limit_price * (1 + self.transaction_cost)
            self.cash -= reserved_cash
            
            self.pending_orders.append({
                'ticker': ticker,
                'order_type': 'LIMIT_BUY',
                'price': limit_price,
                'shares': shares_to_buy,
                'reserved_cash': reserved_cash,
                'placement_step': self.current_step,
                'expiration_step': self.current_step + self.max_order_expiration
            })
            
            # Record order placement
            self.trade_history.append({
                'date': self.aligned_dates[self.current_step],
                'ticker': ticker,
                'action': 'PLACE_LIMIT_BUY',
                'current_price': current_price,
                'limit_price': limit_price,
                'shares': shares_to_buy,
                'reserved_cash': reserved_cash
            })
    
    def _place_limit_sell(self, ticker: str, intensity: float, price_ratio: float):
        """Place a limit sell order"""
        if not self.support_advanced_orders or intensity <= 0:
            return
            
        idx = self.tickers.index(ticker)
        current_price = self._get_current_prices()[idx]
        current_shares = self.shares[ticker]
        
        # Don't proceed if we don't have any shares
        if current_shares <= 0:
            return
            
        # Set limit price above current price (price_ratio > 1) for sells
        limit_price = current_price * price_ratio
        
        # Calculate shares to sell based on intensity
        shares_to_sell = int(current_shares * intensity)
        shares_to_sell = max(1, shares_to_sell)  # Sell at least 1 share if intensity > 0
        shares_to_sell = min(shares_to_sell, current_shares)  # Can't sell more than we have
        
        # Reduce available shares
        self.shares[ticker] -= shares_to_sell
        
        # Add order to pending list
        self.pending_orders.append({
            'ticker': ticker,
            'order_type': 'LIMIT_SELL',
            'price': limit_price,
            'shares': shares_to_sell,
            'placement_step': self.current_step,
            'expiration_step': self.current_step + self.max_order_expiration
        })
        
        # Record order placement
        self.trade_history.append({
            'date': self.aligned_dates[self.current_step],
            'ticker': ticker,
            'action': 'PLACE_LIMIT_SELL',
            'current_price': current_price,
            'limit_price': limit_price,
            'shares': shares_to_sell
        })
    
    def _place_stop_loss(self, ticker: str, intensity: float, price_ratio: float):
        """Place a stop-loss order"""
        if not self.support_advanced_orders or intensity <= 0:
            return
            
        idx = self.tickers.index(ticker)
        current_price = self._get_current_prices()[idx]
        current_shares = self.shares[ticker]
        
        # Don't proceed if we don't have any shares
        if current_shares <= 0:
            return
            
        # Set stop price below current price (price_ratio < 1) for stop losses
        stop_price = current_price * price_ratio
        
        # Calculate shares to sell based on intensity
        shares_to_sell = int(current_shares * intensity)
        shares_to_sell = max(1, shares_to_sell)  # Sell at least 1 share if intensity > 0
        shares_to_sell = min(shares_to_sell, current_shares)  # Can't sell more than we have
        
        # Reduce available shares
        self.shares[ticker] -= shares_to_sell
        
        # Add order to pending list
        self.pending_orders.append({
            'ticker': ticker,
            'order_type': 'STOP_LOSS',
            'price': stop_price,
            'shares': shares_to_sell,
            'placement_step': self.current_step,
            'expiration_step': self.current_step + self.max_order_expiration
        })
        
        # Record order placement
        self.trade_history.append({
            'date': self.aligned_dates[self.current_step],
            'ticker': ticker,
            'action': 'PLACE_STOP_LOSS',
            'current_price': current_price,
            'stop_price': stop_price,
            'shares': shares_to_sell
        })
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation state"""
        observation = []
        prices = self._get_current_prices()
        
        # Add features for each asset
        for i, ticker in enumerate(self.tickers):
            df = self.dataframes[i]
            
            # Price (normalized)
            price = prices[i]
            normalized_price = price / 100.0  # Simple normalization
            
            # Technical indicators
            returns = float(df.loc[self.current_step, "Returns"])
            volatility = float(df.loc[self.current_step, "Volatility"])
            rsi = float(df.loc[self.current_step, "RSI"]) / 100.0  # Normalize to 0-1
            
            # Sentiment
            date = self.aligned_dates[self.current_step]
            sentiment = self.sentiment_func(date, ticker=ticker)
            
            # Add to observation
            observation.extend([normalized_price, returns, volatility, rsi, sentiment])
        
        # Add portfolio state
        total_value = self._calculate_portfolio_value(prices)
        cash_ratio = self.cash / total_value if total_value > 0 else 1.0
        observation.append(cash_ratio)
        
        # Add position ratios for each asset
        for i, ticker in enumerate(self.tickers):
            position_value = self.shares[ticker] * prices[i]
            position_ratio = position_value / total_value if total_value > 0 else 0.0
            observation.append(position_ratio)
        
        # Add pending order information if using advanced orders
        if self.support_advanced_orders:
            for ticker in self.tickers:
                # Check if there are pending orders for this ticker
                has_limit_buy = 0.0
                has_limit_sell = 0.0
                has_stop_loss = 0.0
                
                for order in self.pending_orders:
                    if order['ticker'] == ticker:
                        if order['order_type'] == 'LIMIT_BUY':
                            has_limit_buy = 1.0
                        elif order['order_type'] == 'LIMIT_SELL':
                            has_limit_sell = 1.0
                        elif order['order_type'] == 'STOP_LOSS':
                            has_stop_loss = 1.0
                
                observation.extend([has_limit_buy, has_limit_sell, has_stop_loss])
        
        # Convert to numpy array and ensure values are in bounds
        obs = np.array(observation, dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the portfolio strategy"""
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
        
        # Calculate Sharpe ratio
        daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Annualize assuming 252 trading days
        
        # Calculate per-asset metrics
        asset_metrics = {}
        for i, ticker in enumerate(self.tickers):
            # Count trades per asset
            asset_trades = [t for t in self.trade_history if t['ticker'] == ticker]
            asset_buy_trades = [t for t in asset_trades if t['action'] == 'BUY']
            asset_sell_trades = [t for t in asset_trades if t['action'] == 'SELL']
            
            asset_metrics[ticker] = {
                "num_trades": len(asset_trades),
                "num_buys": len(asset_buy_trades),
                "num_sells": len(asset_sell_trades),
                "final_allocation": self.asset_allocations[-1][ticker] if self.asset_allocations else 0.0
            }
        
        return {
            "initial_value": float(initial_value),
            "final_value": float(final_value),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(annualized_sharpe),
            "num_trades": len(self.trade_history),
            "asset_metrics": asset_metrics
        } 