import gym
import numpy as np
import pandas as pd
from gym import spaces


class PortfolioEnv(gym.Env):
    """
    A simplified environment that:
    - Observes daily price & sentiment
    - Has discrete actions: 0 = HOLD, 1 = BUY, 2 = SELL
    - Tracks a single ticker's position for MVP
    """

    def __init__(self, csv_path, sentiment_func, initial_cash=10000):
        super(PortfolioEnv, self).__init__()

        # Load price data (daily OHLC)
        self.df = pd.read_csv(csv_path)
        if self.df.columns[0] not in ["Date", "date"]:
            self.df.rename(columns={self.df.columns[0]: "Date"}, inplace=True)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.sort_values("Date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.sentiment_func = sentiment_func
        self.initial_cash = initial_cash

        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._max_steps = len(self.df) - 1

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.num_shares = 0
        return self._get_observation()

    def step(self, action):
        price = self._get_current_price()

        if action == 1 and self.cash >= price:  # BUY
            self.num_shares += 1
            self.cash -= price
        elif action == 2 and self.num_shares > 0:  # SELL
            self.num_shares -= 1
            self.cash += price

        self.current_step += 1
        done = self.current_step >= self._max_steps

        new_price = self._get_current_price()
        total_value = self.cash + self.num_shares * new_price
        reward = (total_value - self.initial_cash) / self.initial_cash

        obs = self._get_observation()
        info = {"portfolio_value": total_value}
        return obs, reward, done, info

    def _get_current_price(self):
        return float(self.df.loc[self.current_step, "Close"])

    def _get_observation(self):
        price = self._get_current_price()
        normalized_price = price / 100.0

        date = self.df.loc[self.current_step, "Date"]
        sentiment_score = self.sentiment_func(date, ticker="AAPL")

        obs = np.array([normalized_price, sentiment_score, float(self.num_shares)], dtype=np.float32)
        return obs