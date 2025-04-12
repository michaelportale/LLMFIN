import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.portfolio import PortfolioEnv
from scripts.dummy_sentiment import get_dummy_sentiment


def train_model(csv_path, total_timesteps=10000, model_save_path="models/ppo_portfolio_mvp"):
    def make_env():
        return PortfolioEnv(
            csv_path=csv_path,
            sentiment_func=get_dummy_sentiment,
            initial_cash=10000
        )

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    csv_path = os.path.join("data", "AAPL.csv")  # Change ticker if needed
    train_model(csv_path)
