# FinPort - AI Portfolio Trainer

FinPort is an advanced financial portfolio optimization platform that leverages reinforcement learning to train models for stock trading and portfolio management.

## Features

- 🤖 **Reinforcement Learning Models**: Train models using algorithms like PPO, A2C, SAC and TD3
- 📈 **Portfolio Optimization**: Optimize stock portfolios for maximum returns
- 📊 **Interactive Dashboard**: Visualize model performance and portfolio metrics
- 📉 **Historical Data Analysis**: Train on historical stock data with technical indicators
- 🔄 **Real-time Training**: Monitor training progress with real-time visualizations
- 📝 **Report Generation**: Create detailed performance reports for your models
- 🔍 **Model Comparison**: Compare different models and strategies

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finport.git
   cd finport
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface at http://localhost:5000

### Docker Deployment

For easier deployment, you can use Docker:

```bash
docker-compose up -d
```

## Usage

### Training Models

1. Select a ticker symbol from the dropdown
2. Choose training parameters (steps, algorithm, etc.)
3. Click "Start Training" to begin the training process
4. Monitor progress and view results in real-time

### Managing Models

The platform allows you to:
- View all trained models
- Compare model performance
- Backtest models on different datasets
- Export models for external use

### Data Management

- Automatic data fetching for new stocks
- Data quality checks
- Support for multiple data sources

## Architecture

The application consists of:
- **Flask Backend**: Serves the API and web interface
- **Stable-Baselines3**: Powers the reinforcement learning algorithms
- **Chart.js**: Provides interactive visualizations
- **Bootstrap**: Powers the responsive UI

## Advanced Features

### Multi-Asset Portfolios

Train models that manage multiple assets simultaneously:

```python
from envs.multi_asset_portfolio import MultiAssetPortfolioEnv

# Create a multi-asset environment
env = MultiAssetPortfolioEnv(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2022-01-01"
)
```

### Custom Metrics

Create and use custom performance metrics:

```python
from scripts.custom_metrics import calculate_sortino_ratio

# Calculate the Sortino ratio for a model
sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
