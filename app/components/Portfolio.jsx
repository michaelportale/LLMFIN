import React, { useState, useEffect } from 'react';
import { Box, Button, Card, CardContent, Typography, Grid, TextField, FormControl, 
  InputLabel, Select, MenuItem, Chip, Paper, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, CircularProgress, Autocomplete, Slider, 
  Divider, Dialog, DialogTitle, DialogContent, DialogActions, Alert, LinearProgress, 
  FormControlLabel, Switch } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { CategoryScale, Chart, registerables } from 'chart.js';
import axios from 'axios';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

Chart.register(CategoryScale);
Chart.register(...registerables);

const Portfolio = () => {
  // State for portfolio creation
  const [tickers, setTickers] = useState([]);
  const [availableTickers, setAvailableTickers] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState('');
  const [algorithm, setAlgorithm] = useState('PPO');
  const [positionSize, setPositionSize] = useState(0.2);
  const [maxAllocation, setMaxAllocation] = useState(0.4);
  const [timesteps, setTimesteps] = useState(50000);
  const [isTraining, setIsTraining] = useState(false);
  const [currentPortfolioId, setCurrentPortfolioId] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  
  // State for portfolio listing
  const [portfolios, setPortfolios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [portfolioDetails, setPortfolioDetails] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [runningBacktest, setRunningBacktest] = useState(false);
  
  // State for dialog
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [backtestOpen, setBacktestOpen] = useState(false);
  
  // Advanced options
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [earlyStop, setEarlyStop] = useState(true);
  const [patience, setPatience] = useState(5);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [batchSize, setBatchSize] = useState(64);
  const [nSteps, setNSteps] = useState(2048);
  const [nEpochs, setNEpochs] = useState(10);
  const [gamma, setGamma] = useState(0.99);
  
  // Advanced order options
  const [enableAdvancedOrders, setEnableAdvancedOrders] = useState(true);
  const [maxOrderExpiration, setMaxOrderExpiration] = useState(10);
  
  // Fetch available tickers on component mount
  useEffect(() => {
    const fetchAvailableTickers = async () => {
      try {
        const response = await axios.get('/api/available_tickers');
        setAvailableTickers(response.data.tickers || []);
      } catch (error) {
        console.error('Error fetching available tickers:', error);
      }
    };
    
    fetchAvailableTickers();
  }, []);
  
  // Fetch available portfolios
  useEffect(() => {
    const fetchPortfolios = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/available_portfolios');
        setPortfolios(response.data || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching portfolios:', error);
        setLoading(false);
      }
    };
    
    fetchPortfolios();
    
    // Set up interval to refresh portfolios list
    const interval = setInterval(fetchPortfolios, 30000);
    return () => clearInterval(interval);
  }, []);
  
  // Check training status when portfolio is being trained
  useEffect(() => {
    let interval = null;
    
    if (isTraining && currentPortfolioId) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`/api/portfolio_status/${currentPortfolioId}`);
          setTrainingStatus(response.data);
          
          if (response.data.status === "completed") {
            setIsTraining(false);
            // Refresh portfolios list
            const portfoliosResponse = await axios.get('/api/available_portfolios');
            setPortfolios(portfoliosResponse.data || []);
          } else if (response.data.status === "error") {
            setIsTraining(false);
            setTrainingError(response.data.error || "Unknown error during training");
          }
        } catch (error) {
          console.error('Error checking training status:', error);
        }
      }, 5000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isTraining, currentPortfolioId]);

  // Handle adding ticker to portfolio
  const handleAddTicker = () => {
    if (selectedTicker && !tickers.includes(selectedTicker)) {
      setTickers([...tickers, selectedTicker]);
      setSelectedTicker('');
    }
  };
  
  // Handle removing ticker from portfolio
  const handleRemoveTicker = (tickerToRemove) => {
    setTickers(tickers.filter(ticker => ticker !== tickerToRemove));
  };
  
  // Handle creating a new portfolio
  const handleCreatePortfolio = async () => {
    if (tickers.length < 2) {
      setTrainingError('Portfolio requires at least 2 tickers');
      return;
    }
    
    try {
      setIsTraining(true);
      setTrainingError(null);
      
      const requestData = {
        tickers,
        algorithm,
        position_size: positionSize,
        max_allocation: maxAllocation,
        timesteps,
        early_stopping: earlyStop,
        patience,
        learning_rate: learningRate,
        batch_size: batchSize,
        n_steps: nSteps,
        n_epochs: nEpochs,
        gamma,
        enable_advanced_orders: enableAdvancedOrders,
        max_order_expiration: maxOrderExpiration
      };
      
      const response = await axios.post('/api/create_portfolio', requestData);
      
      if (response.data.status === 'processing') {
        setCurrentPortfolioId(response.data.portfolio_id);
      } else {
        setIsTraining(false);
        setTrainingError('Failed to start training');
      }
    } catch (error) {
      setIsTraining(false);
      setTrainingError(error.response?.data?.error || 'Error creating portfolio');
      console.error('Error creating portfolio:', error);
    }
  };
  
  // Handle viewing portfolio details
  const handleViewDetails = async (portfolioId) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/portfolio/${portfolioId}`);
      setPortfolioDetails(response.data);
      setDetailsOpen(true);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching portfolio details:', error);
      setLoading(false);
    }
  };
  
  // Handle running backtest
  const handleRunBacktest = async (portfolioId) => {
    try {
      setRunningBacktest(true);
      const response = await axios.post(`/api/portfolio_backtest/${portfolioId}`);
      setBacktestResults(response.data);
      setBacktestOpen(true);
      setRunningBacktest(false);
    } catch (error) {
      setRunningBacktest(false);
      console.error('Error running backtest:', error);
    }
  };
  
  // Format number for display
  const formatNumber = (num, decimals = 2) => {
    return Number(num).toFixed(decimals);
  };
  
  // Format percentage for display
  const formatPercentage = (num) => {
    return `${(Number(num) * 100).toFixed(2)}%`;
  };
  
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Multi-Asset Portfolio Management
      </Typography>
      
      <Grid container spacing={3}>
        {/* Portfolio Creation Form */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Create New Portfolio
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Autocomplete
                  options={availableTickers.filter(ticker => !tickers.includes(ticker))}
                  value={selectedTicker}
                  onChange={(event, newValue) => setSelectedTicker(newValue)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Select Ticker"
                      variant="outlined"
                      fullWidth
                    />
                  )}
                />
                <Button 
                  variant="contained" 
                  onClick={handleAddTicker}
                  disabled={!selectedTicker || isTraining}
                  sx={{ mt: 1 }}
                >
                  Add Ticker
                </Button>
              </Box>
              
              <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {tickers.map((ticker) => (
                  <Chip
                    key={ticker}
                    label={ticker}
                    onDelete={() => handleRemoveTicker(ticker)}
                    disabled={isTraining}
                  />
                ))}
              </Box>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Algorithm</InputLabel>
                <Select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  label="Algorithm"
                  disabled={isTraining}
                >
                  <MenuItem value="PPO">PPO</MenuItem>
                  <MenuItem value="A2C">A2C</MenuItem>
                  <MenuItem value="SAC">SAC</MenuItem>
                  <MenuItem value="TD3">TD3</MenuItem>
                </Select>
              </FormControl>
              
              <Box sx={{ mb: 2 }}>
                <Typography gutterBottom>Position Size: {formatPercentage(positionSize)}</Typography>
                <Slider
                  value={positionSize}
                  onChange={(e, newValue) => setPositionSize(newValue)}
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  disabled={isTraining}
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography gutterBottom>Max Allocation: {formatPercentage(maxAllocation)}</Typography>
                <Slider
                  value={maxAllocation}
                  onChange={(e, newValue) => setMaxAllocation(newValue)}
                  min={0.1}
                  max={1}
                  step={0.1}
                  disabled={isTraining}
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography gutterBottom>Training Steps: {timesteps.toLocaleString()}</Typography>
                <Slider
                  value={timesteps}
                  onChange={(e, newValue) => setTimesteps(newValue)}
                  min={10000}
                  max={200000}
                  step={10000}
                  disabled={isTraining}
                />
              </Box>
              
              <Button 
                variant="outlined" 
                onClick={() => setShowAdvanced(!showAdvanced)}
                sx={{ mb: 2 }}
                disabled={isTraining}
              >
                {showAdvanced ? 'Hide' : 'Show'} Advanced Options
              </Button>
              
              {showAdvanced && (
                <Box sx={{ mb: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Early Stopping</InputLabel>
                        <Select
                          value={earlyStop}
                          onChange={(e) => setEarlyStop(e.target.value)}
                          label="Early Stopping"
                          disabled={isTraining}
                        >
                          <MenuItem value={true}>Enabled</MenuItem>
                          <MenuItem value={false}>Disabled</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Patience"
                        type="number"
                        value={patience}
                        onChange={(e) => setPatience(parseInt(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Learning Rate"
                        type="number"
                        inputProps={{ step: 0.0001, min: 0.0001, max: 0.01 }}
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Batch Size"
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(parseInt(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="N Steps"
                        type="number"
                        value={nSteps}
                        onChange={(e) => setNSteps(parseInt(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="N Epochs"
                        type="number"
                        value={nEpochs}
                        onChange={(e) => setNEpochs(parseInt(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Gamma"
                        type="number"
                        inputProps={{ step: 0.01, min: 0, max: 1 }}
                        value={gamma}
                        onChange={(e) => setGamma(parseFloat(e.target.value))}
                        fullWidth
                        disabled={isTraining}
                      />
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Advanced Order Types</Typography>
                      <FormControlLabel
                        control={
                          <Switch 
                            checked={enableAdvancedOrders} 
                            onChange={(e) => setEnableAdvancedOrders(e.target.checked)} 
                          />
                        }
                        label="Enable Limit & Stop-Loss Orders"
                      />
                    </Grid>
                    
                    {enableAdvancedOrders && (
                      <Grid item xs={12} sm={6}>
                        <TextField
                          fullWidth
                          label="Max Order Expiration (days)"
                          type="number"
                          value={maxOrderExpiration}
                          onChange={(e) => setMaxOrderExpiration(parseInt(e.target.value))}
                          InputProps={{ inputProps: { min: 1, max: 30, step: 1 } }}
                          helperText="Days until limit/stop orders expire"
                        />
                      </Grid>
                    )}
                  </Grid>
                </Box>
              )}
              
              {trainingError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {trainingError}
                </Alert>
              )}
              
              <Button
                variant="contained"
                color="primary"
                onClick={handleCreatePortfolio}
                disabled={isTraining || tickers.length < 2}
                fullWidth
              >
                {isTraining ? (
                  <>
                    <CircularProgress size={24} sx={{ mr: 1 }} />
                    Training Portfolio...
                  </>
                ) : (
                  'Create Portfolio'
                )}
              </Button>
              
              {isTraining && trainingStatus && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Status: {trainingStatus.status}
                  </Typography>
                  {trainingStatus.status === 'started' && (
                    <Typography variant="body2">
                      Time elapsed: {Math.floor((Date.now() - trainingStatus.start_time * 1000) / 60000)} minutes
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Portfolio Listing */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Portfolios
              </Typography>
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : portfolios.length === 0 ? (
                <Alert severity="info">
                  No portfolios available. Create one to get started.
                </Alert>
              ) : (
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Assets</TableCell>
                        <TableCell>Algorithm</TableCell>
                        <TableCell>Return</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {portfolios.map((portfolio) => (
                        <TableRow key={portfolio.id}>
                          <TableCell>{portfolio.name}</TableCell>
                          <TableCell>{portfolio.num_assets}</TableCell>
                          <TableCell>{portfolio.algorithm}</TableCell>
                          <TableCell>{formatPercentage(portfolio.total_return)}</TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              onClick={() => handleViewDetails(portfolio.id)}
                              disabled={loading || runningBacktest}
                            >
                              Details
                            </Button>
                            <Button
                              size="small"
                              onClick={() => handleRunBacktest(portfolio.id)}
                              disabled={loading || runningBacktest}
                            >
                              {runningBacktest ? <CircularProgress size={16} /> : 'Backtest'}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Portfolio Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Portfolio Details</DialogTitle>
        <DialogContent>
          {portfolioDetails && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {portfolioDetails.portfolio?.portfolio_id || 'Portfolio'}
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Performance Metrics</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Algorithm</TableCell>
                          <TableCell>{portfolioDetails.algorithm}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Total Return</TableCell>
                          <TableCell>{formatPercentage(portfolioDetails.model?.total_return || 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Benchmark Return</TableCell>
                          <TableCell>{formatPercentage(portfolioDetails.benchmark?.total_return || 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Outperformance</TableCell>
                          <TableCell>{formatPercentage(portfolioDetails.outperformance || 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell>{formatNumber(portfolioDetails.model?.sharpe_ratio || 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Max Drawdown</TableCell>
                          <TableCell>{formatPercentage(portfolioDetails.model?.max_drawdown || 0)}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Assets</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Ticker</TableCell>
                          <TableCell>Avg. Allocation</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {portfolioDetails.portfolio?.tickers?.map((ticker, index) => (
                          <TableRow key={ticker}>
                            <TableCell>{ticker}</TableCell>
                            <TableCell>
                              {portfolioDetails.allocations && 
                               formatPercentage(portfolioDetails.allocations[ticker] || 0)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Performance Chart</Typography>
                  {portfolioDetails.plots?.performance && (
                    <Box sx={{ height: 300, mb: 2 }}>
                      <img 
                        src={portfolioDetails.plots.performance} 
                        alt="Performance Chart" 
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    </Box>
                  )}
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Allocation History</Typography>
                  {portfolioDetails.plots?.allocation && (
                    <Box sx={{ height: 200 }}>
                      <img 
                        src={portfolioDetails.plots.allocation} 
                        alt="Allocation History" 
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    </Box>
                  )}
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Final Allocation</Typography>
                  {portfolioDetails.plots?.pie && (
                    <Box sx={{ height: 200 }}>
                      <img 
                        src={portfolioDetails.plots.pie} 
                        alt="Final Allocation" 
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    </Box>
                  )}
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      
      {/* Backtest Results Dialog */}
      <Dialog
        open={backtestOpen}
        onClose={() => setBacktestOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Backtest Results</DialogTitle>
        <DialogContent>
          {backtestResults && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {backtestResults.portfolio_id || 'Portfolio Backtest'}
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Performance Summary</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Initial Investment</TableCell>
                          <TableCell>${backtestResults.initial_investment.toLocaleString()}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Final Portfolio Value</TableCell>
                          <TableCell>${backtestResults.final_portfolio.toLocaleString()}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Total Return</TableCell>
                          <TableCell>
                            {formatPercentage((backtestResults.final_portfolio / backtestResults.initial_investment) - 1)}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Total Trades</TableCell>
                          <TableCell>{backtestResults.num_trades}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell>{formatNumber(backtestResults.metrics?.sharpe_ratio || 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Max Drawdown</TableCell>
                          <TableCell>{formatPercentage(backtestResults.metrics?.max_drawdown || 0)}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Final Allocations</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Asset</TableCell>
                          <TableCell>Allocation</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(backtestResults.final_allocations || {}).map(([asset, allocation]) => (
                          <TableRow key={asset}>
                            <TableCell>{asset}</TableCell>
                            <TableCell>{formatPercentage(allocation)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Trade History (Latest 20)</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Date</TableCell>
                          <TableCell>Ticker</TableCell>
                          <TableCell>Action</TableCell>
                          <TableCell>Price</TableCell>
                          <TableCell>Shares</TableCell>
                          <TableCell>Value</TableCell>
                          <TableCell>Intensity</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {backtestResults.trades?.map((trade, index) => (
                          <TableRow key={index}>
                            <TableCell>{trade.date}</TableCell>
                            <TableCell>{trade.ticker}</TableCell>
                            <TableCell>{trade.action}</TableCell>
                            <TableCell>${formatNumber(trade.price)}</TableCell>
                            <TableCell>{trade.shares}</TableCell>
                            <TableCell>${formatNumber(trade.value)}</TableCell>
                            <TableCell>{formatNumber(trade.intensity, 3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Action Distribution</Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Asset</TableCell>
                          <TableCell>Holds</TableCell>
                          <TableCell>Buys</TableCell>
                          <TableCell>Sells</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(backtestResults.action_counts || {}).map(([ticker, counts]) => (
                          <TableRow key={ticker}>
                            <TableCell>{ticker}</TableCell>
                            <TableCell>{counts.holds}</TableCell>
                            <TableCell>{counts.buys}</TableCell>
                            <TableCell>{counts.sells}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBacktestOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Portfolio; 