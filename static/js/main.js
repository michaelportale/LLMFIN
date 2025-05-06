// DOM elements
const tickerSelect = document.getElementById('ticker-select');
const trainBtn = document.getElementById('train-btn');
const previewBtn = document.getElementById('preview-btn');
const downloadModelBtn = document.getElementById('download-model-btn');
const downloadPlotBtn = document.getElementById('download-plot-btn');
const plotImg = document.getElementById('plot');
const plotPlaceholder = document.getElementById('plot-placeholder');
const statusContainer = document.getElementById('status-container');
const noTraining = document.getElementById('no-training');
const trainingTicker = document.getElementById('training-ticker');
const statusMessage = document.getElementById('status-message');
const trainingProgress = document.getElementById('training-progress');
const metricsContainer = document.getElementById('metrics-container');
const alertContainer = document.getElementById('alert-container');

// Tab navigation
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const tabId = e.target.getAttribute('data-tab');
    
    // Update active tab
    document.querySelectorAll('.nav-link').forEach(nav => {
      nav.classList.remove('active');
    });
    e.target.classList.add('active');
    
    // Show selected tab
    document.querySelectorAll('.content-tab').forEach(tab => {
      tab.classList.add('d-none');
    });
    document.getElementById(tabId + '-tab').classList.remove('d-none');
    
    // Load data for specific tabs
    if (tabId === 'manage') {
      loadModels();
    } else if (tabId === 'data') {
      loadDataFiles();
    }
  });
});

// Fetch tickers on page load
window.addEventListener('load', async () => {
  console.log('Page loaded, initializing...');
  await fetchTickers();
});

// Fetch tickers function
async function fetchTickers() {
  try {
    console.log('Fetching tickers...');
    const res = await fetch('/tickers');
    console.log('Ticker response:', res);
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const tickers = await res.json();
    console.log('Tickers received:', tickers);
    
    if (tickers.length === 0) {
      showAlert('No ticker data found. Please add data files in the Data Management tab.', 'warning');
      return;
    }
    
    if (!tickerSelect) {
      console.error('Ticker select element not found!');
      return;
    }
    
    console.log('Populating dropdown with tickers:', tickers);
    tickerSelect.innerHTML = tickers.map(t => `<option value="${t}">${t}</option>`).join('');
  } catch (error) {
    console.error('Error fetching tickers:', error);
    showAlert('Error loading ticker data: ' + error.message, 'danger');
  }
}

// Show preview
previewBtn.addEventListener('click', async () => {
  console.log('Preview button clicked');
  const ticker = tickerSelect.value;
  if (!ticker) {
    console.warn('No ticker selected');
    showAlert('Please select a ticker first', 'warning');
    return;
  }
  
  try {
    const res = await fetch(`/ticker_preview/${ticker}`);
    const data = await res.json();
    
    if (res.ok) {
      // Update preview modal
      document.getElementById('preview-modal-title').textContent = `${ticker} Data Preview`;
      document.getElementById('preview-date-range').textContent = `${data.start_date} to ${data.end_date}`;
      document.getElementById('preview-days').textContent = data.days;
      document.getElementById('preview-price').textContent = `$${data.latest_price.toFixed(2)}`;
      
      const changeClass = data.price_change_pct >= 0 ? 'text-success' : 'text-danger';
      const changeIcon = data.price_change_pct >= 0 ? '↑' : '↓';
      document.getElementById('preview-change').textContent = `${changeIcon} ${Math.abs(data.price_change_pct).toFixed(2)}%`;
      document.getElementById('preview-change').className = `fw-bold ${changeClass}`;
      
      document.getElementById('preview-plot').src = data.preview_plot;
      
      // Open the modal
      new bootstrap.Modal(document.getElementById('preview-modal')).show();
    } else {
      showAlert(data.error || 'Error loading ticker preview', 'danger');
    }
  } catch (error) {
    console.error('Preview error:', error);
    showAlert('Error: ' + error.message, 'danger');
  }
});

// Model types for deep learning
const MODEL_TYPES = {
    RL: 'Reinforcement Learning',
    LSTM: 'LSTM Network',
    SENTIMENT: 'Sentiment Enhanced',
    TRANSFORMER: 'Transformer Model'
};

// Add LSTM network options to the UI
function initDeepLearningOptions() {
    // Add model type selector
    const algorithmSection = document.querySelector('#algorithm').closest('.mb-2');
    if (!algorithmSection) return;
    
    // Create model type selection
    const modelTypeDiv = document.createElement('div');
    modelTypeDiv.className = 'mb-2';
    modelTypeDiv.innerHTML = `
        <label for="model-type" class="form-label">Model Type:</label>
        <select class="form-select" id="model-type">
            <option value="RL" selected>Reinforcement Learning</option>
            <option value="LSTM">LSTM Network</option>
            <option value="SENTIMENT">Sentiment Enhanced</option>
            <option value="TRANSFORMER">Transformer Model</option>
        </select>
        <small class="text-muted">Different model architectures for various prediction tasks</small>
    `;
    
    // Insert before the algorithm selection
    algorithmSection.parentNode.insertBefore(modelTypeDiv, algorithmSection);
    
    // Add event listener to update available options based on model type
    document.getElementById('model-type').addEventListener('change', updateModelOptions);
    
    // Add LSTM-specific options (initially hidden)
    const lstmOptionsDiv = document.createElement('div');
    lstmOptionsDiv.id = 'lstm-options';
    lstmOptionsDiv.className = 'mb-2 d-none';
    lstmOptionsDiv.innerHTML = `
        <hr>
        <h6>LSTM Configuration</h6>
        <div class="mb-2">
            <label for="lstm-layers" class="form-label small">LSTM Layers:</label>
            <select class="form-select form-select-sm" id="lstm-layers">
                <option value="1">1 (Simple)</option>
                <option value="2" selected>2 (Standard)</option>
                <option value="3">3 (Complex)</option>
            </select>
        </div>
        <div class="mb-2">
            <label for="sequence-length" class="form-label small">Sequence Length:</label>
            <select class="form-select form-select-sm" id="sequence-length">
                <option value="5">5 days</option>
                <option value="10">10 days</option>
                <option value="20" selected>20 days</option>
                <option value="50">50 days</option>
            </select>
            <small class="text-muted">Historical window for time series prediction</small>
        </div>
        <div class="mb-2">
            <label for="prediction-horizon" class="form-label small">Prediction Horizon:</label>
            <select class="form-select form-select-sm" id="prediction-horizon">
                <option value="1" selected>1 day</option>
                <option value="3">3 days</option>
                <option value="5">5 days</option>
                <option value="10">10 days</option>
            </select>
        </div>
    `;
    
    // Add sentiment analysis options (initially hidden)
    const sentimentOptionsDiv = document.createElement('div');
    sentimentOptionsDiv.id = 'sentiment-options';
    sentimentOptionsDiv.className = 'mb-2 d-none';
    sentimentOptionsDiv.innerHTML = `
        <hr>
        <h6>Sentiment Analysis</h6>
        <div class="mb-2">
            <label for="sentiment-sources" class="form-label small">Data Sources:</label>
            <select class="form-select form-select-sm" id="sentiment-sources" multiple>
                <option value="news" selected>Financial News</option>
                <option value="twitter">Twitter/X</option>
                <option value="reddit">Reddit</option>
                <option value="sec">SEC Filings</option>
            </select>
            <small class="text-muted">Hold Ctrl/Cmd to select multiple</small>
        </div>
        <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="realtime-sentiment" checked>
            <label class="form-check-label" for="realtime-sentiment">Real-time updates</label>
        </div>
    `;
    
    // Insert these options into the advanced options section
    const advancedOptions = document.getElementById('advancedOptions');
    if (advancedOptions) {
        advancedOptions.querySelector('.card-body').appendChild(lstmOptionsDiv);
        advancedOptions.querySelector('.card-body').appendChild(sentimentOptionsDiv);
    }
}

// Update available options based on selected model type
function updateModelOptions() {
    const modelType = document.getElementById('model-type').value;
    
    // Hide all model-specific option sections
    document.getElementById('lstm-options').classList.add('d-none');
    document.getElementById('sentiment-options').classList.add('d-none');
    
    // Show options specific to the selected model type
    if (modelType === 'LSTM') {
        document.getElementById('lstm-options').classList.remove('d-none');
    } else if (modelType === 'SENTIMENT') {
        document.getElementById('sentiment-options').classList.remove('d-none');
        document.getElementById('lstm-options').classList.remove('d-none');
    }
}

// Updated startTraining function to include deep learning parameters
function startTraining() {
    const ticker = document.getElementById('ticker-select').value;
    if (!ticker) {
        showAlert('warning', 'Please select a ticker first');
        return;
    }
    
    // Get standard training parameters
    const timesteps = document.getElementById('timesteps').value;
    const algorithm = document.getElementById('algorithm').value;
    
    // Get deep learning parameters
    const modelType = document.getElementById('model-type')?.value || 'RL';
    let deepLearningParams = {};
    
    if (modelType === 'LSTM' || modelType === 'SENTIMENT') {
        deepLearningParams = {
            lstm_layers: document.getElementById('lstm-layers')?.value || '2',
            sequence_length: document.getElementById('sequence-length')?.value || '20',
            prediction_horizon: document.getElementById('prediction-horizon')?.value || '1'
        };
    }
    
    if (modelType === 'SENTIMENT') {
        // Get selected sentiment sources
        const sourcesSelect = document.getElementById('sentiment-sources');
        const selectedSources = Array.from(sourcesSelect?.selectedOptions || [])
            .map(option => option.value);
        
        deepLearningParams.sentiment_sources = selectedSources;
        deepLearningParams.realtime_updates = document.getElementById('realtime-sentiment')?.checked || false;
    }
    
    // Hide any active alerts
    clearAlerts();
    
    // Show the status container and hide the "no training" message
    document.getElementById('status-container').classList.remove('d-none');
    document.getElementById('no-training').classList.add('d-none');
    
    // Set the ticker in the status display
    document.getElementById('training-ticker').textContent = ticker;
    
    // Update the status message
    const statusMessage = document.getElementById('status-message');
    statusMessage.textContent = `Initializing ${MODEL_TYPES[modelType]} model...`;
    
    // Reset the progress bar
    const progressBar = document.getElementById('training-progress');
    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    
    // Show the spinner
    document.getElementById('training-spinner').classList.remove('d-none');
    
    // Disable the train button
    const trainButton = document.getElementById('train-btn');
    trainButton.disabled = true;
    trainButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
    
    // Send a POST request to start training
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            ticker: ticker,
            timesteps: timesteps,
            algorithm: algorithm,
            model_type: modelType,
            early_stopping: document.getElementById('early-stopping')?.checked || false,
            params: {
                lr: document.getElementById('learning-rate')?.value || '0.0001',
                batch_size: document.getElementById('batch-size')?.value || '64',
                patience: document.getElementById('patience')?.value || '5',
                min_delta: document.getElementById('min-improvement')?.value || '0.01',
                ...deepLearningParams
            }
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        // Start polling for progress
        checkTrainingProgress(data.task_id);
    })
    .catch(error => {
        console.error('Training error:', error);
        resetTrainingUI();
        showAlert('danger', `Training error: ${error.message}`);
    });
}

// Initialize deep learning options when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    
    // Set up Deep Learning options
    initDeepLearningOptions();
    
    // ... existing code ...
});

// Train button click handler
trainBtn.addEventListener('click', async () => {
  console.log('Train button clicked');
  const ticker = tickerSelect.value;
  const timestepsElement = document.getElementById('timesteps');
  const timesteps = timestepsElement ? timestepsElement.value : 50000;
  
  if (!ticker) {
    showAlert('Please select a ticker', 'warning');
    return;
  }
  
  // Get advanced options
  const algorithm = document.getElementById('algorithm')?.value ?? 'PPO';
  const earlyStoppingEnabled = document.getElementById('early-stopping')?.checked ?? true;
  const patience = document.getElementById('patience')?.value ?? 5;
  const minImprovement = document.getElementById('min-improvement')?.value ?? 0.01;
  const learningRate = document.getElementById('learning-rate')?.value ?? 0.0001;
  const batchSize = document.getElementById('batch-size')?.value ?? 64;
  
  // First check if there's an existing model we can resume
  try {
    const checkRes = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        ticker, 
        timesteps: parseInt(timesteps), 
        resume: false,
        algorithm: algorithm,
        early_stopping: earlyStoppingEnabled,
        patience: parseInt(patience),
        min_improvement: parseFloat(minImprovement),
        learning_rate: parseFloat(learningRate),
        batch_size: parseInt(batchSize)
      })
    });
    
    const checkData = await checkRes.json();
    
    if (!checkRes.ok) {
      showAlert(checkData.error || 'Error checking model status', 'danger');
      return;
    }
    
    let shouldResume = false;
    
    // If we can resume, ask the user if they want to
    if (checkData.can_resume) {
      shouldResume = confirm(`An existing ${algorithm} model for ${ticker} was found. Would you like to resume training from the checkpoint? 
      
Click 'OK' to resume training, or 'Cancel' to start fresh.`);
    }
    
    // Show training status
    statusContainer.classList.remove('d-none');
    noTraining.classList.add('d-none');
    trainingTicker.textContent = `${ticker} (${algorithm})`;
    statusMessage.textContent = shouldResume ? 'Resuming training...' : 'Initializing training...';
    trainingProgress.style.width = shouldResume ? '25%' : '5%';
    
    // If user canceled the resume, or there's no model to resume, start fresh
    const res = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        ticker, 
        timesteps: parseInt(timesteps), 
        resume: shouldResume,
        algorithm: algorithm,
        early_stopping: earlyStoppingEnabled,
        patience: parseInt(patience),
        min_improvement: parseFloat(minImprovement),
        learning_rate: parseFloat(learningRate),
        batch_size: parseInt(batchSize)
      })
    });
    
    const data = await res.json();
    
    if (res.ok) {
      statusMessage.textContent = data.message;
      
      // Start polling for training status
      pollTrainingStatus(ticker);
    } else {
      showAlert(data.error || 'Error starting training', 'danger');
      statusContainer.classList.add('d-none');
      noTraining.classList.remove('d-none');
    }
  } catch (error) {
    console.error('Training error:', error);
    showAlert('Error: ' + error.message, 'danger');
    statusContainer.classList.add('d-none');
    noTraining.classList.remove('d-none');
  }
});

// Add event listener to toggle early stopping options
document.addEventListener('DOMContentLoaded', () => {
  const earlyStoppingCheckbox = document.getElementById('early-stopping');
  const earlyStoppingOptions = document.getElementById('early-stopping-options');
  
  if (earlyStoppingCheckbox && earlyStoppingOptions) {
    earlyStoppingCheckbox.addEventListener('change', () => {
      if (earlyStoppingCheckbox.checked) {
        earlyStoppingOptions.style.display = 'block';
      } else {
        earlyStoppingOptions.style.display = 'none';
      }
    });
  }
});

// Poll for training status
function pollTrainingStatus(ticker) {
  const pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/training_status/${ticker}`);
      const status = await res.json();
      
      if (status.status === 'not_started') {
        statusMessage.textContent = 'Training not started';
        trainingProgress.style.width = '0%';
      } else if (status.status === 'started') {
        statusMessage.textContent = 'Training in progress...';
        // Simulate progress based on time elapsed
        const elapsed = (Date.now() - status.start_time * 1000) / 1000;
        const progress = Math.min(90, Math.floor(elapsed / 60 * 10)); // Rough estimate
        trainingProgress.style.width = `${progress}%`;
      } else if (status.status === 'completed') {
        statusMessage.textContent = 'Training completed successfully!';
        trainingProgress.style.width = '100%';
        
        clearInterval(pollInterval);
        
        // After a short delay, update the UI
        setTimeout(() => {
          statusContainer.classList.add('d-none');
          noTraining.classList.remove('d-none');
          
          // Show performance plot
          plotPlaceholder.style.display = 'none';
          plotImg.src = `/plot/${ticker}?t=${Date.now()}`;
          plotImg.style.display = 'block';
          
          // Enable download buttons
          downloadModelBtn.disabled = false;
          downloadPlotBtn.disabled = false;
          
          // Load and display metrics
          loadMetrics(ticker);
        }, 1500);
      } else if (status.status === 'error') {
        statusMessage.textContent = `Error: ${status.error}`;
        trainingProgress.style.width = '100%';
        trainingProgress.classList.remove('bg-primary');
        trainingProgress.classList.add('bg-danger');
        
        clearInterval(pollInterval);
        
        setTimeout(() => {
          statusContainer.classList.add('d-none');
          noTraining.classList.remove('d-none');
          showAlert(`Training error: ${status.error}`, 'danger');
        }, 3000);
      }
    } catch (error) {
      console.error('Error polling status:', error);
    }
  }, 5000);
}

// Load and display performance metrics
async function loadMetrics(ticker) {
  try {
    const res = await fetch(`/metrics/${ticker}`);
    
    if (!res.ok) {
      return;
    }
    
    const metrics = await res.json();
    
    // Show metrics container
    metricsContainer.classList.remove('d-none');
    
    // Update model metrics
    document.getElementById('model-return').textContent = formatPercent(metrics.model.total_return);
    document.getElementById('model-final').textContent = formatDollars(metrics.model.final_value);
    document.getElementById('model-sharpe').textContent = metrics.model.sharpe_ratio.toFixed(2);
    document.getElementById('model-drawdown').textContent = formatPercent(metrics.model.max_drawdown);
    
    // Update benchmark metrics
    document.getElementById('bench-return').textContent = formatPercent(metrics.benchmark.total_return);
    document.getElementById('bench-final').textContent = formatDollars(metrics.benchmark.final_value);
    document.getElementById('bench-sharpe').textContent = metrics.benchmark.sharpe_ratio.toFixed(2);
    document.getElementById('bench-drawdown').textContent = formatPercent(metrics.benchmark.max_drawdown);

    // Update performance comparison alert
    const performanceAlert = document.getElementById('performance-alert');
    const performanceText = document.getElementById('performance-text');

    if (metrics.outperformance > 0) {
      performanceAlert.className = 'alert alert-success';
      performanceText.textContent = `The RL model outperformed the buy & hold strategy by ${formatPercent(metrics.outperformance)}. This suggests the model was able to effectively time the market for this ticker.`;
    } else {
      performanceAlert.className = 'alert alert-warning';
      performanceText.textContent = `The buy & hold strategy outperformed the RL model by ${formatPercent(Math.abs(metrics.outperformance))}. The model may need more training or parameter tuning.`;
    }
  } catch (error) {
    console.error('Error loading metrics:', error);
  }
}

// Download model button
downloadModelBtn.addEventListener('click', () => {
  const ticker = tickerSelect.value;
  window.location.href = `/download/model/${ticker}`;
});

// Download plot button
downloadPlotBtn.addEventListener('click', () => {
  const ticker = tickerSelect.value;
  const a = document.createElement('a');
  a.href = plotImg.src;
  a.download = `${ticker}_performance.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// Preview modal train button
document.getElementById('preview-train-btn').addEventListener('click', () => {
  // Hide modal
  bootstrap.Modal.getInstance(document.getElementById('preview-modal')).hide();
  // Click train button
  trainBtn.click();
});

// Load models for the manage tab
async function loadModels() {
  try {
    const res = await fetch('/available_models');
    const models = await res.json();
    
    const tbody = document.getElementById('models-table');
    
    if (models.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No trained models found</td></tr>';
      return;
    }
    
    tbody.innerHTML = models.map(model => `
      <tr>
        <td>${model.ticker}</td>
        <td><span class="badge bg-primary">${model.algorithm}</span></td>
        <td>${model.trained_date}</td>
        <td>
          ${model.has_metrics ? 
            '<span class="badge bg-success">Metrics Available</span>' : 
            '<span class="badge bg-secondary">No Metrics</span>'}
        </td>
        <td>
          <div class="btn-group btn-group-sm" role="group">
            <button class="btn btn-outline-primary view-model-btn" data-ticker="${model.ticker}">
              <i class="bi bi-eye"></i>
            </button>
            <button class="btn btn-outline-secondary backtest-btn" data-ticker="${model.ticker}">
              <i class="bi bi-graph-up"></i>
            </button>
            <a href="/download/model/${model.ticker}" class="btn btn-outline-success">
              <i class="bi bi-download"></i>
            </a>
          </div>
        </td>
      </tr>
    `).join('');
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-model-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const ticker = btn.getAttribute('data-ticker');
        
        // Switch to train tab
        document.querySelector('[data-tab="train"]').click();
        
        // Select the ticker
        tickerSelect.value = ticker;
        
        // Show the plot and metrics
        plotPlaceholder.style.display = 'none';
        plotImg.src = `/plot/${ticker}?t=${Date.now()}`;
        plotImg.style.display = 'block';
        
        // Enable download buttons
        downloadModelBtn.disabled = false;
        downloadPlotBtn.disabled = false;
        
        // Load metrics
        loadMetrics(ticker);
      });
    });
    
    // Add event listeners to backtest buttons
    document.querySelectorAll('.backtest-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const ticker = btn.getAttribute('data-ticker');
        await runBacktest(ticker);
      });
    });
  } catch (error) {
    showAlert('Error loading models: ' + error.message, 'danger');
  }
}

// Run a backtest for a trained model
async function runBacktest(ticker) {
  try {
    document.getElementById('backtest-card').classList.add('d-none');
    showAlert(`Running backtest for ${ticker}...`, 'info');
    
    const res = await fetch(`/backtest/${ticker}`, {
      method: 'POST'
    });
    
    const data = await res.json();
    
    if (!res.ok) {
      showAlert(data.error || 'Error running backtest', 'danger');
      return;
    }
    
    // Update backtest card
    document.getElementById('backtest-ticker').textContent = `${ticker} (${data.algorithm})`;
    document.getElementById('backtest-final').textContent = formatDollars(data.final_portfolio);
    document.getElementById('backtest-return').textContent = formatPercent(data.metrics.total_return);
    document.getElementById('backtest-trades').textContent = data.num_trades;
    document.getElementById('backtest-sharpe').textContent = data.metrics.sharpe_ratio.toFixed(2);
    
    // Populate trades table
    const tradesTable = document.getElementById('trades-table');
    tradesTable.innerHTML = data.trades.map(trade => `
      <tr>
        <td>${trade.date}</td>
        <td class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">${trade.action}</td>
        <td>$${trade.price.toFixed(2)}</td>
        <td>${trade.shares}</td>
        <td>$${trade.value.toFixed(2)}</td>
      </tr>
    `).join('');
    
    // Create action distribution chart
    const actionsChart = document.getElementById('actions-chart');
    if (window.actionChart) {
      window.actionChart.destroy();
    }
    
    window.actionChart = new Chart(actionsChart, {
      type: 'pie',
      data: {
        labels: ['Hold', 'Buy', 'Sell'],
        datasets: [{
          data: [
            data.action_counts.hold || 0,
            data.action_counts.buy || 0,
            data.action_counts.sell || 0
          ],
          backgroundColor: ['#6c757d', '#28a745', '#dc3545']
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    });
    
    // Show backtest card
    document.getElementById('backtest-card').classList.remove('d-none');
    
    // Remove alert
    removeAlerts();
  } catch (error) {
    showAlert('Error running backtest: ' + error.message, 'danger');
  }
}

// Load data files for the data tab
async function loadDataFiles() {
  try {
    const res = await fetch('/tickers');
    const tickers = await res.json();
    
    const tbody = document.getElementById('data-table');
    
    if (tickers.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No data files found</td></tr>';
      return;
    }
    
    // Get file details using ticker preview
    const rows = [];
    
    for (const ticker of tickers) {
      try {
        const previewRes = await fetch(`/ticker_preview/${ticker}`);
        if (previewRes.ok) {
          const data = await previewRes.json();
          
          rows.push(`
            <tr>
              <td>${ticker}</td>
              <td>${data.end_date}</td>
              <td>${data.days}</td>
              <td>
                <div class="btn-group btn-group-sm" role="group">
                  <button class="btn btn-outline-primary preview-data-btn" data-ticker="${ticker}">
                    <i class="bi bi-eye"></i>
                  </button>
                  <a href="/download/csv/${ticker}" class="btn btn-outline-success">
                    <i class="bi bi-download"></i>
                  </a>
                </div>
              </td>
            </tr>
          `);
        }
      } catch (e) {
        console.error(`Error getting preview for ${ticker}:`, e);
      }
    }
    
    tbody.innerHTML = rows.join('');
    
    // Add event listeners to preview buttons
    document.querySelectorAll('.preview-data-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const ticker = btn.getAttribute('data-ticker');
        
        // Set the ticker in the dropdown and click preview
        tickerSelect.value = ticker;
        previewBtn.click();
      });
    });
  } catch (error) {
    showAlert('Error loading data files: ' + error.message, 'danger');
  }
}

// Fetch data form submission
document.getElementById('fetch-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const tickersInput = document.getElementById('tickers-input').value;
  const tickers = tickersInput.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
  
  if (tickers.length === 0) {
    showAlert('Please enter at least one ticker symbol', 'warning');
    return;
  }
  
  const period = document.getElementById('period-select').value;
  const interval = document.getElementById('interval-select').value;
  
  showAlert(`Fetching data for ${tickers.join(', ')}...`, 'info');
  
  try {
    const res = await fetch('/fetch_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers, period, interval })
    });
    
    const data = await res.json();
    
    if (res.ok) {
      showAlert(data.message, 'success');
      
      // After a short delay, reload the data files
      setTimeout(() => {
        loadDataFiles();
      }, 2000);
    } else {
      showAlert(data.error || 'Error fetching data', 'danger');
    }
  } catch (error) {
    showAlert('Error: ' + error.message, 'danger');
  }
});

// Utility function to show alerts
function showAlert(message, type = 'info') {
  const alert = document.createElement('div');
  alert.className = `alert alert-${type} alert-dismissible fade show`;
  alert.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  alertContainer.appendChild(alert);
  
  // Auto dismiss after 5 seconds
  setTimeout(() => {
    if (alert.parentNode === alertContainer) {
      alert.remove();
    }
  }, 5000);
}

// Utility function to remove all alerts
function removeAlerts() {
  alertContainer.innerHTML = '';
}

// Format number as dollars
function formatDollars(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
}

// Format number as percent
function formatPercent(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
}