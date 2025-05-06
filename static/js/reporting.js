/**
 * Reporting functionality for FinPort
 */

// Global variables
let reportUrl = null;
let selectedModels = [];
let availableMetrics = [];

// Mobile detection
const isMobile = () => window.innerWidth < 768;

// Configure mobile-optimized chart options
const getChartOptions = () => {
    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: !isMobile(),
                position: 'top',
            },
            tooltip: {
                enabled: true,
                intersect: false,
                mode: 'index'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    font: {
                        size: isMobile() ? 10 : 12
                    }
                }
            },
            x: {
                ticks: {
                    font: {
                        size: isMobile() ? 10 : 12
                    },
                    maxRotation: isMobile() ? 45 : 0
                }
            }
        },
        interaction: {
            mode: 'nearest'
        }
    };
    
    return baseOptions;
};

// Update device-specific adjustments when window resizes
window.addEventListener('resize', () => {
    // If we have charts, they'll automatically rerender
    // due to the responsive: true option
});

// Add data-labels to table cells for mobile view
function addDataLabelsToRow(row, headers) {
    const cells = row.querySelectorAll('td');
    cells.forEach((cell, index) => {
        if (index < headers.length) {
            cell.setAttribute('data-label', headers[index]);
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize reporting UI when the tab is first shown
    document.querySelector('a[data-tab="reporting"]').addEventListener('click', initReportingTab);
    
    // Add event listeners for report generation
    document.getElementById('generate-report-btn').addEventListener('click', generateReport);
    document.getElementById('email-report-btn').addEventListener('click', toggleEmailForm);
    document.getElementById('send-email-btn').addEventListener('click', sendReportEmail);
    document.getElementById('subscribe-email').addEventListener('change', handleSubscription);
    
    // Add event listeners for model comparison
    document.getElementById('compare-models-btn').addEventListener('click', compareModels);
    document.getElementById('comparison-report-btn').addEventListener('click', generateComparisonReport);
    
    // Add event listener for custom metric
    document.getElementById('add-metric-btn').addEventListener('click', showAddMetricModal);

    // Add buttons for LSTM explanation and sentiment analysis
    const reportCard = document.querySelector('#reporting-tab .card-body');
    if (reportCard) {
        const explainButton = document.createElement('div');
        explainButton.className = 'mt-3';
        explainButton.innerHTML = `
            <div class="d-grid gap-2">
                <button id="explain-model-btn" class="btn btn-outline-info">
                    <i class="bi bi-lightbulb"></i> Explain Model Prediction
                </button>
                <button id="analyze-sentiment-btn" class="btn btn-outline-info">
                    <i class="bi bi-chat-square-text"></i> Analyze Market Sentiment
                </button>
                <button id="generate-summary-btn" class="btn btn-outline-info">
                    <i class="bi bi-file-text"></i> Generate Trading Summary
                </button>
            </div>
        `;
        
        // Add after the email form
        const emailForm = document.getElementById('email-form');
        if (emailForm) {
            emailForm.parentNode.insertBefore(explainButton, emailForm.nextSibling);
            
            // Add event listeners
            document.getElementById('explain-model-btn').addEventListener('click', () => {
                const ticker = document.getElementById('report-ticker-select').value;
                if (!ticker) {
                    showAlert('warning', 'Please select a model first');
                    return;
                }
                displayModelExplanation(ticker);
            });
            
            document.getElementById('analyze-sentiment-btn').addEventListener('click', () => {
                const ticker = document.getElementById('report-ticker-select').value;
                if (!ticker) {
                    showAlert('warning', 'Please select a ticker first');
                    return;
                }
                displaySentimentAnalysis(ticker, ['news', 'twitter']);
            });
            
            document.getElementById('generate-summary-btn').addEventListener('click', () => {
                const ticker = document.getElementById('report-ticker-select').value;
                if (!ticker) {
                    showAlert('warning', 'Please select a ticker first');
                    return;
                }
                generateTradingSummary(ticker);
            });
        }
    }
});

/**
 * Initialize the reporting tab
 */
function initReportingTab() {
    loadAvailableModels();
    loadCustomMetrics();
}

/**
 * Load available models for reporting
 */
function loadAvailableModels() {
    fetch('/available_models')
        .then(response => response.json())
        .then(data => {
            const tickerSelect = document.getElementById('report-ticker-select');
            const comparisonModels = document.getElementById('comparison-models');
            
            // Clear existing options
            tickerSelect.innerHTML = '<option value="" disabled selected>Choose a model...</option>';
            comparisonModels.innerHTML = '';
            
            if (data.length === 0) {
                comparisonModels.innerHTML = `
                    <div class="alert alert-info">
                        No models available. Train models first.
                    </div>
                `;
                return;
            }
            
            // Add model options to select
            data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.ticker;
                option.textContent = `${model.ticker} (${model.algorithm})`;
                tickerSelect.appendChild(option);
                
                // Add checkbox for comparison
                const checkboxDiv = document.createElement('div');
                checkboxDiv.className = 'form-check';
                checkboxDiv.innerHTML = `
                    <input class="form-check-input model-check" type="checkbox" 
                           id="model-${model.ticker}" value="${model.ticker}" 
                           data-algorithm="${model.algorithm}">
                    <label class="form-check-label" for="model-${model.ticker}">
                        ${model.ticker} (${model.algorithm})
                    </label>
                `;
                comparisonModels.appendChild(checkboxDiv);
            });
            
            // Add event listeners to model checkboxes
            document.querySelectorAll('.model-check').forEach(checkbox => {
                checkbox.addEventListener('change', updateSelectedModels);
            });
        })
        .catch(error => {
            console.error('Error loading models:', error);
            showAlert('error', 'Failed to load models. See console for details.');
        });
}

/**
 * Update the list of selected models for comparison
 */
function updateSelectedModels() {
    selectedModels = [];
    document.querySelectorAll('.model-check:checked').forEach(checkbox => {
        selectedModels.push({
            ticker: checkbox.value,
            algorithm: checkbox.dataset.algorithm
        });
    });
}

/**
 * Load custom metrics
 */
function loadCustomMetrics() {
    fetch('/custom_metrics')
        .then(response => response.json())
        .then(data => {
            availableMetrics = data;
            
            const metricTable = document.getElementById('custom-metrics-table').querySelector('tbody');
            
            if (data.length === 0) {
                metricTable.innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center text-muted">No custom metrics defined</td>
                    </tr>
                `;
                return;
            }
            
            // Clear existing content
            metricTable.innerHTML = '';
            
            // Add metrics to table
            data.forEach(metric => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td data-label="Metric">${metric.name}</td>
                    <td data-label="Description">${metric.description}</td>
                    <td data-label="Format">${metric.format}</td>
                    <td data-label="Actions">
                        <button class="btn btn-sm btn-outline-secondary view-metric-btn" 
                                data-metric-id="${metric.metric_id}"
                                aria-label="View details for ${metric.name}">
                            <i class="bi bi-eye"></i>
                        </button>
                    </td>
                `;
                metricTable.appendChild(row);
                
                // Add to metrics checkboxes for comparison
                addMetricToComparisonList(metric);
            });
            
            // Add event listeners to view buttons
            document.querySelectorAll('.view-metric-btn').forEach(btn => {
                btn.addEventListener('click', () => viewMetricDetails(btn.dataset.metricId));
            });
        })
        .catch(error => {
            console.error('Error loading custom metrics:', error);
            showAlert('error', 'Failed to load custom metrics. See console for details.');
        });
}

/**
 * Add a metric to the comparison list
 */
function addMetricToComparisonList(metric) {
    // Check if we already have this metric in the list
    const existingMetric = document.getElementById(`metric-${metric.metric_id}`);
    if (existingMetric) return;
    
    // Get the metrics section
    const metricsSection = document.querySelector('.metric-check').closest('.row');
    
    // Find which column to add it to (add to the shorter column)
    const columns = metricsSection.querySelectorAll('.col-md-6');
    const column = Array.from(columns)
        .sort((a, b) => a.children.length - b.children.length)[0];
    
    // Create and add the checkbox
    const checkboxDiv = document.createElement('div');
    checkboxDiv.className = 'form-check';
    checkboxDiv.innerHTML = `
        <input class="form-check-input metric-check" type="checkbox" 
               id="metric-${metric.metric_id}" value="${metric.metric_id}">
        <label class="form-check-label" for="metric-${metric.metric_id}">
            ${metric.name}
        </label>
    `;
    column.appendChild(checkboxDiv);
}

/**
 * Generate a PDF report for a model
 */
function generateReport() {
    const ticker = document.getElementById('report-ticker-select').value;
    
    if (!ticker) {
        showAlert('warning', 'Please select a model first.');
        return;
    }
    
    // Show loading state
    const generateBtn = document.getElementById('generate-report-btn');
    const originalText = generateBtn.innerHTML;
    generateBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    generateBtn.disabled = true;
    
    // Request report generation
    fetch(`/generate_report/${ticker}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        reportUrl = data.report_url;
        
        // Show report in iframe
        const reportPreview = document.getElementById('report-preview');
        reportPreview.src = reportUrl;
        reportPreview.classList.remove('initially-hidden');
        document.getElementById('report-placeholder').classList.add('initially-hidden');
        
        // Enable download button
        const downloadBtn = document.getElementById('download-report-btn');
        downloadBtn.disabled = false;
        downloadBtn.onclick = () => window.open(reportUrl, '_blank');
        
        showAlert('success', 'Report generated successfully.');
    })
    .catch(error => {
        console.error('Error generating report:', error);
        showAlert('error', error.message || 'Failed to generate report. See console for details.');
    })
    .finally(() => {
        // Reset button
        generateBtn.innerHTML = originalText;
        generateBtn.disabled = false;
    });
}

/**
 * Toggle the email form display
 */
function toggleEmailForm() {
    const emailForm = document.getElementById('email-form');
    emailForm.classList.toggle('d-none');
    
    if (!emailForm.classList.contains('d-none')) {
        document.getElementById('report-email').focus();
    }
}

/**
 * Send a report via email
 */
function sendReportEmail() {
    const ticker = document.getElementById('report-ticker-select').value;
    const email = document.getElementById('report-email').value;
    
    if (!ticker) {
        showAlert('warning', 'Please select a model first.');
        return;
    }
    
    if (!email) {
        showAlert('warning', 'Please enter an email address.');
        return;
    }
    
    if (!validateEmail(email)) {
        showAlert('warning', 'Please enter a valid email address.');
        return;
    }
    
    // Show loading state
    const sendBtn = document.getElementById('send-email-btn');
    const originalText = sendBtn.textContent;
    sendBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';
    sendBtn.disabled = true;
    
    // Send email request
    fetch(`/send_report_email/${ticker}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        showAlert('success', 'Report sent successfully.');
        
        // Hide the email form
        document.getElementById('email-form').classList.add('d-none');
    })
    .catch(error => {
        console.error('Error sending report email:', error);
        showAlert('error', error.message || 'Failed to send email. See console for details.');
    })
    .finally(() => {
        // Reset button
        sendBtn.textContent = originalText;
        sendBtn.disabled = false;
    });
}

/**
 * Handle subscription toggle
 */
function handleSubscription(event) {
    const isSubscribing = event.target.checked;
    const email = document.getElementById('report-email').value;
    
    if (!email) {
        showAlert('warning', 'Please enter an email address.');
        event.target.checked = false;
        return;
    }
    
    if (!validateEmail(email)) {
        showAlert('warning', 'Please enter a valid email address.');
        event.target.checked = false;
        return;
    }
    
    const endpoint = isSubscribing ? '/register_notification_email' : '/unregister_notification_email';
    
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        showAlert('success', data.message);
    })
    .catch(error => {
        console.error('Error changing subscription:', error);
        showAlert('error', error.message || 'Failed to update subscription. See console for details.');
        // Revert the checkbox state
        event.target.checked = !isSubscribing;
    });
}

/**
 * Compare selected models
 */
function compareModels() {
    if (selectedModels.length < 2) {
        showAlert('warning', 'Please select at least two models to compare.');
        return;
    }
    
    // Get selected metrics
    const selectedMetrics = Array.from(document.querySelectorAll('.metric-check:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedMetrics.length === 0) {
        showAlert('warning', 'Please select at least one metric for comparison.');
        return;
    }
    
    // Show loading state
    const compareBtn = document.getElementById('compare-models-btn');
    const originalText = compareBtn.innerHTML;
    compareBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Comparing...';
    compareBtn.disabled = true;
    
    // Request model comparison
    fetch('/model_comparison', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            tickers: selectedModels.map(model => model.ticker),
            algorithms: selectedModels.map(model => model.algorithm),
            metrics: selectedMetrics
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display comparison results
        displayComparisonResults(data, selectedMetrics);
    })
    .catch(error => {
        console.error('Error comparing models:', error);
        showAlert('error', error.message || 'Failed to compare models. See console for details.');
    })
    .finally(() => {
        // Reset button
        compareBtn.innerHTML = originalText;
        compareBtn.disabled = false;
    });
}

/**
 * Display model comparison results
 */
function displayComparisonResults(data, metrics) {
    const resultsContainer = document.getElementById('comparison-results');
    const placeholder = document.getElementById('comparison-placeholder');
    
    // Show results container
    resultsContainer.classList.remove('initially-hidden');
    placeholder.classList.add('initially-hidden');
    
    // Clear existing content
    resultsContainer.innerHTML = '';
    
    // Check if there's data to display
    const hasData = metrics.some(metric => data[metric] && data[metric].length > 0);
    
    if (!hasData) {
        resultsContainer.innerHTML = `
            <div class="alert alert-info">
                No comparison data available for the selected models and metrics.
            </div>
        `;
        return;
    }
    
    // Create a section for each metric
    metrics.forEach(metricId => {
        if (!data[metricId] || data[metricId].length === 0) {
            return; // Skip metrics with no data
        }
        
        // Get metric name
        const metricName = getMetricName(metricId);
        
        // Create metric section
        const metricSection = document.createElement('div');
        metricSection.className = 'mb-4';
        metricSection.innerHTML = `
            <h5>${metricName}</h5>
            <div class="table-responsive">
                <table class="table table-sm table-mobile-responsive">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody id="${metricId}-results">
                    </tbody>
                </table>
            </div>
        `;
        resultsContainer.appendChild(metricSection);
        
        // Add model values
        const resultsBody = document.getElementById(`${metricId}-results`);
        
        data[metricId].forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td data-label="Model">${item.model}</td>
                <td data-label="Value">${formatMetricValue(item.value, metricId)}</td>
            `;
            resultsBody.appendChild(row);
        });
    });
}

/**
 * Generate a model comparison report
 */
function generateComparisonReport() {
    if (selectedModels.length < 2) {
        showAlert('warning', 'Please select at least two models to compare.');
        return;
    }
    
    // Show loading state
    const reportBtn = document.getElementById('comparison-report-btn');
    const originalText = reportBtn.innerHTML;
    reportBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    reportBtn.disabled = true;
    
    // Request comparison report generation
    fetch('/generate_comparison_report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            tickers: selectedModels.map(model => model.ticker)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        reportUrl = data.report_url;
        
        // Show report in iframe
        const reportPreview = document.getElementById('report-preview');
        reportPreview.src = reportUrl;
        reportPreview.classList.remove('initially-hidden');
        document.getElementById('report-placeholder').classList.add('initially-hidden');
        
        // Enable download button
        const downloadBtn = document.getElementById('download-report-btn');
        downloadBtn.disabled = false;
        downloadBtn.onclick = () => window.open(reportUrl, '_blank');
        
        showAlert('success', 'Comparison report generated successfully.');
    })
    .catch(error => {
        console.error('Error generating comparison report:', error);
        showAlert('error', error.message || 'Failed to generate comparison report. See console for details.');
    })
    .finally(() => {
        // Reset button
        reportBtn.innerHTML = originalText;
        reportBtn.disabled = false;
    });
}

/**
 * Show form to add a custom metric
 */
function showAddMetricModal() {
    // TODO: Create a modal for adding custom metrics
    alert('Add custom metric functionality will be implemented in a future update.');
}

/**
 * View details of a custom metric
 */
function viewMetricDetails(metricId) {
    // Get the metric
    const metric = availableMetrics.find(m => m.metric_id === metricId);
    if (!metric) return;
    
    // For mobile, we'll use a simple alert with line breaks
    if (isMobile()) {
        alert(
            `Metric: ${metric.name}\n` +
            `Description: ${metric.description}\n` +
            `Formula: ${metric.formula || 'Built-in'}\n` +
            `Format: ${metric.format}`
        );
        return;
    }
    
    // On desktop we could use a modal or more sophisticated display
    alert(`Metric: ${metric.name}\nDescription: ${metric.description}\nFormula: ${metric.formula || 'Built-in'}\nFormat: ${metric.format}`);
}

/**
 * Format a metric value based on its type
 */
function formatMetricValue(value, metricId) {
    // Find the metric definition
    const metric = availableMetrics.find(m => m.metric_id === metricId);
    
    if (!metric) {
        // Default formatting if metric definition not found
        if (metricId.includes('return') || metricId.includes('drawdown')) {
            return `${(value * 100).toFixed(2)}%`;
        }
        return value.toFixed(4);
    }
    
    // Format based on metric type
    switch(metric.format) {
        case 'percentage':
            return `${(value * 100).toFixed(2)}%`;
        case 'currency':
            return `$${value.toFixed(2)}`;
        case 'integer':
            return Math.round(value);
        case 'decimal':
        default:
            return value.toFixed(4);
    }
}

/**
 * Get the display name for a metric
 */
function getMetricName(metricId) {
    // Find the metric in available metrics
    const metric = availableMetrics.find(m => m.metric_id === metricId);
    
    if (metric) {
        return metric.name;
    }
    
    // Default names for standard metrics
    switch(metricId) {
        case 'sharpe_ratio':
            return 'Sharpe Ratio';
        case 'sortino_ratio':
            return 'Sortino Ratio';
        case 'annualized_return':
            return 'Annualized Return';
        case 'max_drawdown':
            return 'Maximum Drawdown';
        case 'volatility':
            return 'Volatility';
        case 'win_rate':
            return 'Win Rate';
        case 'calmar_ratio':
            return 'Calmar Ratio';
        default:
            return metricId.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }
}

/**
 * Validate email format
 */
function validateEmail(email) {
    const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
}

/**
 * Show an alert message
 */
function showAlert(type, message) {
    const alertContainer = document.getElementById('alert-container');
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => {
            alert.remove();
        }, 150);
    }, 5000);
}

// Function to display LSTM model explanation
function displayModelExplanation(ticker, modelType = 'LSTM') {
    // Display loading state
    const reportPreview = document.getElementById('report-preview');
    reportPreview.classList.add('initially-hidden');
    
    const placeholder = document.getElementById('report-placeholder');
    placeholder.classList.remove('initially-hidden');
    placeholder.innerHTML = `
        <div class="d-flex flex-column align-items-center">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted">Generating model explanation...</p>
        </div>
    `;
    
    // Fetch model explanation
    fetch(`/explain_prediction/${ticker}?model_type=${modelType}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Hide the loading placeholder
            placeholder.classList.add('initially-hidden');
            
            // Create a container for visualization
            const container = document.createElement('div');
            container.id = 'explanation-container';
            container.className = 'p-3 border rounded';
            
            // Create the explanation content
            container.innerHTML = `
                <h4 class="mb-3">AI Model Explanation</h4>
                <p class="mb-3">${data.explanation.explanation_text.replace(/\n/g, '<br>')}</p>
                
                <div class="mb-3">
                    <h5>Feature Importance</h5>
                    <canvas id="feature-importance-chart" height="250"></canvas>
                </div>
                
                <div class="alert alert-info">
                    <h6 class="alert-heading">Key Insights:</h6>
                    <ul>
                        ${data.explanation.top_features.map(feature => `<li><strong>${feature}</strong> has significant influence on the prediction</li>`).join('')}
                    </ul>
                </div>
            `;
            
            // Show the container
            document.getElementById('report-container').appendChild(container);
            
            // Create feature importance chart
            createFeatureImportanceChart(data.explanation.features, data.explanation.importances);
            
            showAlert('success', 'Model explanation generated successfully');
        })
        .catch(error => {
            console.error('Error generating explanation:', error);
            placeholder.innerHTML = `
                <i class="bi bi-exclamation-triangle display-1 text-warning"></i>
                <p class="text-muted">Error generating explanation: ${error.message || 'Unknown error'}</p>
            `;
            showAlert('error', 'Failed to generate model explanation');
        });
}

// Create feature importance chart
function createFeatureImportanceChart(features, importances) {
    const ctx = document.getElementById('feature-importance-chart').getContext('2d');
    
    // Convert importances to percentages
    const importancePercentages = importances.map(value => value * 100);
    
    // Generate colors
    const colors = generateChartColors(features.length);
    
    // Create horizontal bar chart
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance (%)',
                data: importancePercentages,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.7', '1')),
                borderWidth: 1
            }]
        },
        options: getChartOptions()
    });
}

// Generate chart colors
function generateChartColors(count) {
    const baseColors = [
        'rgba(75, 192, 192, 0.7)',
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 206, 86, 0.7)',
        'rgba(255, 99, 132, 0.7)',
        'rgba(153, 102, 255, 0.7)'
    ];
    
    // If we have fewer colors than needed, repeat the pattern
    const colors = [];
    for (let i = 0; i < count; i++) {
        colors.push(baseColors[i % baseColors.length]);
    }
    
    return colors;
}

// Function to display sentiment analysis
function displaySentimentAnalysis(ticker, sources = ['news']) {
    // Display loading state
    const reportPreview = document.getElementById('report-preview');
    reportPreview.classList.add('initially-hidden');
    
    const placeholder = document.getElementById('report-placeholder');
    placeholder.classList.remove('initially-hidden');
    placeholder.innerHTML = `
        <div class="d-flex flex-column align-items-center">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted">Analyzing sentiment data...</p>
        </div>
    `;
    
    // Fetch sentiment analysis
    fetch(`/sentiment/${ticker}?sources=${sources.join(',')}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Hide the loading placeholder
            placeholder.classList.add('initially-hidden');
            
            // Create a container for sentiment analysis
            const container = document.createElement('div');
            container.id = 'sentiment-container';
            container.className = 'p-3 border rounded';
            
            // Create the sentiment content
            container.innerHTML = `
                <h4 class="mb-3">Sentiment Analysis for ${ticker}</h4>
                
                <div class="alert alert-${getSentimentAlertClass(data.analysis.avg_sentiment)}">
                    <h6 class="alert-heading">Sentiment Summary:</h6>
                    <p>${data.summary.replace(/\n/g, '<br>')}</p>
                </div>
                
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-header">Sentiment Metrics</div>
                            <div class="card-body">
                                <table class="table table-sm table-mobile-responsive">
                                    <tr>
                                        <td data-label="Metric">Average Sentiment</td>
                                        <td data-label="Value">${formatSentimentValue(data.analysis.avg_sentiment)}</td>
                                    </tr>
                                    <tr>
                                        <td data-label="Metric">Trend</td>
                                        <td data-label="Value">${data.analysis.trend}</td>
                                    </tr>
                                    <tr>
                                        <td data-label="Metric">Volatility</td>
                                        <td data-label="Value">${data.analysis.volatility.toFixed(3)}</td>
                                    </tr>
                                    <tr>
                                        <td data-label="Metric">Positive Days</td>
                                        <td data-label="Value">${data.analysis.positive_days}</td>
                                    </tr>
                                    <tr>
                                        <td data-label="Metric">Negative Days</td>
                                        <td data-label="Value">${data.analysis.negative_days}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <canvas id="sentiment-chart" height="250"></canvas>
                    </div>
                </div>
                
                <h5 class="mb-3">Daily Sentiment Data</h5>
                <div class="table-responsive">
                    <table class="table table-sm table-striped table-mobile-responsive">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Compound</th>
                                <th>Positive</th>
                                <th>Negative</th>
                                <th>Neutral</th>
                            </tr>
                        </thead>
                        <tbody id="sentiment-data-table">
                            ${createSentimentDataRows(data.data.slice(0, 10))}
                        </tbody>
                    </table>
                </div>
            `;
            
            // Show the container
            document.getElementById('report-container').appendChild(container);
            
            // Create sentiment chart
            createSentimentChart(data.data);
            
            showAlert('success', 'Sentiment analysis completed successfully');
        })
        .catch(error => {
            console.error('Error analyzing sentiment:', error);
            placeholder.innerHTML = `
                <i class="bi bi-exclamation-triangle display-1 text-warning"></i>
                <p class="text-muted">Error analyzing sentiment: ${error.message || 'Unknown error'}</p>
            `;
            showAlert('error', 'Failed to generate sentiment analysis');
        });
}

// Create sentiment data table rows
function createSentimentDataRows(data) {
    return data.map(item => `
        <tr>
            <td data-label="Date">${item.date}</td>
            <td data-label="Compound" class="${getSentimentClass(item.compound)}">${item.compound.toFixed(3)}</td>
            <td data-label="Positive">${item.positive.toFixed(3)}</td>
            <td data-label="Negative">${item.negative.toFixed(3)}</td>
            <td data-label="Neutral">${item.neutral.toFixed(3)}</td>
        </tr>
    `).join('');
}

// Create sentiment chart
function createSentimentChart(data) {
    const ctx = document.getElementById('sentiment-chart').getContext('2d');
    
    // Extract dates and sentiment values
    const dates = data.map(item => item.date);
    const sentiments = data.map(item => item.compound);
    
    // Create line chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Sentiment Score',
                data: sentiments,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            ...getChartOptions(),
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    ticks: {
                        stepSize: 0.5
                    }
                }
            }
        }
    });
}

// Format sentiment value with color
function formatSentimentValue(value) {
    const formattedValue = value.toFixed(3);
    const sentimentClass = getSentimentClass(value);
    return `<span class="${sentimentClass}">${formattedValue}</span>`;
}

// Get CSS class for sentiment value
function getSentimentClass(value) {
    if (value > 0.3) return 'text-success fw-bold';
    if (value > 0) return 'text-success';
    if (value < -0.3) return 'text-danger fw-bold';
    if (value < 0) return 'text-danger';
    return 'text-muted';
}

// Get alert class for sentiment value
function getSentimentAlertClass(value) {
    if (value > 0.3) return 'success';
    if (value > 0) return 'info';
    if (value < -0.3) return 'danger';
    if (value < 0) return 'warning';
    return 'secondary';
}

// Function to generate natural language trading summary
function generateTradingSummary(ticker, modelType = 'LSTM') {
    // Display loading state
    const comparisonResults = document.getElementById('comparison-results');
    comparisonResults.classList.remove('initially-hidden');
    comparisonResults.innerHTML = `
        <div class="d-flex flex-column align-items-center p-4">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted">Generating trading summary...</p>
        </div>
    `;
    
    document.getElementById('comparison-placeholder').classList.add('initially-hidden');
    
    // Fetch trading summary
    fetch(`/generate_summary/${ticker}?model_type=${modelType}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display the summary
            comparisonResults.innerHTML = `
                <div class="card mb-3">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">AI-Generated Trading Summary</h5>
                    </div>
                    <div class="card-body">
                        <p class="mb-0">${data.summary.replace(/\n/g, '<br>')}</p>
                    </div>
                    <div class="card-footer">
                        <small class="text-muted">
                            Generated on ${new Date().toLocaleString()} using natural language generation
                        </small>
                    </div>
                </div>
                
                <div class="d-grid gap-2">
                    <button id="generate-summary-pdf-btn" class="btn btn-outline-primary">
                        <i class="bi bi-file-earmark-pdf"></i> Generate PDF
                    </button>
                </div>
            `;
            
            // Add event listener for PDF generation
            document.getElementById('generate-summary-pdf-btn').addEventListener('click', () => {
                generateSummaryPDF(ticker, data.summary);
            });
            
            showAlert('success', 'Trading summary generated successfully');
        })
        .catch(error => {
            console.error('Error generating trading summary:', error);
            comparisonResults.innerHTML = `
                <div class="alert alert-danger">
                    <h6 class="alert-heading">Error</h6>
                    <p>Failed to generate trading summary: ${error.message || 'Unknown error'}</p>
                </div>
            `;
            showAlert('error', 'Failed to generate trading summary');
        });
}

// Function to generate a PDF from the trading summary
function generateSummaryPDF(ticker, summary) {
    // In a real implementation, this would call an API to generate the PDF
    // For now, we'll just show a success message
    showAlert('info', 'PDF generation would happen here in a real implementation');
} 