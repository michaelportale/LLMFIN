// Dashboard JavaScript

// Chart instances
let performanceChart = null;
let trainingChart = null;
let modelComparisonChart = null;
let algorithmChart = null;
let assetAllocationChart = null;
let riskAssessmentChart = null;
let historicalReturnsChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initDashboard();
    initThemeToggle();
    initWidgetControls();
    initSortable();
    initCharts();
    loadDashboardData();
    
    // Initialize period selectors
    document.querySelectorAll('.chart-period-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const parent = this.closest('.chart-period-selector');
            parent.querySelectorAll('.chart-period-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            const period = this.getAttribute('data-period');
            const chartId = this.closest('.dashboard-card').getAttribute('data-widget-id');
            updateChartPeriod(chartId, period);
        });
    });
    
    // Add widget button
    document.getElementById('add-widget-btn').addEventListener('click', function() {
        const modal = new bootstrap.Modal(document.getElementById('add-widget-modal'));
        modal.show();
    });
    
    // Add widget from modal
    document.querySelectorAll('[data-widget-type]').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const widgetType = this.getAttribute('data-widget-type');
            addWidget(widgetType);
            bootstrap.Modal.getInstance(document.getElementById('add-widget-modal')).hide();
        });
    });
    
    // Save layout button
    document.getElementById('save-layout-btn').addEventListener('click', saveLayout);
    
    // Setup mobile-specific controls
    setupTouchControls();
    
    // Update when window resizes
    window.addEventListener('resize', () => {
        setupTouchControls();
    });
    
    // Initialize drag and drop for the layout
    initDragAndDrop();
});

// Initialize dashboard
function initDashboard() {
    console.log('Initializing dashboard...');
    loadSavedLayout();
}

// Initialize theme toggle
function initThemeToggle() {
    const themeSwitch = document.getElementById('theme-switch');
    
    // Check if user has a theme preference in localStorage
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    // Set the toggle state based on current theme
    if (currentTheme === 'dark') {
        themeSwitch.checked = true;
    }
    
    // Add event listener for theme switch
    themeSwitch.addEventListener('change', function() {
        if (this.checked) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
        }
        
        // Redraw charts with new theme
        updateChartsTheme();
    });
}

// Initialize widget controls
function initWidgetControls() {
    document.querySelectorAll('.widget-control-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            const widgetId = this.closest('.dashboard-card').getAttribute('data-widget-id');
            
            switch(action) {
                case 'refresh':
                    refreshWidget(widgetId);
                    break;
                case 'edit':
                    editWidget(widgetId);
                    break;
                case 'remove':
                    removeWidget(widgetId);
                    break;
            }
        });
    });
}

// Initialize sortable grid
function initSortable() {
    if (typeof Sortable !== 'undefined') {
        new Sortable(document.getElementById('dashboard-grid'), {
            animation: 150,
            ghostClass: 'sortable-ghost',
            onEnd: function() {
                saveLayout();
            }
        });
    }
}

// Functions for widget actions
function refreshWidget(widgetId) {
    console.log(`Refreshing widget: ${widgetId}`);
    // Reload data for the specific widget
    loadWidgetData(widgetId);
}

function editWidget(widgetId) {
    console.log(`Editing widget: ${widgetId}`);
    // Implement widget editing logic
}

function removeWidget(widgetId) {
    if (confirm('Are you sure you want to remove this widget?')) {
        const widget = document.querySelector(`.dashboard-card[data-widget-id="${widgetId}"]`);
        if (widget) {
            widget.remove();
            saveLayout();
        }
    }
}

function addWidget(widgetType) {
    console.log(`Adding widget of type: ${widgetType}`);
    
    // Generate unique ID for the new widget
    const widgetId = `${widgetType}-${Date.now()}`;
    
    // Create widget HTML based on type
    let widgetHTML = '';
    let widgetSize = 'widget-md';
    
    switch(widgetType) {
        case 'performance':
            widgetHTML = createPerformanceWidget(widgetId);
            widgetSize = 'widget-md';
            break;
        case 'metrics':
            widgetHTML = createMetricsWidget(widgetId);
            widgetSize = 'widget-sm';
            break;
        case 'training':
            widgetHTML = createTrainingWidget(widgetId);
            widgetSize = 'widget-md';
            break;
        case 'comparison':
            widgetHTML = createComparisonWidget(widgetId);
            widgetSize = 'widget-lg';
            break;
    }
    
    // Add the widget to the dashboard
    const dashboardGrid = document.getElementById('dashboard-grid');
    
    // Create wrapper element
    const widgetElement = document.createElement('div');
    widgetElement.className = `dashboard-card ${widgetSize}`;
    widgetElement.setAttribute('data-widget-id', widgetId);
    widgetElement.innerHTML = widgetHTML;
    
    // Add to grid
    dashboardGrid.appendChild(widgetElement);
    
    // Initialize controls for the new widget
    initWidgetControls();
    
    // Initialize charts if needed
    if (widgetType === 'performance' || widgetType === 'training' || widgetType === 'comparison') {
        initWidgetChart(widgetId);
    }
    
    // Save the new layout
    saveLayout();
}

// Save dashboard layout
function saveLayout() {
    const dashboardGrid = document.getElementById('dashboard-grid');
    const widgets = dashboardGrid.querySelectorAll('.dashboard-card');
    
    const layout = Array.from(widgets).map(widget => {
        return {
            id: widget.getAttribute('data-widget-id'),
            type: widget.getAttribute('data-widget-id').split('-')[0],
            size: Array.from(widget.classList).find(cls => cls.startsWith('widget-'))
        };
    });
    
    localStorage.setItem('dashboardLayout', JSON.stringify(layout));
    console.log('Layout saved:', layout);
}

// Load saved layout
function loadSavedLayout() {
    const savedLayout = localStorage.getItem('dashboardLayout');
    if (savedLayout) {
        try {
            const layout = JSON.parse(savedLayout);
            console.log('Loading saved layout:', layout);
            // Implementation for restoring layout would go here
        } catch (e) {
            console.error('Error loading saved layout:', e);
        }
    }
}

// Create widget HTML templates
function createPerformanceWidget(widgetId) {
    return `
        <div class="dashboard-card-header">
            <span>Performance Overview</span>
            <div class="widget-controls">
                <span class="widget-control-btn" data-action="refresh"><i class="bi bi-arrow-clockwise"></i></span>
                <span class="widget-control-btn" data-action="edit"><i class="bi bi-gear"></i></span>
                <span class="widget-control-btn" data-action="remove"><i class="bi bi-x"></i></span>
            </div>
        </div>
        <div class="dashboard-card-body">
            <div class="dashboard-chart" id="${widgetId}-chart"></div>
        </div>
    `;
}

function createMetricsWidget(widgetId) {
    return `
        <div class="dashboard-card-header">
            <span>Performance Metrics</span>
            <div class="widget-controls">
                <span class="widget-control-btn" data-action="refresh"><i class="bi bi-arrow-clockwise"></i></span>
                <span class="widget-control-btn" data-action="edit"><i class="bi bi-gear"></i></span>
                <span class="widget-control-btn" data-action="remove"><i class="bi bi-x"></i></span>
            </div>
        </div>
        <div class="dashboard-card-body">
            <div class="d-flex align-items-center">
                <div class="metric-icon">
                    <i class="bi bi-lightning"></i>
                </div>
                <div>
                    <div class="metric-value" id="${widgetId}-value">0%</div>
                    <div class="metric-label">Return Rate</div>
                    <div class="metric-change metric-change-up">
                        <i class="bi bi-arrow-up"></i> <span id="${widgetId}-change">0%</span> since last period
                    </div>
                </div>
            </div>
        </div>
    `;
}

function createTrainingWidget(widgetId) {
    return `
        <div class="dashboard-card-header">
            <span>Training Progress</span>
            <div class="realtime-indicator"></div>
            <div class="widget-controls">
                <span class="widget-control-btn" data-action="refresh"><i class="bi bi-arrow-clockwise"></i></span>
                <span class="widget-control-btn" data-action="edit"><i class="bi bi-gear"></i></span>
                <span class="widget-control-btn" data-action="remove"><i class="bi bi-x"></i></span>
            </div>
        </div>
        <div class="dashboard-card-body">
            <div class="dashboard-chart" id="${widgetId}-chart"></div>
        </div>
    `;
}

function createComparisonWidget(widgetId) {
    return `
        <div class="dashboard-card-header">
            <span>Model Comparison</span>
            <div class="widget-controls">
                <span class="widget-control-btn" data-action="refresh"><i class="bi bi-arrow-clockwise"></i></span>
                <span class="widget-control-btn" data-action="edit"><i class="bi bi-gear"></i></span>
                <span class="widget-control-btn" data-action="remove"><i class="bi bi-x"></i></span>
            </div>
        </div>
        <div class="dashboard-card-body">
            <div class="chart-controls">
                <div class="chart-period-selector">
                    <span class="chart-period-btn active" data-period="1m">1M</span>
                    <span class="chart-period-btn" data-period="3m">3M</span>
                    <span class="chart-period-btn" data-period="6m">6M</span>
                    <span class="chart-period-btn" data-period="1y">1Y</span>
                    <span class="chart-period-btn" data-period="all">All</span>
                </div>
            </div>
            <div class="dashboard-chart" id="${widgetId}-chart"></div>
        </div>
    `;
}

// Initialize charts
function initCharts() {
    // Performance chart
    initPerformanceChart();
    // Training progress chart
    initTrainingProgressChart();
    // Model comparison chart
    initModelComparisonChart();
    // Algorithm performance chart
    initAlgorithmPerformanceChart();
    // Asset Allocation chart
    initAssetAllocationChart();
    // Risk Assessment chart
    initRiskAssessmentChart();
    // Historical Returns chart
    initHistoricalReturnsChart();
}

// Load dashboard data
function loadDashboardData() {
    fetch('/api/dashboard/data')
        .then(response => response.json())
        .then(data => {
            updateDashboardWidgets(data);
            updateCharts(data);
            startRealTimeUpdates();
        })
        .catch(error => console.error('Error loading dashboard data:', error));
}

// Update dashboard widgets with data
function updateDashboardWidgets(data) {
    // Update metrics
    if (data.metrics) {
        document.getElementById('total-models-value').textContent = data.metrics.totalModels || 0;
        document.getElementById('best-model-value').textContent = data.metrics.bestModel?.name || 'None';
        
        // Update current training info if available
        if (data.metrics.currentTraining) {
            document.getElementById('current-training-value').textContent = data.metrics.currentTraining.name;
            document.getElementById('training-progress-text').textContent = 
                `Epoch ${data.metrics.currentTraining.currentEpoch}/${data.metrics.currentTraining.totalEpochs}`;
            
            // Add active class to indicator
            document.querySelector('.realtime-indicator')?.classList.add('active');
        }
    }
}

// Update all charts with new data
function updateCharts(data) {
    if (data.charts) {
        updatePerformanceChart(data.charts.performance);
        updateTrainingProgressChart(data.charts.training);
        updateModelComparisonChart(data.charts.comparison);
        updateAlgorithmPerformanceChart(data.charts.algorithms);
    }
}

// Start real-time updates for training progress
function startRealTimeUpdates() {
    // Use WebSocket for real-time updates if a training is in progress
    const socket = new WebSocket(`ws://${window.location.host}/ws/training`);
    
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'training_update') {
            updateTrainingProgressChart(data.training);
            updateTrainingMetrics(data.training);
        }
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        // Fallback to polling if WebSocket fails
        startPollingUpdates();
    };
}

// Fallback polling method for updates
function startPollingUpdates() {
    setInterval(() => {
        fetch('/api/training/progress')
            .then(response => response.json())
            .then(data => {
                if (data.active) {
                    updateTrainingProgressChart(data);
                    updateTrainingMetrics(data);
                }
            })
            .catch(error => console.error('Error polling updates:', error));
    }, 5000); // Poll every 5 seconds
}

// Update training metrics display
function updateTrainingMetrics(trainingData) {
    const currentTrainingValue = document.getElementById('current-training-value');
    const trainingProgressText = document.getElementById('training-progress-text');
    
    if (trainingData.active) {
        currentTrainingValue.textContent = trainingData.model_name;
        trainingProgressText.textContent = `Epoch ${trainingData.current_epoch}/${trainingData.total_epochs}`;
        trainingProgressText.classList.add('metric-change-active');
        
        // Update realtime indicator
        document.querySelector('.realtime-indicator')?.classList.add('active');
    } else {
        currentTrainingValue.textContent = 'None';
        trainingProgressText.textContent = 'No active training';
        trainingProgressText.classList.remove('metric-change-active');
        
        // Update realtime indicator
        document.querySelector('.realtime-indicator')?.classList.remove('active');
    }
}

// Check if device is mobile
const isMobile = () => window.innerWidth < 768;

// Configure chart options for different devices
function getResponsiveChartOptions(chartType) {
    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: isMobile() ? 'nearest' : 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: !isMobile() || chartType === 'radar',
                position: 'top',
                labels: {
                    boxWidth: isMobile() ? 10 : 20,
                    font: {
                        size: isMobile() ? 10 : 12
                    }
                }
            },
            tooltip: {
                enabled: true
            }
        },
        scales: {
            x: {
                grid: {
                    display: !isMobile()
                },
                ticks: {
                    maxRotation: isMobile() ? 45 : 0,
                    font: {
                        size: isMobile() ? 10 : 12
                    }
                }
            },
            y: {
                beginAtZero: true,
                ticks: {
                    font: {
                        size: isMobile() ? 10 : 12
                    }
                }
            }
        }
    };
    
    // Add chart-specific options
    if (chartType === 'radar') {
        baseOptions.scales = {}; // No scales for radar chart
        baseOptions.elements = {
            line: {
                borderWidth: isMobile() ? 2 : 3
            },
            point: {
                radius: isMobile() ? 2 : 3
            }
        };
    } else if (chartType === 'bar') {
        baseOptions.scales.x.stacked = false;
        baseOptions.scales.y.stacked = false;
    }
    
    return baseOptions;
}

// Update chart options on window resize
window.addEventListener('resize', () => {
    updateChartsForMobile();
});

// Update all charts for mobile
function updateChartsForMobile() {
    // No need to manually update charts as they're responsive by default
    // Just update theme if needed
    updateChartsTheme();
}

// Performance chart initialization
function initPerformanceChart() {
    const ctx = document.getElementById('performance-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('line');
    options.plugins.zoom = {
        zoom: {
            wheel: {
                enabled: true,
            },
            pinch: {
                enabled: true
            },
            mode: 'xy',
        },
        pan: {
            enabled: true,
            mode: 'xy',
        }
    };
    
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Model Performance',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.1,
                fill: false
            }]
        },
        options: options
    });
}

// Training progress chart initialization
function initTrainingProgressChart() {
    const ctx = document.getElementById('training-progress-chart')?.getContext('2d');
    if (!ctx) return;
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                zoom: {
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy',
                    },
                    pan: {
                        enabled: true,
                        mode: 'xy',
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            animation: {
                duration: 500
            }
        }
    });
}

// Model comparison chart initialization
function initModelComparisonChart() {
    const ctx = document.getElementById('model-comparison-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('radar');
    
    modelComparisonChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Profit', 'Efficiency', 'Risk', 'Speed', 'Stability'],
            datasets: [{
                label: 'Model A',
                data: [85, 75, 90, 60, 95, 80],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(255, 99, 132, 1)'
            }, {
                label: 'Model B',
                data: [70, 90, 80, 85, 75, 95],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
            }]
        },
        options: options
    });
}

// Algorithm performance chart initialization
function initAlgorithmPerformanceChart() {
    const ctx = document.getElementById('algorithm-performance-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('line');
    
    algorithmChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Algorithm 1',
                data: [12, 19, 13, 15, 22, 27],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                fill: false
            }, {
                label: 'Algorithm 2',
                data: [15, 12, 17, 13, 28, 23],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.1,
                fill: false
            }]
        },
        options: options
    });
}

// Asset Allocation chart initialization
function initAssetAllocationChart() {
    const ctx = document.getElementById('asset-allocation-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('pie');
    
    assetAllocationChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Stocks', 'Bonds', 'Cash', 'Commodities'],
            datasets: [{
                data: [40, 30, 20, 10],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ],
                borderWidth: 1
            }]
        },
        options: options
    });
}

// Risk Assessment chart initialization
function initRiskAssessmentChart() {
    const ctx = document.getElementById('risk-assessment-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('radar');
    
    riskAssessmentChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Volatility', 'Drawdown', 'VaR', 'Beta', 'Sharpe Ratio'],
            datasets: [{
                label: 'Current',
                data: [65, 75, 70, 80, 60],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(54, 162, 235, 1)'
            }, {
                label: 'Target',
                data: [70, 65, 60, 70, 80],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(255, 99, 132, 1)'
            }]
        },
        options: options
    });
}

// Historical Returns chart initialization
function initHistoricalReturnsChart() {
    const ctx = document.getElementById('historical-returns-chart')?.getContext('2d');
    if (!ctx) return;
    
    const options = getResponsiveChartOptions('bar');
    
    historicalReturnsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Monthly Returns (%)',
                data: [2.5, -1.2, 3.1, 1.8, -0.5, 2.2],
                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: options
    });
}

// Update charts when theme changes
function updateChartsTheme() {
    const isDarkMode = document.body.classList.contains('dark-mode');
    const textColor = isDarkMode ? '#fff' : '#666';
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    Chart.defaults.color = textColor;
    
    // Update each chart if they exist
    const updateChartTheme = (chart) => {
        if (!chart) return;
        
        // Update scales if the chart has them
        if (chart.options.scales) {
            for (const axisKey in chart.options.scales) {
                const axis = chart.options.scales[axisKey];
                if (axis.grid) {
                    axis.grid.color = gridColor;
                }
                if (axis.ticks) {
                    axis.ticks.color = textColor;
                }
            }
        }
        
        // Update legend
        if (chart.options.plugins && chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = textColor;
        }
        
        // Update tooltip
        if (chart.options.plugins && chart.options.plugins.tooltip) {
            chart.options.plugins.tooltip.backgroundColor = isDarkMode ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.7)';
            chart.options.plugins.tooltip.titleColor = isDarkMode ? '#fff' : '#000';
            chart.options.plugins.tooltip.bodyColor = isDarkMode ? '#fff' : '#000';
            chart.options.plugins.tooltip.borderColor = isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)';
        }
        
        // Apply responsive options based on current device
        const chartType = chart.config._config.type;
        const mobileOptions = getResponsiveChartOptions(chartType);
        
        // Merge mobile-specific options
        if (isMobile()) {
            if (mobileOptions.plugins && mobileOptions.plugins.legend) {
                chart.options.plugins.legend = {
                    ...chart.options.plugins.legend,
                    ...mobileOptions.plugins.legend
                };
            }
            
            if (mobileOptions.scales) {
                for (const scaleKey in mobileOptions.scales) {
                    if (chart.options.scales[scaleKey]) {
                        chart.options.scales[scaleKey] = {
                            ...chart.options.scales[scaleKey],
                            ...mobileOptions.scales[scaleKey]
                        };
                    }
                }
            }
        }
        
        chart.update();
    };
    
    // Update all charts
    updateChartTheme(performanceChart);
    updateChartTheme(modelComparisonChart);
    updateChartTheme(algorithmChart);
    updateChartTheme(assetAllocationChart);
    updateChartTheme(riskAssessmentChart);
    updateChartTheme(historicalReturnsChart);
}

// Update chart data based on period selection
function updateChartPeriod(chartId, period) {
    console.log(`Updating ${chartId} chart to period: ${period}`);
    
    fetch(`/api/chart-data/${chartId}?period=${period}`)
        .then(response => response.json())
        .then(data => {
            switch(chartId) {
                case 'model-comparison':
                    updateModelComparisonChart(data);
                    break;
                case 'performance-overview':
                    updatePerformanceChart(data);
                    break;
                // Add other cases as needed
            }
        })
        .catch(error => console.error(`Error updating chart period for ${chartId}:`, error));
}

// Load data for a specific widget
function loadWidgetData(widgetId) {
    fetch(`/api/widget-data/${widgetId}`)
        .then(response => response.json())
        .then(data => {
            // Update widget based on type
            const widgetType = widgetId.split('-')[0];
            
            switch(widgetType) {
                case 'performance':
                    updatePerformanceChart(data);
                    break;
                case 'training':
                    updateTrainingProgressChart(data);
                    break;
                case 'metrics':
                    updateMetricsWidget(widgetId, data);
                    break;
                case 'comparison':
                    updateModelComparisonChart(data);
                    break;
            }
        })
        .catch(error => console.error(`Error loading data for widget ${widgetId}:`, error));
}

// Update performance chart with new data
function updatePerformanceChart(data) {
    if (!performanceChart || !data) return;
    
    performanceChart.data.labels = data.labels;
    performanceChart.data.datasets[0].data = data.values;
    performanceChart.update();
}

// Update training progress chart with new data
function updateTrainingProgressChart(data) {
    if (!trainingChart || !data) return;
    
    // Update training chart with real-time data
    trainingChart.data.labels = data.epochs || [];
    trainingChart.data.datasets[0].data = data.training_loss || [];
    trainingChart.data.datasets[1].data = data.validation_loss || [];
    trainingChart.update();
}

// Update model comparison chart with new data
function updateModelComparisonChart(data) {
    if (!modelComparisonChart || !data) return;
    
    modelComparisonChart.data.labels = data.models || [];
    modelComparisonChart.data.datasets[0].data = data.returns || [];
    
    // Generate colors based on values
    const colors = data.returns.map(value => {
        // Use green for positive, red for negative returns
        return value >= 0 
            ? `rgba(75, 192, 192, ${0.6 + (value/100) * 0.4})` 
            : `rgba(255, 99, 132, ${0.6 + (Math.abs(value)/100) * 0.4})`;
    });
    
    modelComparisonChart.data.datasets[0].backgroundColor = colors;
    modelComparisonChart.data.datasets[0].borderColor = colors.map(c => c.replace(/[\d.]+\)$/, '1)'));
    modelComparisonChart.update();
}

// Update algorithm performance chart with new data
function updateAlgorithmPerformanceChart(data) {
    if (!algorithmChart || !data || !data.algorithms) return;
    
    // Clear existing datasets
    algorithmChart.data.datasets = [];
    
    // Add new datasets for each algorithm
    data.algorithms.forEach((algo, index) => {
        const colors = [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)'
        ];
        
        algorithmChart.data.datasets.push({
            label: algo.name,
            data: algo.metrics,
            backgroundColor: colors[index % colors.length].replace('0.7', '0.2'),
            borderColor: colors[index % colors.length],
            pointBackgroundColor: colors[index % colors.length],
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[index % colors.length]
        });
    });
    
    algorithmChart.update();
}

// Update metrics widget with data
function updateMetricsWidget(widgetId, data) {
    const valueEl = document.getElementById(`${widgetId}-value`);
    const changeEl = document.getElementById(`${widgetId}-change`);
    
    if (valueEl) valueEl.textContent = data.value || '0%';
    if (changeEl) {
        changeEl.textContent = data.change || '0%';
        
        // Update change indicator
        const changeContainer = changeEl.closest('.metric-change');
        if (changeContainer) {
            if (parseFloat(data.change) >= 0) {
                changeContainer.classList.remove('metric-change-down');
                changeContainer.classList.add('metric-change-up');
                changeContainer.querySelector('i').className = 'bi bi-arrow-up';
            } else {
                changeContainer.classList.remove('metric-change-up');
                changeContainer.classList.add('metric-change-down');
                changeContainer.querySelector('i').className = 'bi bi-arrow-down';
            }
        }
    }
}

// Initialize chart for a new widget
function initWidgetChart(widgetId) {
    const chartContainer = document.getElementById(`${widgetId}-chart`);
    if (!chartContainer) return;
    
    const ctx = chartContainer.getContext('2d');
    let widgetChart;
    
    // Determine chart type based on widget ID
    if (widgetId.includes('performance')) {
        widgetChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Performance',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            mode: 'xy',
                        },
                        pan: {
                            enabled: true,
                        }
                    }
                }
            }
        });
    } else if (widgetId.includes('training')) {
        widgetChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            mode: 'xy',
                        }
                    }
                },
                animation: {
                    duration: 500
                }
            }
        });
    } else if (widgetId.includes('comparison')) {
        widgetChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Comparison',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Load data for the new widget
    loadWidgetData(widgetId);
}

// Add touch-friendly controls for mobile users
function setupTouchControls() {
    if (!isMobile()) return;
    
    // Add swipe functionality to tabs
    const tabs = document.querySelectorAll('.nav-link');
    let tabsContainer = document.querySelector('.nav-tabs');
    
    if (tabsContainer) {
        let startX, startY;
        let threshold = 100; // Minimum distance to be considered a swipe
        
        tabsContainer.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        }, false);
        
        tabsContainer.addEventListener('touchend', (e) => {
            if (!startX || !startY) return;
            
            let endX = e.changedTouches[0].clientX;
            let endY = e.changedTouches[0].clientY;
            
            let diffX = startX - endX;
            let diffY = startY - endY;
            
            // Only horizontal swipe if the vertical movement is less than half of horizontal
            if (Math.abs(diffX) > threshold && Math.abs(diffY) < Math.abs(diffX) / 2) {
                let activeTab = document.querySelector('.nav-link.active');
                let activeIndex = Array.from(tabs).indexOf(activeTab);
                let targetIndex;
                
                if (diffX > 0) { // Swipe left
                    targetIndex = Math.min(activeIndex + 1, tabs.length - 1);
                } else { // Swipe right
                    targetIndex = Math.max(activeIndex - 1, 0);
                }
                
                if (targetIndex !== activeIndex && tabs[targetIndex]) {
                    tabs[targetIndex].click();
                }
            }
            
            startX = startY = null;
        }, false);
    }
    
    // Make buttons larger on mobile
    const buttons = document.querySelectorAll('button, .btn');
    buttons.forEach(button => {
        if (!button.classList.contains('mobile-friendly')) {
            button.classList.add('mobile-friendly');
            // Increase touch target size
            button.style.minHeight = '44px';
            button.style.minWidth = '44px';
        }
    });
    
    // Add pinch-to-zoom for charts (using the existing zoom plugin)
    const chartCanvases = document.querySelectorAll('canvas');
    chartCanvases.forEach(canvas => {
        if (canvas.chart && canvas.chart.options.plugins && canvas.chart.options.plugins.zoom) {
            canvas.chart.options.plugins.zoom.pan.enabled = true;
            canvas.chart.options.plugins.zoom.pan.mode = 'xy';
            canvas.chart.options.plugins.zoom.zoom.pinch.enabled = true;
            canvas.chart.update();
        }
    });
    
    // Make dropdowns easier to use on mobile
    const selects = document.querySelectorAll('select');
    selects.forEach(select => {
        select.style.fontSize = '16px'; // Prevent iOS zoom on focus
        select.style.height = '44px';   // Larger touch target
    });
    
    // Improve form inputs for mobile
    const inputs = document.querySelectorAll('input[type="text"], input[type="number"], input[type="email"]');
    inputs.forEach(input => {
        input.style.fontSize = '16px'; // Prevent iOS zoom on focus
        input.style.height = '44px';   // Larger touch target
        input.style.padding = '10px';  // More padding for easier touch
    });
}

// Initialize drag and drop for the layout
function initDragAndDrop() {
    const dashboard = document.querySelector('.dashboard');
    if (!dashboard) return;
    
    let draggedItem = null;
    
    // Disable dragging on mobile and use a different approach
    if (isMobile()) {
        // Add reorder buttons to each card header
        const cards = document.querySelectorAll('.dashboard-card');
        cards.forEach(card => {
            const header = card.querySelector('.dashboard-card-header');
            if (header && !header.querySelector('.mobile-ordering-buttons')) {
                const orderingButtons = document.createElement('div');
                orderingButtons.className = 'mobile-ordering-buttons';
                orderingButtons.innerHTML = `
                    <button class="btn btn-sm btn-outline-primary move-up" aria-label="Move widget up">
                        <i class="fas fa-arrow-up"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-primary move-down" aria-label="Move widget down">
                        <i class="fas fa-arrow-down"></i>
                    </button>
                `;
                header.appendChild(orderingButtons);
                
                // Add event listeners for the buttons
                const moveUpBtn = orderingButtons.querySelector('.move-up');
                const moveDownBtn = orderingButtons.querySelector('.move-down');
                
                moveUpBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const currentWidget = card;
                    const previousWidget = currentWidget.previousElementSibling;
                    
                    if (previousWidget && previousWidget.classList.contains('dashboard-card')) {
                        dashboard.insertBefore(currentWidget, previousWidget);
                        saveLayout(); // Save the new layout
                    }
                });
                
                moveDownBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const currentWidget = card;
                    const nextWidget = currentWidget.nextElementSibling;
                    
                    if (nextWidget && nextWidget.classList.contains('dashboard-card')) {
                        dashboard.insertBefore(nextWidget, currentWidget);
                        saveLayout(); // Save the new layout
                    }
                });
            }
        });
        
        // Return early since we don't need drag and drop on mobile
        return;
    }
    
    // Standard drag and drop for desktop
    const cards = document.querySelectorAll('.dashboard-card');
    cards.forEach(card => {
        const header = card.querySelector('.dashboard-card-header');
        
        // Make sure we don't add duplicate listeners
        if (header && !header.dataset.dragInitialized) {
            header.dataset.dragInitialized = 'true';
            
            header.addEventListener('mousedown', (e) => {
                if (e.target.closest('.dashboard-card-controls')) return;
                
                draggedItem = card;
                card.classList.add('dragging');
                
                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
            });
        }
    });
    
    function handleMouseMove(e) {
        if (!draggedItem) return;
        
        const dashboardRect = dashboard.getBoundingClientRect();
        const mouseY = e.clientY - dashboardRect.top;
        
        // Find the closest card to the current mouse position
        let closestCard = null;
        let closestDistance = Infinity;
        
        cards.forEach(card => {
            if (card === draggedItem) return;
            
            const cardRect = card.getBoundingClientRect();
            const cardMiddleY = (cardRect.top + cardRect.bottom) / 2 - dashboardRect.top;
            const distance = Math.abs(mouseY - cardMiddleY);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestCard = card;
            }
        });
        
        if (closestCard) {
            const draggedRect = draggedItem.getBoundingClientRect();
            const closestRect = closestCard.getBoundingClientRect();
            
            if (draggedRect.top < closestRect.top) {
                dashboard.insertBefore(draggedItem, closestCard.nextSibling);
            } else {
                dashboard.insertBefore(draggedItem, closestCard);
            }
        }
    }
    
    function handleMouseUp() {
        if (draggedItem) {
            draggedItem.classList.remove('dragging');
            draggedItem = null;
            
            // Save the new layout
            saveLayout();
        }
        
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
} 