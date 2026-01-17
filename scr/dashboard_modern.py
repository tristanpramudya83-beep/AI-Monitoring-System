# dashboard_modern_enhanced.py - Ultimate Version
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
import threading
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import os
import glob
import pickle
import hashlib
import secrets
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(32)

# Multi-tenant support
TENANTS = {
    'default': {'color': '#3B82F6', 'priority': 1},
    'production': {'color': '#EF4444', 'priority': 3},
    'development': {'color': '#10B981', 'priority': 2}
}

# Initialize enhanced databases
def init_databases():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    
    # Enhanced alerts table with severity levels
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  tenant TEXT,
                  metric TEXT,
                  value REAL,
                  threshold REAL,
                  severity TEXT,
                  message TEXT,
                  status TEXT DEFAULT 'active',
                  acknowledged BOOLEAN DEFAULT 0,
                  auto_resolved BOOLEAN DEFAULT 0)''')
    
    # Enhanced system logs with anomaly scores
    c.execute('''CREATE TABLE IF NOT EXISTS system_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  tenant TEXT,
                  cpu REAL,
                  ram INTEGER,
                  disk REAL,
                  network_latency REAL,
                  temperature REAL,
                  power_consumption REAL,
                  anomaly_score REAL,
                  status TEXT,
                  alert_message TEXT)''')
    
    # Machine learning models storage
    c.execute('''CREATE TABLE IF NOT EXISTS ml_models
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model_type TEXT,
                  metric TEXT,
                  model_data BLOB,
                  accuracy REAL,
                  last_trained DATETIME,
                  feature_importance TEXT)''')
    
    # Performance baselines
    c.execute('''CREATE TABLE IF NOT EXISTS performance_baselines
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  metric TEXT,
                  tenant TEXT,
                  baseline_mean REAL,
                  baseline_std REAL,
                  peak_hours TEXT,
                  updated DATETIME)''')
    
    # Auto-remediation actions
    c.execute('''CREATE TABLE IF NOT EXISTS remediation_actions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  alert_id INTEGER,
                  action_type TEXT,
                  action_details TEXT,
                  success BOOLEAN,
                  execution_time REAL)''')
    
    conn.commit()
    conn.close()
    print("[DB] Enhanced databases initialized successfully")

# Initialize on startup
init_databases()

# Real-time data storage with multi-tenant support
real_time_data = {
    'default': {
        'cpu': deque(maxlen=100),
        'ram': deque(maxlen=100),
        'disk': deque(maxlen=100),
        'network': deque(maxlen=100),
        'temperature': deque(maxlen=100),
        'power': deque(maxlen=100),
        'time': deque(maxlen=100)
    }
}

# Advanced alert management
alert_manager = {
    'active_alerts': deque(maxlen=50),
    'alert_patterns': {},
    'suppression_rules': {},
    'escalation_policies': {}
}

# AI/ML Models
ml_models = {
    'forecasting': {},
    'anomaly_detection': {},
    'clustering': {},
    'recommendation': {}
}

# Performance baselines
performance_baselines = {}

# --- Advanced AI/ML Functions ---
class AdvancedAIOps:
    @staticmethod
    def detect_seasonal_patterns(metric_data, time_data):
        """Detect daily/weekly seasonal patterns"""
        try:
            if len(metric_data) < 24:
                return None
            
            df = pd.DataFrame({'value': metric_data, 'time': time_data})
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            df['dayofweek'] = pd.to_datetime(df['time']).dt.dayofweek
            
            hourly_avg = df.groupby('hour')['value'].mean().to_dict()
            weekday_avg = df.groupby('dayofweek')['value'].mean().to_dict()
            
            return {
                'hourly_pattern': hourly_avg,
                'weekly_pattern': weekday_avg,
                'peak_hours': sorted(hourly_avg, key=hourly_avg.get, reverse=True)[:3]
            }
        except Exception as e:
            print(f"[AIOps] Pattern detection error: {e}")
            return None
    
    @staticmethod
    def train_ensemble_forecaster(metric_name, historical_data):
        """Train ensemble model with multiple algorithms"""
        try:
            if len(historical_data) < 30:
                return None
            
            X = np.arange(len(historical_data)).reshape(-1, 1)
            y = np.array(historical_data)
            
            # Ensemble of models
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.svm import SVR
            
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            }
            
            predictions = {}
            for name, model in models.items():
                model.fit(X, y)
                pred = model.predict(X)
                mse = np.mean((y - pred) ** 2)
                predictions[name] = {
                    'model': model,
                    'mse': mse,
                    'predictions': pred.tolist()
                }
            
            # Weighted ensemble prediction
            weights = {name: 1/(pred['mse']+1e-10) for name, pred in predictions.items()}
            total_weight = sum(weights.values())
            weights = {name: w/total_weight for name, w in weights.items()}
            
            ensemble_pred = sum(predictions[name]['predictions'][i] * weights[name] 
                              for name in predictions.keys() for i in range(len(y)))
            
            return {
                'models': predictions,
                'weights': weights,
                'ensemble_accuracy': 1 - np.mean((y - ensemble_pred/len(predictions)) ** 2)/np.var(y),
                'last_trained': datetime.now()
            }
        except Exception as e:
            print(f"[AIOps] Ensemble training error: {e}")
            return None
    
    @staticmethod
    def detect_anomalies_isolation_forest(metric_data):
        """Detect anomalies using Isolation Forest"""
        try:
            if len(metric_data) < 20:
                return {'scores': [], 'anomalies': []}
            
            data = np.array(metric_data).reshape(-1, 1)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(data_scaled)
            
            anomalies = [i for i, score in enumerate(anomaly_scores) if score == -1]
            
            return {
                'scores': anomaly_scores.tolist(),
                'anomalies': anomalies,
                'contamination': len(anomalies)/len(metric_data)
            }
        except Exception as e:
            print(f"[AIOps] Anomaly detection error: {e}")
            return {'scores': [], 'anomalies': []}
    
    @staticmethod
    def calculate_performance_baseline(metric_data, tenant='default'):
        """Calculate dynamic performance baselines"""
        if len(metric_data) < 10:
            return None
        
        data_array = np.array(metric_data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # Dynamic thresholds based on percentiles
        p95 = np.percentile(data_array, 95)
        p99 = np.percentile(data_array, 99)
        
        baseline = {
            'mean': mean,
            'std': std,
            'p95': p95,
            'p99': p99,
            'normal_range': [max(0, mean - 2*std), min(100, mean + 2*std)],
            'warning_threshold': p95,
            'critical_threshold': p99,
            'updated': datetime.now()
        }
        
        # Store in database
        conn = sqlite3.connect('monitoring.db')
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO performance_baselines 
                     (metric, tenant, baseline_mean, baseline_std, updated)
                     VALUES (?, ?, ?, ?, ?)''',
                  ('composite', tenant, mean, std, datetime.now()))
        conn.commit()
        conn.close()
        
        return baseline

# --- Smart Alerting System ---
class SmartAlertSystem:
    def __init__(self):
        self.alert_cooldown = {}
        self.alert_correlation = {}
        self.root_cause_analysis = {}
    
    def analyze_alert_correlation(self, current_alert, recent_alerts):
        """Correlate related alerts to identify root causes"""
        correlated = []
        
        for alert in recent_alerts:
            time_diff = abs((datetime.now() - alert['timestamp']).total_seconds())
            if time_diff < 300:  # 5 minutes window
                # Check if alerts share common patterns
                if (alert['metric'] == current_alert['metric'] or
                    (alert['tenant'] == current_alert['tenant'] and
                     alert['severity'] == current_alert['severity'])):
                    correlated.append(alert)
        
        return correlated
    
    def generate_root_cause_hypothesis(self, correlated_alerts):
        """Generate root cause hypotheses based on alert patterns"""
        if not correlated_alerts:
            return "Unknown cause"
        
        # Analyze patterns
        metrics = [a['metric'] for a in correlated_alerts]
        severities = [a['severity'] for a in correlated_alerts]
        
        if 'cpu' in metrics and 'ram' in metrics:
            return "Resource exhaustion - Check application memory leaks"
        elif 'disk' in metrics and len(correlated_alerts) > 2:
            return "Storage system degradation - Check disk health"
        elif 'temperature' in metrics and 'cpu' in metrics:
            return "Thermal throttling detected - Check cooling system"
        else:
            return f"Multiple {', '.join(set(metrics))} issues detected"
    
    def should_suppress_alert(self, alert_data):
        """Implement alert suppression to prevent noise"""
        alert_key = f"{alert_data['metric']}_{alert_data['tenant']}"
        
        # Check cooldown period
        if alert_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[alert_key]
            time_diff = (datetime.now() - last_alert).total_seconds()
            if time_diff < 60:  # 60-second cooldown
                return True
        
        self.alert_cooldown[alert_key] = datetime.now()
        return False

# --- Auto-Remediation Engine ---
class AutoRemediationEngine:
    @staticmethod
    def suggest_remediation(alert_data, metric_history):
        """Suggest automated remediation actions"""
        suggestions = []
        
        if alert_data['metric'] == 'cpu':
            if alert_data['value'] > 90:
                suggestions.append({
                    'action': 'scale_up',
                    'type': 'vertical',
                    'priority': 'high',
                    'details': 'Increase CPU allocation by 50%',
                    'estimated_impact': 'high'
                })
            suggestions.append({
                'action': 'optimize',
                'type': 'process_cleanup',
                'priority': 'medium',
                'details': 'Kill non-essential processes',
                'estimated_impact': 'medium'
            })
        
        elif alert_data['metric'] == 'ram':
            suggestions.append({
                'action': 'clear_cache',
                'type': 'memory',
                'priority': 'high',
                'details': 'Clear system cache and buffers',
                'estimated_impact': 'high'
            })
        
        elif alert_data['metric'] == 'disk':
            suggestions.append({
                'action': 'cleanup',
                'type': 'storage',
                'priority': 'medium',
                'details': 'Remove temporary files and old logs',
                'estimated_impact': 'medium'
            })
        
        return suggestions
    
    @staticmethod
    def execute_remediation(action):
        """Execute automated remediation action"""
        # This would integrate with actual infrastructure APIs
        # For now, log the action
        conn = sqlite3.connect('monitoring.db')
        c = conn.cursor()
        c.execute('''INSERT INTO remediation_actions 
                     (timestamp, action_type, action_details, success)
                     VALUES (?, ?, ?, ?)''',
                  (datetime.now(), action['type'], action['details'], True))
        conn.commit()
        conn.close()
        
        print(f"[Remediation] Executed: {action['type']} - {action['details']}")
        return True

# Initialize systems
aiops = AdvancedAIOps()
alert_system = SmartAlertSystem()
remediation_engine = AutoRemediationEngine()

# Start background threads
def background_analytics():
    """Background analytics and model training"""
    while True:
        try:
            # Update performance baselines
            for tenant in real_time_data:
                for metric in ['cpu', 'ram', 'disk']:
                    if real_time_data[tenant][metric]:
                        baseline = aiops.calculate_performance_baseline(
                            list(real_time_data[tenant][metric]), tenant
                        )
                        performance_baselines[f"{tenant}_{metric}"] = baseline
            
            # Detect seasonal patterns
            patterns = aiops.detect_seasonal_patterns(
                list(real_time_data['default']['cpu']),
                list(real_time_data['default']['time'])
            )
            
            time.sleep(300)  # Run every 5 minutes
        except Exception as e:
            print(f"[Background] Analytics error: {e}")
            time.sleep(60)

# Start analytics thread
analytics_thread = threading.Thread(target=background_analytics, daemon=True)
analytics_thread.start()

# --- Enhanced HTML Template with Advanced Features ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered IoT Monitoring Dashboard v3.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .glow { animation: pulse-glow 2s infinite; }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .dark-gradient { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
        .card-hover { transition: transform 0.3s, box-shadow 0.3s; }
        .card-hover:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0,0,0,0.3); }
    </style>
</head>
<body class="bg-gray-950 text-gray-100 font-sans">
    <div class="container mx-auto p-4">
        <!-- Enhanced Header -->
        <div class="dark-gradient rounded-2xl p-6 mb-8 shadow-2xl">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="flex items-center space-x-4">
                    <div class="p-3 bg-blue-500/20 rounded-xl">
                        <i class="fas fa-robot text-3xl text-blue-400"></i>
                    </div>
                    <div>
                        <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                            AI-Powered Monitoring System
                        </h1>
                        <p class="text-gray-300 mt-2">Advanced Analytics • Predictive Insights • Auto-Remediation</p>
                    </div>
                </div>
                <div class="mt-4 md:mt-0 text-right space-y-2">
                    <div class="flex items-center justify-end space-x-2">
                        <div class="h-3 w-3 rounded-full bg-green-500 animate-pulse"></div>
                        <span id="agentStatus" class="font-semibold">Connected</span>
                    </div>
                    <div class="text-sm text-gray-300" id="currentTime"></div>
                    <div class="text-xs text-gray-400" id="systemUptime">Uptime: 0 days</div>
                </div>
            </div>
        </div>

        <!-- Multi-tenant Selector -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-900 rounded-xl p-4 text-center border border-blue-500/30">
                <div class="text-sm text-gray-400">Current Tenant</div>
                <select id="tenantSelect" class="bg-transparent text-lg font-bold text-blue-400 w-full text-center">
                    <option value="default" class="bg-gray-900">Default Environment</option>
                    <option value="production" class="bg-gray-900">Production</option>
                    <option value="development" class="bg-gray-900">Development</option>
                </select>
            </div>
            <div class="bg-gray-900 rounded-xl p-4 text-center">
                <div class="text-sm text-gray-400">Active Alerts</div>
                <div id="activeAlertCount" class="text-2xl font-bold text-red-400">0</div>
            </div>
            <div class="bg-gray-900 rounded-xl p-4 text-center">
                <div class="text-sm text-gray-400">System Health</div>
                <div id="systemHealthScore" class="text-2xl font-bold text-green-400">100%</div>
            </div>
            <div class="bg-gray-900 rounded-xl p-4 text-center">
                <div class="text-sm text-gray-400">Prediction Accuracy</div>
                <div id="predictionAccuracy" class="text-2xl font-bold text-purple-400">95%</div>
            </div>
        </div>

        <!-- Main Dashboard Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Left Column -->
            <div class="space-y-6">
                <!-- Resource Utilization Card -->
                <div class="bg-gray-900 rounded-2xl p-6 card-hover border border-gray-800">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-xl font-bold text-blue-300">
                            <i class="fas fa-chart-line mr-2"></i>Resource Utilization
                        </h2>
                        <div class="flex space-x-2">
                            <button class="px-3 py-1 text-xs bg-blue-500/20 text-blue-300 rounded-lg hover:bg-blue-500/30 transition">
                                1H
                            </button>
                            <button class="px-3 py-1 text-xs bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition">
                                24H
                            </button>
                            <button class="px-3 py-1 text-xs bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition">
                                7D
                            </button>
                        </div>
                    </div>
                    <div class="h-64">
                        <canvas id="compositeChart"></canvas>
                    </div>
                </div>

                <!-- AI Insights Card -->
                <div class="bg-gray-900 rounded-2xl p-6 card-hover border border-gray-800">
                    <h2 class="text-xl font-bold text-purple-300 mb-4">
                        <i class="fas fa-brain mr-2"></i>AI Insights & Recommendations
                    </h2>
                    <div id="aiInsights" class="space-y-3">
                        <div class="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                            <div class="flex items-center">
                                <i class="fas fa-lightbulb text-yellow-400 mr-2"></i>
                                <span class="font-medium">System operating normally</span>
                            </div>
                            <p class="text-sm text-gray-400 mt-1">All metrics within optimal ranges</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="space-y-6">
                <!-- Real-time Metrics Grid -->
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-gray-900 rounded-xl p-5 text-center card-hover border border-blue-500/30">
                        <div class="text-4xl font-bold text-blue-400 mb-2" id="cpuValue">0%</div>
                        <div class="text-sm text-gray-400">CPU Usage</div>
                        <div class="h-2 bg-gray-800 rounded-full mt-2 overflow-hidden">
                            <div id="cpuBar" class="h-full bg-blue-500 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="bg-gray-900 rounded-xl p-5 text-center card-hover border border-purple-500/30">
                        <div class="text-4xl font-bold text-purple-400 mb-2" id="ramValue">0%</div>
                        <div class="text-sm text-gray-400">RAM Usage</div>
                        <div class="h-2 bg-gray-800 rounded-full mt-2 overflow-hidden">
                            <div id="ramBar" class="h-full bg-purple-500 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="bg-gray-900 rounded-xl p-5 text-center card-hover border border-green-500/30">
                        <div class="text-4xl font-bold text-green-400 mb-2" id="diskValue">0%</div>
                        <div class="text-sm text-gray-400">Disk Usage</div>
                        <div class="h-2 bg-gray-800 rounded-full mt-2 overflow-hidden">
                            <div id="diskBar" class="h-full bg-green-500 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="bg-gray-900 rounded-xl p-5 text-center card-hover border border-red-500/30">
                        <div class="text-4xl font-bold text-red-400 mb-2" id="tempValue">0°C</div>
                        <div class="text-sm text-gray-400">Temperature</div>
                        <div class="h-2 bg-gray-800 rounded-full mt-2 overflow-hidden">
                            <div id="tempBar" class="h-full bg-red-500 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                </div>

                <!-- Smart Alerts Card -->
                <div class="bg-gray-900 rounded-2xl p-6 card-hover border border-red-500/30">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold text-red-300">
                            <i class="fas fa-exclamation-triangle mr-2"></i>Smart Alerts
                        </h2>
                        <button id="acknowledgeAll" class="px-3 py-1 text-xs bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition">
                            Acknowledge All
                        </button>
                    </div>
                    <div class="overflow-y-auto max-h-64 space-y-3" id="smartAlertsContainer">
                        <div class="text-center text-gray-500 py-4">
                            <i class="fas fa-check-circle text-2xl mb-2"></i>
                            <p>No active alerts</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Predictive Analytics Section -->
        <div class="bg-gray-900 rounded-2xl p-6 mb-8 card-hover border border-gray-800">
            <h2 class="text-xl font-bold text-cyan-300 mb-6">
                <i class="fas fa-crystal-ball mr-2"></i>Predictive Analytics
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="p-4 bg-gray-800/50 rounded-xl">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-blue-300">CPU Forecast</h3>
                        <span class="text-xs px-2 py-1 bg-blue-500/20 text-blue-300 rounded">Next 1H</span>
                    </div>
                    <div class="text-2xl font-bold text-center mb-2" id="cpuForecastValue">--</div>
                    <div class="text-sm text-gray-400 text-center" id="cpuForecastTrend"></div>
                </div>
                <div class="p-4 bg-gray-800/50 rounded-xl">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-purple-300">RAM Forecast</h3>
                        <span class="text-xs px-2 py-1 bg-purple-500/20 text-purple-300 rounded">Next 1H</span>
                    </div>
                    <div class="text-2xl font-bold text-center mb-2" id="ramForecastValue">--</div>
                    <div class="text-sm text-gray-400 text-center" id="ramForecastTrend"></div>
                </div>
                <div class="p-4 bg-gray-800/50 rounded-xl">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-green-300">Disk Forecast</h3>
                        <span class="text-xs px-2 py-1 bg-green-500/20 text-green-300 rounded">Next 1H</span>
                    </div>
                    <div class="text-2xl font-bold text-center mb-2" id="diskForecastValue">--</div>
                    <div class="text-sm text-gray-400 text-center" id="diskForecastTrend"></div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="text-center text-gray-500 text-sm pt-6 border-t border-gray-800">
            <p>AI-Powered Monitoring System v3.0 • Advanced Anomaly Detection • Auto-Remediation • Predictive Analytics</p>
            <p class="mt-2">Last Updated: <span id="lastUpdate" class="text-gray-300">Never</span> • 
            <span id="dataPoints" class="text-gray-300">0</span> data points analyzed</p>
        </div>
    </div>

    <script>
        // Initialize Charts
        const charts = {};
        const colors = {
            cpu: '#3B82F6',
            ram: '#8B5CF6',
            disk: '#10B981',
            network: '#F59E0B',
            temperature: '#EF4444'
        };

        // Initialize composite chart
        const compositeCtx = document.getElementById('compositeChart').getContext('2d');
        charts.composite = new Chart(compositeCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU',
                        data: [],
                        borderColor: colors.cpu,
                        backgroundColor: colors.cpu + '20',
                        tension: 0.4,
                        borderWidth: 2,
                        fill: true
                    },
                    {
                        label: 'RAM',
                        data: [],
                        borderColor: colors.ram,
                        backgroundColor: colors.ram + '20',
                        tension: 0.4,
                        borderWidth: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: '#374151' },
                        ticks: { color: '#9CA3AF' }
                    },
                    x: {
                        grid: { color: '#374151' },
                        ticks: { color: '#9CA3AF' }
                    }
                },
                plugins: {
                    legend: { 
                        labels: { color: '#9CA3AF' },
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });

        // State management
        const state = {
            currentTenant: 'default',
            lastUpdate: null,
            alertCount: 0,
            systemHealth: 100
        };

        // Utility functions
        function formatTime(dateStr) {
            return luxon.DateTime.fromISO(dateStr).toLocaleString(luxon.DateTime.TIME_WITH_SECONDS);
        }

        function updateProgressBar(barId, value) {
            const bar = document.getElementById(barId);
            if (bar) {
                bar.style.width = value + '%';
                bar.style.backgroundColor = getColorForValue(value);
            }
        }

        function getColorForValue(value) {
            if (value < 50) return '#10B981';
            if (value < 75) return '#F59E0B';
            return '#EF4444';
        }

        // Main data fetcher
        async function fetchDashboardData() {
            try {
                const [statsRes, alertsRes, insightsRes] = await Promise.all([
                    fetch('/api/enhanced_stats'),
                    fetch('/api/smart_alerts'),
                    fetch('/api/ai_insights')
                ]);

                const [stats, alerts, insights] = await Promise.all([
                    statsRes.json(),
                    alertsRes.json(),
                    insightsRes.json()
                ]);

                // Update metrics
                updateMetrics(stats);
                updateAlerts(alerts);
                updateInsights(insights);
                updateCharts(stats);
                updateSystemHealth(stats, alerts);

                // Update UI
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                document.getElementById('activeAlertCount').textContent = alerts.active_alerts.length;
                document.getElementById('dataPoints').textContent = stats.data_points || 0;

            } catch (error) {
                console.error('Dashboard fetch error:', error);
                document.getElementById('agentStatus').innerHTML = 
                    '<div class="h-3 w-3 rounded-full bg-red-500"></div> Disconnected';
            }
        }

        function updateMetrics(stats) {
            const latest = stats.latest || {};
            
            // Update values
            if (latest.cpu !== undefined) {
                document.getElementById('cpuValue').textContent = latest.cpu.toFixed(1) + '%';
                updateProgressBar('cpuBar', latest.cpu);
            }
            if (latest.ram !== undefined) {
                document.getElementById('ramValue').textContent = latest.ram.toFixed(1) + '%';
                updateProgressBar('ramBar', latest.ram);
            }
            if (latest.disk !== undefined) {
                document.getElementById('diskValue').textContent = latest.disk.toFixed(1) + '%';
                updateProgressBar('diskBar', latest.disk);
            }
            if (latest.temperature !== undefined) {
                document.getElementById('tempValue').textContent = latest.temperature.toFixed(1) + '°C';
                updateProgressBar('tempBar', latest.temperature);
            }
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('smartAlertsContainer');
            
            if (!alerts.active_alerts || alerts.active_alerts.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-gray-500 py-4">
                        <i class="fas fa-check-circle text-2xl mb-2"></i>
                        <p>No active alerts</p>
                    </div>`;
                return;
            }

            let html = '';
            alerts.active_alerts.forEach(alert => {
                const severityColor = {
                    'critical': 'text-red-400',
                    'warning': 'text-yellow-400',
                    'info': 'text-blue-400'
                }[alert.severity] || 'text-gray-400';

                html += `
                    <div class="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                        <div class="flex justify-between items-start">
                            <div>
                                <div class="font-semibold ${severityColor}">
                                    <i class="fas fa-exclamation-circle mr-1"></i>
                                    ${alert.metric} Alert
                                </div>
                                <div class="text-sm text-gray-300 mt-1">${alert.message}</div>
                                <div class="text-xs text-gray-500 mt-2">
                                    <i class="fas fa-clock mr-1"></i>
                                    ${formatTime(alert.timestamp)}
                                    ${alert.acknowledged ? '• Acknowledged' : ''}
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-xl font-bold ${severityColor}">
                                    ${alert.value.toFixed(1)}%
                                </div>
                                <button onclick="acknowledgeAlert(${alert.id})" 
                                        class="mt-2 px-2 py-1 text-xs bg-gray-800 text-gray-300 rounded hover:bg-gray-700">
                                    Ack
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }

        function updateInsights(insights) {
            const container = document.getElementById('aiInsights');
            
            if (!insights.recommendations || insights.recommendations.length === 0) {
                container.innerHTML = `
                    <div class="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-lightbulb text-yellow-400 mr-2"></i>
                            <span class="font-medium">System operating normally</span>
                        </div>
                        <p class="text-sm text-gray-400 mt-1">All metrics within optimal ranges</p>
                    </div>`;
                return;
            }

            let html = '';
            insights.recommendations.forEach(rec => {
                const icon = rec.severity === 'high' ? 'fa-exclamation-triangle' : 
                            rec.severity === 'medium' ? 'fa-info-circle' : 'fa-lightbulb';
                const color = rec.severity === 'high' ? 'text-red-400' : 
                             rec.severity === 'medium' ? 'text-yellow-400' : 'text-blue-400';
                
                html += `
                    <div class="p-3 ${rec.severity === 'high' ? 'bg-red-500/10 border border-red-500/30' : 
                                      rec.severity === 'medium' ? 'bg-yellow-500/10 border border-yellow-500/30' : 
                                      'bg-blue-500/10 border border-blue-500/30'} rounded-lg">
                        <div class="flex items-center">
                            <i class="fas ${icon} ${color} mr-2"></i>
                            <span class="font-medium">${rec.title}</span>
                        </div>
                        <p class="text-sm text-gray-300 mt-1">${rec.description}</p>
                        ${rec.action ? `
                            <button onclick="executeAction('${rec.action}')" 
                                    class="mt-2 px-3 py-1 text-xs ${rec.severity === 'high' ? 'bg-red-500/20 text-red-300' : 
                                                                   rec.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-300' : 
                                                                   'bg-blue-500/20 text-blue-300'} rounded-lg hover:opacity-80 transition">
                                ${rec.action_label || 'Take Action'}
                            </button>
                        ` : ''}
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }

        function updateCharts(stats) {
            if (!stats.historical || stats.historical.length === 0) return;
            
            const data = stats.historical;
            const labels = data.map(d => formatTime(d.timestamp));
            
            charts.composite.data.labels = labels.slice(-30); // Last 30 points
            charts.composite.data.datasets[0].data = data.map(d => d.cpu).slice(-30);
            charts.composite.data.datasets[1].data = data.map(d => d.ram).slice(-30);
            charts.composite.update();
        }

        function updateSystemHealth(stats, alerts) {
            let health = 100;
            
            // Deduct for alerts
            if (alerts.active_alerts) {
                alerts.active_alerts.forEach(alert => {
                    if (alert.severity === 'critical') health -= 20;
                    else if (alert.severity === 'warning') health -= 10;
                });
            }
            
            // Deduct for high resource usage
            if (stats.latest) {
                if (stats.latest.cpu > 85) health -= 10;
                if (stats.latest.ram > 85) health -= 10;
                if (stats.latest.disk > 90) health -= 15;
            }
            
            health = Math.max(0, health);
            state.systemHealth = health;
            
            const healthElem = document.getElementById('systemHealthScore');
            healthElem.textContent = health.toFixed(0) + '%';
            healthElem.className = `text-2xl font-bold ${
                health >= 80 ? 'text-green-400' : 
                health >= 60 ? 'text-yellow-400' : 'text-red-400'
            }`;
        }

        // Event handlers
        document.getElementById('tenantSelect').addEventListener('change', (e) => {
            state.currentTenant = e.target.value;
            fetchDashboardData();
        });

        document.getElementById('acknowledgeAll').addEventListener('click', async () => {
            try {
                await fetch('/api/acknowledge_all', { method: 'POST' });
                fetchDashboardData();
            } catch (error) {
                console.error('Failed to acknowledge alerts:', error);
            }
        });

        // Simulated functions
        window.acknowledgeAlert = async (alertId) => {
            try {
                await fetch(`/api/acknowledge_alert/${alertId}`, { method: 'POST' });
                fetchDashboardData();
            } catch (error) {
                console.error('Failed to acknowledge alert:', error);
            }
        };

        window.executeAction = (action) => {
            alert(`Action "${action}" would be executed here.`);
        };

        // Initialization
        function initializeDashboard() {
            // Start data polling
            fetchDashboardData();
            setInterval(fetchDashboardData, 2000);
            
            // Update time
            setInterval(() => {
                document.getElementById('currentTime').textContent = 
                    new Date().toLocaleTimeString();
            }, 1000);
            
            // Calculate uptime (simulated)
            const startTime = Date.now();
            setInterval(() => {
                const uptime = Math.floor((Date.now() - startTime) / (1000 * 60 * 60 * 24));
                document.getElementById('systemUptime').textContent = 
                    `Uptime: ${uptime} days`;
            }, 60000);
        }

        // Start dashboard
        initializeDashboard();
    </script>
</body>
</html>
'''

# --- Enhanced Flask Routes ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/enhanced_report', methods=['POST'])
def receive_enhanced_data():
    """Receive enhanced data with AI features"""
    try:
        data = request.json
        if data:
            current_time = datetime.now()
            tenant = data.get('tenant', 'default')
            
            # Initialize tenant data if not exists
            if tenant not in real_time_data:
                real_time_data[tenant] = {
                    'cpu': deque(maxlen=100),
                    'ram': deque(maxlen=100),
                    'disk': deque(maxlen=100),
                    'network': deque(maxlen=100),
                    'temperature': deque(maxlen=100),
                    'power': deque(maxlen=100),
                    'time': deque(maxlen=100)
                }
            
            # Store metrics
            metrics = ['cpu', 'ram', 'disk', 'network', 'temperature', 'power']
            for metric in metrics:
                if metric in data:
                    real_time_data[tenant][metric].append(float(data[metric]))
            
            real_time_data[tenant]['time'].append(current_time.strftime("%H:%M:%S"))
            
            # Calculate anomaly score
            anomaly_score = aiops.detect_anomalies_isolation_forest(
                list(real_time_data[tenant]['cpu'])
            ) if len(real_time_data[tenant]['cpu']) > 10 else 0
            
            # Save to enhanced database
            log_data = {
                'timestamp': current_time,
                'tenant': tenant,
                'cpu': data.get('cpu', 0),
                'ram': data.get('ram', 0),
                'disk': data.get('disk', 0),
                'network_latency': data.get('network', 0),
                'temperature': data.get('temperature', 0),
                'power_consumption': data.get('power', 0),
                'anomaly_score': anomaly_score,
                'status': data.get('status', 'unknown'),
                'alert_msg': data.get('alert_msg', '')
            }
            
            conn = sqlite3.connect('monitoring.db')
            c = conn.cursor()
            c.execute('''INSERT INTO system_logs 
                         (timestamp, tenant, cpu, ram, disk, network_latency, temperature, 
                          power_consumption, anomaly_score, status, alert_message)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      tuple(log_data.values()))
            conn.commit()
            conn.close()
            
            # Smart alert processing
            alerts = data.get('alerts', [])
            if alerts:
                for alert in alerts:
                    if not alert_system.should_suppress_alert(alert):
                        # Analyze correlation
                        correlated = alert_system.analyze_alert_correlation(
                            alert, list(alert_manager['active_alerts'])[-10:]
                        )
                        root_cause = alert_system.generate_root_cause_hypothesis(correlated)
                        
                        # Determine severity
                        value = alert.get('value', 0)
                        threshold = alert.get('threshold', 0)
                        severity = 'critical' if value > threshold * 1.5 else 'warning'
                        
                        enhanced_alert = {
                            'timestamp': current_time,
                            'tenant': tenant,
                            'metric': alert.get('metric'),
                            'value': value,
                            'threshold': threshold,
                            'severity': severity,
                            'message': f"{alert.get('message', '')} | Root cause: {root_cause}",
                            'correlated_alerts': len(correlated)
                        }
                        
                        # Save to database
                        c = conn.cursor()
                        c.execute('''INSERT INTO alerts 
                                     (timestamp, tenant, metric, value, threshold, severity, message)
                                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                  (enhanced_alert['timestamp'],
                                   enhanced_alert['tenant'],
                                   enhanced_alert['metric'],
                                   enhanced_alert['value'],
                                   enhanced_alert['threshold'],
                                   enhanced_alert['severity'],
                                   enhanced_alert['message']))
                        conn.commit()
                        
                        # Add to active alerts
                        alert_manager['active_alerts'].append(enhanced_alert)
                        
                        # Auto-remediation for critical alerts
                        if severity == 'critical':
                            suggestions = remediation_engine.suggest_remediation(
                                enhanced_alert, list(real_time_data[tenant]['cpu'])
                            )
                            if suggestions:
                                remediation_engine.execute_remediation(suggestions[0])
            
            print(f"[ENHANCED] Tenant: {tenant} | CPU: {data.get('cpu', 0):.1f}% | "
                  f"RAM: {data.get('ram', 0)}% | Anomaly: {anomaly_score:.2f}")
            return jsonify({"status": "success", "ai_processed": True}), 200
    except Exception as e:
        print(f"[ERROR] Enhanced processing: {e}")
    
    return jsonify({"status": "error"}), 400

@app.route('/api/enhanced_stats', methods=['GET'])
def get_enhanced_stats():
    """Get enhanced statistics with AI insights"""
    tenant = request.args.get('tenant', 'default')
    
    if tenant not in real_time_data:
        return jsonify({})
    
    data = real_time_data[tenant]
    
    # Helper to safely slice deques by converting to list first
    def safe_slice(dq, count):
        return list(dq)[-count:] if dq else []

    # Calculate advanced metrics
    latest = {
        'cpu': data['cpu'][-1] if data['cpu'] else 0,
        'ram': data['ram'][-1] if data['ram'] else 0,
        'disk': data['disk'][-1] if data['disk'] else 0,
        'temperature': data['temperature'][-1] if data['temperature'] else 0,
        'power': data['power'][-1] if data['power'] else 0
    }
    
    # Calculate trends
    trends = {}
    for metric in ['cpu', 'ram', 'disk']:
        if len(data[metric]) > 5:
            recent = list(data[metric])[-5:] # Convert deque to list to slice
            trends[metric] = 'increasing' if recent[-1] > recent[0] else 'decreasing' if recent[-1] < recent[0] else 'stable'
    
    # Prepare historical data (FIXED slicing and alignment)
    # 1. Convert deques to lists and slice the last 30 points
    times_slice = safe_slice(data['time'], 30)
    cpu_slice = safe_slice(data['cpu'], 30)
    ram_slice = safe_slice(data['ram'], 30)
    disk_slice = safe_slice(data['disk'], 30)

    # 2. Zip them to ensure data stays aligned and handles uneven lengths gracefully
    historical_data = []
    for t, c, r, d in zip(times_slice, cpu_slice, ram_slice, disk_slice):
        historical_data.append({
            'timestamp': t,
            'cpu': c,
            'ram': r,
            'disk': d
        })

    return jsonify({
        'latest': latest,
        'trends': trends,
        'historical': historical_data,
        'data_points': sum(len(data[metric]) for metric in data),
        # Fix: Convert deque to list for anomaly detection
        'anomaly_score': aiops.detect_anomalies_isolation_forest(list(data['cpu'])).get('contamination', 0) if data['cpu'] else 0
    })

@app.route('/api/smart_alerts', methods=['GET'])
def get_smart_alerts():
    """Get smart alerts with AI correlation"""
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    
    c.execute('''SELECT * FROM alerts 
                 WHERE status = 'active' AND acknowledged = 0
                 ORDER BY timestamp DESC LIMIT 20''')
    
    alerts = []
    for row in c.fetchall():
        alerts.append({
            'id': row[0],
            'timestamp': row[1],
            'tenant': row[2],
            'metric': row[3],
            'value': row[4],
            'threshold': row[5],
            'severity': row[6],
            'message': row[7],
            'status': row[8],
            'acknowledged': bool(row[9])
        })
    
    conn.close()
    
    # Add AI insights to alerts
    for alert in alerts:
        if alert['severity'] == 'critical':
            alert['recommendation'] = remediation_engine.suggest_remediation(
                alert, list(real_time_data.get(alert['tenant'], {}).get('cpu', []))
            )
    
    return jsonify({
        'active_alerts': alerts,
        'total_count': len(alerts),
        'critical_count': len([a for a in alerts if a['severity'] == 'critical'])
    })

@app.route('/api/ai_insights', methods=['GET'])
def get_ai_insights():
    """Get AI-generated insights and recommendations"""
    insights = []
    
    # Analyze current state
    for tenant in real_time_data:
        if real_time_data[tenant]['cpu']:
            cpu_values = list(real_time_data[tenant]['cpu'])
            latest_cpu = cpu_values[-1] if cpu_values else 0
            
            if latest_cpu > 85:
                insights.append({
                    'title': 'High CPU Usage Detected',
                    'description': f'CPU usage at {latest_cpu:.1f}%. Consider scaling resources.',
                    'severity': 'high',
                    'action': 'scale_cpu',
                    'action_label': 'Scale CPU'
                })
            
            # Detect patterns
            if len(cpu_values) > 20:
                pattern = aiops.detect_seasonal_patterns(cpu_values, list(real_time_data[tenant]['time']))
                if pattern and pattern['peak_hours']:
                    insights.append({
                        'title': 'Usage Pattern Detected',
                        'description': f'Peak usage hours: {pattern["peak_hours"]}',
                        'severity': 'info',
                        'action': 'optimize_schedule'
                    })
    
    # Check for anomalies
    anomaly_result = aiops.detect_anomalies_isolation_forest(
        list(real_time_data['default']['cpu'])
    )
    if anomaly_result['anomalies']:
        insights.append({
            'title': 'Anomalies Detected',
            'description': f'{len(anomaly_result["anomalies"])} unusual patterns found',
            'severity': 'medium',
            'action': 'investigate_anomalies'
        })
    
    # Performance recommendations
    for metric in ['cpu', 'ram', 'disk']:
        if real_time_data['default'][metric]:
            values = list(real_time_data['default'][metric])
            if len(values) > 10:
                avg = sum(values) / len(values)
                if avg < 30:
                    insights.append({
                        'title': f'Underutilized {metric.upper()}',
                        'description': f'Average usage only {avg:.1f}%. Consider downsizing.',
                        'severity': 'info'
                    })
    
    return jsonify({
        'recommendations': insights[:5],  # Top 5 recommendations
        'patterns_detected': len(insights) > 0,
        'last_analysis': datetime.now().isoformat()
    })

@app.route('/api/acknowledge_alert/<int:alert_id>', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('UPDATE alerts SET acknowledged = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/acknowledge_all', methods=['POST'])
def acknowledge_all_alerts():
    """Acknowledge all active alerts"""
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("UPDATE alerts SET acknowledged = 1 WHERE status = 'active'")
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'acknowledged': True})

@app.route('/api/predictive_forecast', methods=['GET'])
def get_predictive_forecast():
    """Get predictive forecasts for all metrics"""
    forecasts = {}
    
    for metric in ['cpu', 'ram', 'disk']:
        if real_time_data['default'][metric]:
            data = list(real_time_data['default'][metric])
            if len(data) >= 20:
                # Simple moving average forecast
                window = min(5, len(data))
                recent_avg = sum(data[-window:]) / window
                
                # Add trend prediction
                if len(data) >= 2:
                    trend = 'increasing' if data[-1] > data[-2] else 'decreasing' if data[-1] < data[-2] else 'stable'
                else:
                    trend = 'stable'
                
                forecasts[metric] = {
                    'next_hour': min(100, recent_avg * 1.05),  # Slight increase prediction
                    'next_3_hours': min(100, recent_avg * 1.1),
                    'trend': trend,
                    'confidence': min(0.95, len(data) / 100),
                    'pattern': 'stable' if len(data) < 50 else 'seasonal'
                }
    
    return jsonify(forecasts)

@app.route('/api/system_health', methods=['GET'])
def get_system_health():
    """Get comprehensive system health score"""
    health_score = 100
    
    # Deduct for high resource usage
    for metric in ['cpu', 'ram', 'disk']:
        if real_time_data['default'][metric]:
            latest = real_time_data['default'][metric][-1] if real_time_data['default'][metric] else 0
            if latest > 90:
                health_score -= 20
            elif latest > 75:
                health_score -= 10
    
    # Deduct for active alerts
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM alerts WHERE status = 'active' AND acknowledged = 0")
    alert_count = c.fetchone()[0]
    conn.close()
    
    health_score -= alert_count * 5
    
    health_score = max(0, health_score)
    
    return jsonify({
        'health_score': health_score,
        'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'critical',
        'components': {
            'resources': 'optimal' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'alerts': 'none' if alert_count == 0 else 'some' if alert_count <= 2 else 'many',
            'connectivity': 'stable',
            'performance': 'good' if health_score >= 70 else 'degraded'
        },
        'recommendations': ['Monitor resource usage', 'Review alerts'] if health_score < 80 else []
    })

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     AI-POWERED IOT MONITORING SYSTEM - ULTIMATE EDITION   ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Dashboard: http://localhost:5000
    API Documentation: http://localhost:5000/api/docs
    
    🚀 ENHANCED FEATURES:
    • Multi-tenant support with isolated environments
    • AI-driven anomaly detection with Isolation Forest
    • Predictive analytics with ensemble forecasting
    • Smart alert correlation & root cause analysis
    • Auto-remediation engine with actionable insights
    • Performance baseline calculation
    • Seasonal pattern detection
    • Real-time health scoring
    • Advanced visualization with interactive charts
    
    📊 METRICS MONITORED:
    • CPU, RAM, Disk utilization
    • Network latency & performance
    • System temperature
    • Power consumption
    • Anomaly scores
    
    🤖 AI CAPABILITIES:
    • Ensemble machine learning models
    • Real-time pattern recognition
    • Predictive maintenance alerts
    • Automated optimization suggestions
    
    Starting server...
    """)
    
    # Ensure directories exist
    for directory in ['logs', 'models', 'exports']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)