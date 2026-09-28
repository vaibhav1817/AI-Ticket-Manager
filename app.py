import os
import json
import logging
from flask import Flask, request, jsonify, render_template_string
from model_pipeline import TicketClassifier, train_and_save_all_models, MODELS_DIR
import review_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize or load classifier
classifier = TicketClassifier(confidence_threshold=0.50)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Support Ticket Intelligence & Auto-Routing</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(26, 34, 54, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(99, 102, 241, 0.5);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.35);
      --accent: #8b5cf6;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.12);
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.15);
      --info: #06b6d4;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 15% 15%, #151b2e 0%, #090d16 60%, #05070c 100%);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
      padding-bottom: 60px;
    }

    /* Ambient glow elements */
    .glow-blob {
      position: fixed;
      width: 450px;
      height: 450px;
      border-radius: 50%;
      background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
      filter: blur(60px);
      top: -100px;
      right: -100px;
      opacity: 0.6;
    }

    .container {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 20px;
      position: relative;
      z-index: 1;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 0 28px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
      flex-wrap: wrap;
      gap: 16px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-icon {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 0 24px var(--primary-glow);
    }

    .brand h1 {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand p {
      font-size: 13px;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .badge-status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 30px;
      font-size: 12px;
      font-weight: 600;
      background: var(--success-bg);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 10px var(--success);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Tabs Navigation */
    .tabs-nav {
      display: flex;
      gap: 8px;
      margin-bottom: 28px;
      background: rgba(15, 23, 42, 0.6);
      padding: 6px;
      border-radius: 14px;
      border: 1px solid var(--border);
      width: fit-content;
      max-width: 100%;
      overflow-x: auto;
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 10px 20px;
      border-radius: 10px;
      font-family: inherit;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
      white-space: nowrap;
    }

    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
    }

    .tab-btn.active {
      color: #fff;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      box-shadow: 0 4px 16px var(--primary-glow);
    }

    .tab-badge {
      background: rgba(239, 68, 68, 0.8);
      color: #fff;
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 12px;
      font-weight: 700;
    }

    /* Tab Content panels */
    .tab-panel {
      display: none;
    }

    .tab-panel.active {
      display: block;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Grid Layouts */
    .grid-2 {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }

    @media (max-width: 900px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    /* Glass Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      backdrop-filter: blur(16px);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      position: relative;
    }

    .card-title {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title-icon {
      margin-right: 8px;
      color: var(--primary);
    }

    /* Form Inputs */
    .form-group {
      margin-bottom: 18px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    input[type="text"], textarea {
      width: 100%;
      background: rgba(10, 15, 26, 0.7);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
      transition: all 0.2s ease;
      outline: none;
    }

    input[type="text"]:focus, textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }

    textarea {
      min-height: 120px;
      resize: vertical;
    }

    /* Sample ticket chips */
    .samples-wrap {
      margin-bottom: 20px;
    }

    .samples-label {
      font-size: 12px;
      color: var(--text-dim);
      margin-bottom: 8px;
    }

    .samples-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .sample-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 6px 12px;
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .sample-chip:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--primary);
      color: #fff;
    }

    /* Threshold Slider */
    .slider-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
      background: rgba(10, 15, 26, 0.5);
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }

    .slider-row input[type="range"] {
      width: 180px;
      accent-color: var(--primary);
    }

    /* Action Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 22px;
      border-radius: 12px;
      font-family: inherit;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #fff;
      box-shadow: 0 4px 18px var(--primary-glow);
      width: 100%;
    }

    .btn-primary:hover {
      box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5);
      transform: translateY(-1px);
    }

    .btn-sm {
      padding: 6px 14px;
      font-size: 12px;
      border-radius: 8px;
    }

    .btn-outline {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
    }

    .btn-outline:hover {
      background: rgba(255, 255, 255, 0.06);
    }

    .btn-success {
      background: var(--success);
      color: #fff;
    }

    /* Result Panels */
    .result-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 280px;
      text-align: center;
      color: var(--text-dim);
    }

    .result-empty-icon {
      font-size: 40px;
      margin-bottom: 12px;
      opacity: 0.6;
    }

    .banner-decision {
      padding: 14px 18px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
      font-size: 13px;
      font-weight: 600;
    }

    .banner-decision.auto {
      background: var(--success-bg);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--success);
    }

    .banner-decision.review {
      background: var(--warning-bg);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: var(--warning);
    }

    .metric-grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 20px;
    }

    .metric-box {
      background: rgba(10, 15, 26, 0.6);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 12px;
    }

    .metric-box-label {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    .metric-box-val {
      font-size: 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Urgency Tag styles */
    .urgency-badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .urgency-critical { background: var(--danger-bg); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.4); }
    .urgency-high { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.4); }
    .urgency-medium { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.4); }
    .urgency-low { background: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Category Tag */
    .category-chip {
      background: rgba(99, 102, 241, 0.18);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
    }

    /* Confidence bar */
    .conf-bar-wrap {
      margin-top: 14px;
    }

    .conf-header {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 6px;
      color: var(--text-muted);
    }

    .conf-bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
    }

    .conf-bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Distribution bars */
    .prob-row {
      display: flex;
      align-items: center;
      font-size: 12px;
      gap: 10px;
      margin-bottom: 8px;
    }

    .prob-name {
      width: 140px;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .prob-bar {
      flex: 1;
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 4px;
      overflow: hidden;
    }

    .prob-fill {
      height: 100%;
      background: var(--primary);
      border-radius: 4px;
    }

    .prob-pct {
      width: 44px;
      text-align: right;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
    }

    /* Review Queue Table */
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(10, 15, 26, 0.4);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }

    th, td {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }

    th {
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .status-badge {
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      display: inline-block;
    }

    .status-pending { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-resolved { background: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-auto { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }

    /* Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .modal-overlay.open {
      display: flex;
    }

    .modal {
      background: #111827;
      border: 1px solid var(--border);
      border-radius: 18px;
      width: 100%;
      max-width: 560px;
      padding: 26px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    }

    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }

    .modal-close {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 20px;
      cursor: pointer;
    }

    select {
      width: 100%;
      background: rgba(10, 15, 26, 0.8);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
      outline: none;
    }

    /* Stats summary grid */
    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
    }

    .stat-card-title {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    .stat-card-val {
      font-size: 24px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Matrix Table */
    .cm-table {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      text-align: center;
    }

    .cm-table td, .cm-table th {
      padding: 8px 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .cm-cell-val {
      padding: 4px;
      border-radius: 4px;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="glow-blob"></div>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">⚡</div>
        <div>
          <h1>AI Ticket Manager & Auto-Router</h1>
          <p>Text Feature Extraction (TF-IDF), Multiclass Classification, Urgency Modeling & Human-in-the-Loop Review</p>
        </div>
      </div>
      <div class="header-actions">
        <span class="badge-status">
          <span class="status-dot"></span>
          Models Active & Serving
        </span>
      </div>
    </header>

    <!-- Operational Stats Row -->
    <div class="stats-row" id="statsRow">
      <div class="stat-card">
        <div class="stat-card-title">Total Tickets Processed</div>
        <div class="stat-card-val" id="statTotal">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Auto-Routed Rate</div>
        <div class="stat-card-val" id="statAuto" style="color: var(--success);">--%</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Awaiting Human Review</div>
        <div class="stat-card-val" id="statPending" style="color: var(--warning);">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Human Resolved</div>
        <div class="stat-card-val" id="statResolved" style="color: var(--info);">--</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('classifierTab', this)">
        <span>🎯 Live Classifier & Router</span>
      </button>
      <button class="tab-btn" onclick="switchTab('reviewsTab', this)">
        <span>📋 Human Review Queue</span>
        <span class="tab-badge" id="reviewBadge" style="display:none;">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('metricsTab', this)">
        <span>📊 Model Evaluation & Metrics</span>
      </button>
    </div>

    <!-- TAB 1: CLASSIFIER & ROUTER -->
    <div id="classifierTab" class="tab-panel active">
      <div class="grid-2">
        <!-- Input Form Card -->
        <div class="card">
          <div class="card-title">
            <span>Incoming Support Ticket</span>
          </div>

          <!-- Quick Samples -->
          <div class="samples-wrap">
            <div class="samples-label">Click a pre-set sample to test auto-routing & confidence scoring:</div>
            <div class="samples-chips">
              <span class="sample-chip" onclick="loadSample(1)">💳 Double Charge</span>
              <span class="sample-chip" onclick="loadSample(2)">🔥 Server Outage / Downtime</span>
              <span class="sample-chip" onclick="loadSample(3)">📦 Refund Damaged Goods</span>
              <span class="sample-chip" onclick="loadSample(4)">❓ Spec Inquiry</span>
              <span class="sample-chip" onclick="loadSample(5)">⚠️ Ambiguous Issue (Low Conf)</span>
            </div>
          </div>

          <form id="classifyForm" onsubmit="handleClassify(event)">
            <div class="form-group">
              <label for="ticketSubject">Ticket Subject</label>
              <input type="text" id="ticketSubject" placeholder="e.g. Unauthorized charge on credit card" required>
            </div>

            <div class="form-group">
              <label for="ticketBody">Ticket Description</label>
              <textarea id="ticketBody" placeholder="Paste or type customer complaint / query text here..." required></textarea>
            </div>

            <div class="slider-row">
              <div>
                <label style="margin:0;">Confidence Routing Threshold</label>
                <div style="font-size: 11px; color: var(--text-dim);">Flag for human review if confidence falls below this</div>
              </div>
              <div style="display: flex; align-items: center; gap: 10px;">
                <input type="range" id="confSlider" min="0.20" max="0.90" step="0.05" value="0.50" oninput="updateThreshold(this.value)">
                <span id="sliderVal" style="font-family:'JetBrains Mono', monospace; font-size:13px; font-weight:700;">50%</span>
              </div>
            </div>

            <button type="submit" class="btn btn-primary" id="btnSubmit">
              <span>Classify & Route Ticket</span>
            </button>
          </form>
        </div>

        <!-- Output Card -->
        <div class="card" id="outputCard">
          <div class="card-title">
            <span>Routing Decision & Intelligence</span>
          </div>

          <div id="outputEmpty" class="result-empty">
            <div class="result-empty-icon">🤖</div>
            <p>Submit a ticket on the left to see TF-IDF feature extraction, predicted Category, separate Urgency model output, and automated routing decision.</p>
          </div>

          <div id="outputContent" style="display: none;">
            <!-- Routing Banner -->
            <div id="decisionBanner" class="banner-decision auto">
              <span id="bannerText">✅ Auto-Routed to Support Queue</span>
              <span id="ticketIdTag" style="font-family:'JetBrains Mono', monospace; font-size:11px; opacity:0.9;"></span>
            </div>

            <!-- Predicted Category & Urgency -->
            <div class="metric-grid-2">
              <div class="metric-box">
                <div class="metric-box-label">Predicted Category</div>
                <div class="metric-box-val">
                  <span id="resCategory" class="category-chip">--</span>
                </div>
                <div class="conf-bar-wrap">
                  <div class="conf-header">
                    <span>Model Confidence</span>
                    <span id="catConfPct">--%</span>
                  </div>
                  <div class="conf-bar-bg">
                    <div id="catConfBar" class="conf-bar-fill" style="background: var(--primary); width: 0%;"></div>
                  </div>
                </div>
              </div>

              <div class="metric-box">
                <div class="metric-box-label">Predicted Urgency</div>
                <div class="metric-box-val">
                  <span id="resUrgency" class="urgency-badge urgency-medium">--</span>
                </div>
                <div class="conf-bar-wrap">
                  <div class="conf-header">
                    <span>Model Confidence</span>
                    <span id="urgConfPct">--%</span>
                  </div>
                  <div class="conf-bar-bg">
                    <div id="urgConfBar" class="conf-bar-fill" style="background: #f97316; width: 0%;"></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Preprocessed Text Accordion -->
            <div style="margin-bottom: 18px; font-size:12px; background: rgba(10,15,26,0.5); padding:10px 14px; border-radius:10px; border:1px solid var(--border);">
              <span style="color:var(--text-dim); font-weight:600;">Cleaned Text for TF-IDF:</span>
              <div id="cleanedTextPreview" style="color:var(--text-muted); font-family:'JetBrains Mono', monospace; margin-top:4px; max-height:60px; overflow-y:auto;"></div>
            </div>

            <!-- Candidate Probabilities -->
            <div style="font-size: 13px; font-weight:600; margin-bottom: 8px;">Category Class Probabilities</div>
            <div id="categoryProbsList" style="margin-bottom: 16px;"></div>

            <div style="font-size: 13px; font-weight:600; margin-bottom: 8px;">Urgency Class Probabilities</div>
            <div id="urgencyProbsList"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: HUMAN REVIEW QUEUE -->
    <div id="reviewsTab" class="tab-panel">
      <div class="card">
        <div class="card-title">
          <span>Human-in-the-Loop Review Queue (Low Confidence Flagged)</span>
          <div style="display:flex; gap:10px;">
            <button class="btn btn-outline btn-sm" onclick="loadReviews('pending')">Pending Only</button>
            <button class="btn btn-outline btn-sm" onclick="loadReviews('all')">View All</button>
          </div>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 18px;">
          Tickets where the classifier confidence fell below the safety threshold are automatically held here for human verification and active learning escalation.
        </p>

        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticket ID</th>
                <th>Timestamp</th>
                <th>Content Preview</th>
                <th>Predicted Category</th>
                <th>Predicted Urgency</th>
                <th>Overall Conf.</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="reviewsTableBody">
              <tr>
                <td colspan="8" style="text-align: center; color: var(--text-dim); padding: 30px;">Loading review queue...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: MODEL EVALUATION & METRICS -->
    <div id="metricsTab" class="tab-panel">
      <div class="grid-2">
        <!-- Category Model Metrics -->
        <div class="card">
          <div class="card-title">
            <span>Multiclass Category Classifier Evaluation</span>
            <span class="category-chip" id="catAccHeader">Acc: --%</span>
          </div>

          <div style="display:flex; gap:16px; margin-bottom: 16px;">
            <div class="metric-box" style="flex:1;">
              <div class="metric-box-label">Macro F1-Score</div>
              <div class="metric-box-val" id="catMacroF1">--</div>
            </div>
            <div class="metric-box" style="flex:1;">
              <div class="metric-box-label">Weighted F1-Score</div>
              <div class="metric-box-val" id="catWeightedF1">--</div>
            </div>
          </div>

          <div style="font-size: 13px; font-weight:600; margin-bottom: 10px;">Precision, Recall & F1-Score by Class</div>
          <div class="table-wrap" style="margin-bottom: 20px;">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody id="catMetricsTable"></tbody>
            </table>
          </div>

          <div style="font-size: 13px; font-weight:600; margin-bottom: 10px;">Confusion Matrix (Actual vs Predicted)</div>
          <div class="table-wrap" id="catCmContainer"></div>
        </div>

        <!-- Urgency Model Metrics -->
        <div class="card">
          <div class="card-title">
            <span>Separate Urgency Classifier Evaluation</span>
            <span class="urgency-badge urgency-high" id="urgAccHeader">Acc: --%</span>
          </div>

          <div style="display:flex; gap:16px; margin-bottom: 16px;">
            <div class="metric-box" style="flex:1;">
              <div class="metric-box-label">Macro F1-Score</div>
              <div class="metric-box-val" id="urgMacroF1">--</div>
            </div>
            <div class="metric-box" style="flex:1;">
              <div class="metric-box-label">Weighted F1-Score</div>
              <div class="metric-box-val" id="urgWeightedF1">--</div>
            </div>
          </div>

          <div style="font-size: 13px; font-weight:600; margin-bottom: 10px;">Precision, Recall & F1-Score by Class</div>
          <div class="table-wrap" style="margin-bottom: 20px;">
            <table>
              <thead>
                <tr>
                  <th>Urgency Level</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody id="urgMetricsTable"></tbody>
            </table>
          </div>

          <div style="font-size: 13px; font-weight:600; margin-bottom: 10px;">Confusion Matrix (Actual vs Predicted)</div>
          <div class="table-wrap" id="urgCmContainer"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Review Modal -->
  <div class="modal-overlay" id="reviewModal">
    <div class="modal">
      <div class="modal-header">
        <h3 style="font-size: 18px;">Review Ticket & Correct Labels</h3>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>

      <div style="background: rgba(10,15,26,0.6); padding: 12px; border-radius: 10px; margin-bottom: 16px; font-size: 13px;">
        <div style="color: var(--text-dim); margin-bottom: 4px;">Ticket ID: <span id="modalReviewId" style="color:var(--text); font-family:'JetBrains Mono', monospace;"></span></div>
        <div id="modalRawText" style="color: var(--text-muted); max-height: 90px; overflow-y: auto;"></div>
      </div>

      <div class="form-group">
        <label>Final Category</label>
        <select id="modalFinalCategory">
          <option value="Billing inquiry">Billing inquiry</option>
          <option value="Technical issue">Technical issue</option>
          <option value="Cancellation request">Cancellation request</option>
          <option value="Product inquiry">Product inquiry</option>
          <option value="Refund request">Refund request</option>
        </select>
      </div>

      <div class="form-group">
        <label>Final Urgency</label>
        <select id="modalFinalUrgency">
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <div class="form-group">
        <label>Reviewer Notes / Feedback</label>
        <input type="text" id="modalNotes" placeholder="Reason for change or approval note...">
      </div>

      <div style="display:flex; justify-content: flex-end; gap: 10px; margin-top: 20px;">
        <button class="btn btn-outline btn-sm" onclick="closeModal()">Cancel</button>
        <button class="btn btn-success btn-sm" onclick="submitReviewResolution()">Approve & Resolve</button>
      </div>
    </div>
  </div>

  <script>
    let currentThreshold = 0.50;
    let selectedReviewId = null;

    // Sample Tickets
    const SAMPLES = {
      1: {
        subject: "Charged twice for monthly cloud subscription",
        body: "I noticed on my bank statement that I was billed twice on September 1st for the same subscription. Please refund the duplicate transaction of $49.99 immediately."
      },
      2: {
        subject: "CRITICAL: Database connection pool exhausted - production outage",
        body: "All backend microservices are throwing 500 internal server errors. The primary PostgreSQL connection pool is saturated and users cannot log in. High business impact!"
      },
      3: {
        subject: "Package arrived with broken glass and missing power adapter",
        body: "My order delivered today, but when I opened the box, the product casing was completely smashed and the charger was not included. I would like a replacement or refund."
      },
      4: {
        subject: "Does this device support Bluetooth 5.3 and dual pairing?",
        body: "Hello support, I am planning to purchase this headset for my office. Could you please confirm if it supports multipoint Bluetooth pairing with both a PC and iPhone simultaneously?"
      },
      5: {
        subject: "Checking on something general",
        body: "Not sure what happened but thing feels different today maybe it is fine though."
      }
    };

    function loadSample(id) {
      const sample = SAMPLES[id];
      if (!sample) return;
      document.getElementById('ticketSubject').value = sample.subject;
      document.getElementById('ticketBody').value = sample.body;
    }

    function updateThreshold(val) {
      currentThreshold = parseFloat(val);
      document.getElementById('sliderVal').innerText = Math.round(currentThreshold * 100) + '%';
    }

    function switchTab(tabId, el) {
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      el.classList.add('active');

      if (tabId === 'reviewsTab') loadReviews('pending');
      if (tabId === 'metricsTab') loadMetrics();
    }

    // Classification Request
    async function handleClassify(e) {
      e.preventDefault();
      const subject = document.getElementById('ticketSubject').value.trim();
      const body = document.getElementById('ticketBody').value.trim();
      const btn = document.getElementById('btnSubmit');

      btn.disabled = true;
      btn.innerText = 'Analyzing & Vectorizing...';

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            subject: subject,
            text: body,
            threshold: currentThreshold
          })
        });

        const data = await res.json();
        renderPrediction(data);
        loadStats();
      } catch (err) {
        alert('Error processing ticket: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerText = 'Classify & Route Ticket';
      }
    }

    function renderPrediction(data) {
      document.getElementById('outputEmpty').style.display = 'none';
      document.getElementById('outputContent').style.display = 'block';

      // Decision Banner
      const banner = document.getElementById('decisionBanner');
      const bannerText = document.getElementById('bannerText');
      const ticketIdTag = document.getElementById('ticketIdTag');
      ticketIdTag.innerText = data.review_id || '';

      if (data.needs_human_review) {
        banner.className = 'banner-decision review';
        bannerText.innerHTML = `⚠️ Flagged for Human Review (${data.review_reason})`;
      } else {
        banner.className = 'banner-decision auto';
        bannerText.innerHTML = `✅ Auto-Routed (${Math.round(data.overall_confidence * 100)}% Confidence)`;
      }

      // Category
      document.getElementById('resCategory').innerText = data.category.prediction;
      const catConfPct = Math.round(data.category.confidence * 100);
      document.getElementById('catConfPct').innerText = catConfPct + '%';
      document.getElementById('catConfBar').style.width = catConfPct + '%';

      // Urgency
      const urgEl = document.getElementById('resUrgency');
      const urgPred = data.urgency.prediction;
      urgEl.innerText = urgPred;
      urgEl.className = 'urgency-badge urgency-' + urgPred.toLowerCase();
      const urgConfPct = Math.round(data.urgency.confidence * 100);
      document.getElementById('urgConfPct').innerText = urgConfPct + '%';
      document.getElementById('urgConfBar').style.width = urgConfPct + '%';

      // Cleaned text preview
      document.getElementById('cleanedTextPreview').innerText = data.clean_text;

      // Probabilities list
      renderProbBars('categoryProbsList', data.category.all_probabilities);
      renderProbBars('urgencyProbsList', data.urgency.all_probabilities);
    }

    function renderProbBars(containerId, probs) {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      for (const [cls, val] of Object.entries(probs)) {
        const pct = Math.round(val * 100);
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
          <span class="prob-name" title="${cls}">${cls}</span>
          <div class="prob-bar">
            <div class="prob-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="prob-pct">${pct}%</span>
        `;
        container.appendChild(row);
      }
    }

    // Reviews Queue
    async function loadReviews(filter = 'pending') {
      const tbody = document.getElementById('reviewsTableBody');
      try {
        const res = await fetch(`/api/reviews?filter=${filter}`);
        const data = await res.json();

        if (!data || data.length === 0) {
          tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--text-dim); padding:24px;">No tickets currently in this view.</td></tr>`;
          return;
        }

        tbody.innerHTML = '';
        data.forEach(r => {
          const tr = document.createElement('tr');
          const isPending = r.status === 'PENDING_REVIEW';
          const statusClass = isPending ? 'status-pending' : (r.status === 'AUTO_ROUTED' ? 'status-auto' : 'status-resolved');
          const urgClass = 'urgency-' + (r.final_urgency || r.predicted_urgency || 'medium').toLowerCase();

          tr.innerHTML = `
            <td style="font-family:'JetBrains Mono', monospace; font-size:12px; font-weight:600;">${r.review_id}</td>
            <td style="font-size:11px; color:var(--text-dim);">${r.timestamp}</td>
            <td style="max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${r.raw_text}">${r.raw_text}</td>
            <td><span class="category-chip" style="font-size:11px;">${r.final_category || r.predicted_category}</span></td>
            <td><span class="urgency-badge ${urgClass}" style="font-size:10px;">${r.final_urgency || r.predicted_urgency}</span></td>
            <td style="font-family:'JetBrains Mono', monospace;">${Math.round(r.overall_confidence * 100)}%</td>
            <td><span class="status-badge ${statusClass}">${r.status}</span></td>
            <td>
              ${isPending ? `<button class="btn btn-outline btn-sm" onclick="openReviewModal('${r.review_id}', '${escapeQuotes(r.raw_text)}', '${r.predicted_category}', '${r.predicted_urgency}')">Review</button>` : `<span style="color:var(--text-dim); font-size:11px;">Resolved</span>`}
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--danger);">Error loading reviews: ${err}</td></tr>`;
      }
    }

    function escapeQuotes(str) {
      return (str || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
    }

    function openReviewModal(id, rawText, cat, urg) {
      selectedReviewId = id;
      document.getElementById('modalReviewId').innerText = id;
      document.getElementById('modalRawText').innerText = rawText;
      document.getElementById('modalFinalCategory').value = cat;
      document.getElementById('modalFinalUrgency').value = urg;
      document.getElementById('modalNotes').value = '';
      document.getElementById('reviewModal').classList.add('open');
    }

    function closeModal() {
      document.getElementById('reviewModal').classList.remove('open');
      selectedReviewId = null;
    }

    async function submitReviewResolution() {
      if (!selectedReviewId) return;
      const finalCat = document.getElementById('modalFinalCategory').value;
      const finalUrg = document.getElementById('modalFinalUrgency').value;
      const notes = document.getElementById('modalNotes').value;

      try {
        const res = await fetch(`/api/reviews/${selectedReviewId}/resolve`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            final_category: finalCat,
            final_urgency: finalUrg,
            notes: notes
          })
        });

        const data = await res.json();
        if (data.success) {
          closeModal();
          loadReviews('pending');
          loadStats();
        } else {
          alert('Could not resolve: ' + data.error);
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    // Stats
    async function loadStats() {
      try {
        const res = await fetch('/api/stats');
        const d = await res.json();
        document.getElementById('statTotal').innerText = d.total_tickets;
        document.getElementById('statAuto').innerText = d.auto_route_rate_pct + '%';
        document.getElementById('statPending').innerText = d.pending_review_count;
        document.getElementById('statResolved').innerText = d.human_resolved_count;

        const badge = document.getElementById('reviewBadge');
        if (d.pending_review_count > 0) {
          badge.style.display = 'inline-block';
          badge.innerText = d.pending_review_count;
        } else {
          badge.style.display = 'none';
        }
      } catch (e) {
        console.error('Stats error:', e);
      }
    }

    // Metrics
    async function loadMetrics() {
      try {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        // Category Metrics
        const cat = data.category_model;
        document.getElementById('catAccHeader').innerText = `Acc: ${(cat.accuracy * 100).toFixed(1)}%`;
        document.getElementById('catMacroF1').innerText = (cat.macro_f1).toFixed(3);
        document.getElementById('catWeightedF1').innerText = (cat.weighted_f1).toFixed(3);

        renderReportTable('catMetricsTable', cat.classification_report, cat.classes);
        renderConfusionMatrix('catCmContainer', cat.classes, cat.confusion_matrix);

        // Urgency Metrics
        const urg = data.urgency_model;
        document.getElementById('urgAccHeader').innerText = `Acc: ${(urg.accuracy * 100).toFixed(1)}%`;
        document.getElementById('urgMacroF1').innerText = (urg.macro_f1).toFixed(3);
        document.getElementById('urgWeightedF1').innerText = (urg.weighted_f1).toFixed(3);

        renderReportTable('urgMetricsTable', urg.classification_report, urg.classes);
        renderConfusionMatrix('urgCmContainer', urg.classes, urg.confusion_matrix);
      } catch (e) {
        console.error('Metrics error:', e);
      }
    }

    function renderReportTable(targetId, report, classes) {
      const tbody = document.getElementById(targetId);
      tbody.innerHTML = '';
      classes.forEach(cls => {
        const item = report[cls] || {};
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight:600;">${cls}</td>
          <td style="font-family:'JetBrains Mono', monospace;">${((item.precision || 0) * 100).toFixed(1)}%</td>
          <td style="font-family:'JetBrains Mono', monospace;">${((item.recall || 0) * 100).toFixed(1)}%</td>
          <td style="font-family:'JetBrains Mono', monospace; font-weight:700; color:var(--primary);">${(item['f1-score'] || 0).toFixed(3)}</td>
          <td style="color:var(--text-dim);">${item.support || 0}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function renderConfusionMatrix(containerId, classes, matrix) {
      const container = document.getElementById(containerId);
      let html = '<table class="cm-table"><thead><tr><th>Actual \\ Pred</th>';
      classes.forEach(c => {
        html += `<th>${c.length > 9 ? c.substring(0, 8) + '..' : c}</th>`;
      });
      html += '</tr></thead><tbody>';

      matrix.forEach((row, i) => {
        const rowSum = row.reduce((a, b) => a + b, 0);
        html += `<tr><td style="font-weight:600; text-align:left;">${classes[i]}</td>`;
        row.forEach((val, j) => {
          const isDiag = i === j;
          const bgStyle = isDiag ? 'background: rgba(99, 102, 241, 0.25); color: #fff;' : 'color: var(--text-dim);';
          html += `<td><span class="cm-cell-val" style="${bgStyle}">${val}</span></td>`;
        });
        html += '</tr>';
      });

      html += '</tbody></table>';
      container.innerHTML = html;
    }

    // Init
    loadStats();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "")
    subject = data.get("subject", "")
    threshold = float(data.get("threshold", classifier.confidence_threshold))

    pred = classifier.predict(text=text, subject=subject, threshold=threshold)

    # Log to CSV / JSON for human review if flagged, or auto-routed
    logged_record = review_logger.log_ticket_prediction(pred)
    pred["review_id"] = logged_record["review_id"]
    pred["status"] = logged_record["status"]

    return jsonify(pred)


@app.route("/api/reviews", methods=["GET"])
def get_reviews():
    filter_type = request.args.get("filter", "pending")
    if filter_type == "all":
        records = review_logger.get_all_reviews()
    else:
        records = review_logger.get_pending_reviews()
    return jsonify(records)


@app.route("/api/reviews/<review_id>/resolve", methods=["POST"])
def resolve_review(review_id):
    data = request.get_json() or {}
    final_cat = data.get("final_category", "")
    final_urg = data.get("final_urgency", "")
    notes = data.get("notes", "")

    success = review_logger.resolve_review(review_id, final_cat, final_urg, notes)
    if success:
        return jsonify({"success": True, "review_id": review_id})
    return jsonify({"success": False, "error": "Review record not found"}), 404


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(review_logger.get_review_stats())


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found. Train models first."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI Ticket Manager Dashboard & API on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
