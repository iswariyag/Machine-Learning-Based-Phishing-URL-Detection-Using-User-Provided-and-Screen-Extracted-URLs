# src/gui_main.py
import sys
import os
import time
from datetime import datetime
import pandas as pd
import xgboost as xgb

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QFrame, QGraphicsDropShadowEffect,
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QAbstractItemView, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt6.QtGui import QFont, QColor, QCursor

# --- BACKEND IMPORTS ---
from feature_extractor import get_url_features, feature_names
from notifications import DynamicIsland
from screen_monitor import PhishingMonitor
from api_verifier import check_virustotal

# --- PATHS ---
current_file_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_directory)
MODELS_DIR = os.path.join(project_root, 'models')

# --- THEME CONSTANTS ---
MACOS_THEME = """
QWidget { 
    font-family: 'SF Pro Display', 'Segoe UI', sans-serif; 
    color: #FFFFFF; 
}

QWidget#MainWidget { 
    background-color: #121212; 
    border-radius: 12px; 
    border: 1px solid #333; 
}

QFrame#Sidebar { 
    background-color: #1a1a1a; 
    border-top-left-radius: 12px; 
    border-bottom-left-radius: 12px; 
    border-right: 1px solid #252525;
}

QPushButton[class="SidebarBtn"] {
    background-color: transparent; 
    color: #888; 
    text-align: left; 
    padding: 10px 20px;
    margin: 4px 15px;
    border-radius: 10px; 
    font-size: 14px; 
    border: none;
}

QPushButton[class="SidebarBtn"]:checked { 
    background-color: #007AFF; 
    color: white; 
    font-weight: bold; 
}

QFrame[class="DashCard"] { 
    background-color: #1e1e1e; 
    border-radius: 12px; 
    border: 1px solid #2d2d2d;
}

/* TERMINAL STYLE TABLE */
QTableWidget#EventStream {
    background-color: #050505; /* Pitch Black */
    border: 1px solid #333;
    border-radius: 8px;
    gridline-color: transparent;
    font-family: 'Consolas', 'Courier New', monospace; /* Hacker Font */
}

QLineEdit {
    background-color: #252525;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px;
    color: white;
}

QPushButton[class="ActionBtn"] {
    background-color: #333;
    border: 1px solid #444;
    color: white; 
    padding: 12px;
    border-radius: 8px; 
    font-weight: 600;
}
QPushButton[class="ActionBtn"]:hover { background-color: #444; }
"""

class PhishingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShieldAI Protection")
        self.setFixedSize(1100, 750)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint) 
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(MACOS_THEME)

        # --- BACKEND INIT ---
        self.manual_model = xgb.XGBClassifier()
        try:
            self.manual_model.load_model(os.path.join(MODELS_DIR, "url_classifier.json"))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # App State
        self.is_monitoring = False
        self.total_threats = 0
        self.old_pos = None
        self.worker = None

        # Notification Popup
        self.popup = DynamicIsland()

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.central_widget.setObjectName("MainWidget")
        self.setCentralWidget(self.central_widget)

        # Drop Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 200))
        self.central_widget.setGraphicsEffect(shadow)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setup_sidebar()
        self.setup_content_area()

    def setup_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(230)
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(0, 20, 0, 20)

        # Traffic Lights
        lights = QHBoxLayout()
        lights.setContentsMargins(20, 0, 0, 10)
        lights.setSpacing(8)
        for c, f in [("#FF5F56", self.close), ("#FFBD2E", self.showMinimized), ("#27C93F", lambda: None)]:
            btn = QPushButton()
            btn.setFixedSize(12, 12)
            btn.setStyleSheet(f"background-color: {c}; border-radius: 6px; border: none;")
            btn.clicked.connect(f)
            lights.addWidget(btn)
        lights.addStretch()
        layout.addLayout(lights)

        # Logo
        logo = QLabel("ShieldAI")
        logo.setStyleSheet("font-size: 22px; font-weight: bold; padding: 20px;")
        layout.addWidget(logo)

        # Navigation
        self.btn_dashboard = self.create_nav_btn("Dashboard", 0)
        self.btn_deepscan = self.create_nav_btn("Deep Scan", 1)
        layout.addWidget(self.btn_dashboard)
        layout.addWidget(self.btn_deepscan)

        layout.addStretch()
        self.main_layout.addWidget(self.sidebar)

    def create_nav_btn(self, text, index):
        btn = QPushButton(text)
        btn.setProperty("class", "SidebarBtn")
        btn.setCheckable(True)
        btn.setAutoExclusive(True)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(lambda: self.stack.setCurrentIndex(index))
        if index == 0: btn.setChecked(True)
        return btn

    def setup_content_area(self):
        self.stack = QStackedWidget()
        
        # --- Dashboard Page ---
        self.page_dash = QWidget()
        dash_layout = QVBoxLayout(self.page_dash)
        dash_layout.setContentsMargins(30, 40, 30, 30)
        dash_layout.setSpacing(20)

        # Stats Cards
        stats_row = QHBoxLayout()
        self.card_threats = self.create_stat_card("Threats Blocked", "0", "#FF453A")
        self.card_status = self.create_stat_card("Status", "Passive", "#888")
        stats_row.addWidget(self.card_threats)
        stats_row.addWidget(self.card_status)
        dash_layout.addLayout(stats_row)

        # Action Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Protection")
        self.btn_start.setProperty("class", "ActionBtn")
        self.btn_start.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_start.clicked.connect(self.toggle_guard)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setProperty("class", "ActionBtn")
        self.btn_stop.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_stop.clicked.connect(self.toggle_guard) # Reuse logic for toggle
        
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        dash_layout.addLayout(btn_row)

        # Terminal / Event Stream
        dash_layout.addWidget(QLabel("Live Event Stream:", styleSheet="color: #ccc; font-weight: bold; font-family: Segoe UI;"))
        self.history_table = QTableWidget(0, 1)
        self.history_table.setObjectName("EventStream")
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.horizontalHeader().setVisible(False)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        dash_layout.addWidget(self.history_table)

        self.stack.addWidget(self.page_dash)

        # --- Deep Scan Page ---
        self.page_scan = QWidget()
        scan_layout = QVBoxLayout(self.page_scan)
        scan_layout.setContentsMargins(40, 60, 40, 40)
        
        scan_card = QFrame()
        scan_card.setProperty("class", "DashCard")
        sc_layout = QVBoxLayout(scan_card)
        sc_layout.setContentsMargins(30, 30, 30, 30)
        
        sc_layout.addWidget(QLabel("Enter URL for Analysis:", styleSheet="color: #888;"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com")
        sc_layout.addWidget(self.url_input)
        
        btn_run = QPushButton("Analyze Now")
        btn_run.setProperty("class", "ActionBtn")
        btn_run.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_run.clicked.connect(self.run_deep_scan)
        sc_layout.addWidget(btn_run)

        self.lbl_scan_res = QLabel("")
        self.lbl_scan_res.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_scan_res.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 20px;")
        sc_layout.addWidget(self.lbl_scan_res)

        scan_layout.addWidget(scan_card)
        scan_layout.addStretch()
        self.stack.addWidget(self.page_scan)

        self.main_layout.addWidget(self.stack)

    def create_stat_card(self, title, value, color):
        card = QFrame()
        card.setProperty("class", "DashCard")
        layout = QVBoxLayout(card)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #888; font-size: 12px; text-transform: uppercase;")
        v_lbl = QLabel(value)
        v_lbl.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold;")
        layout.addWidget(t_lbl)
        layout.addWidget(v_lbl)
        
        # Save reference for updates
        if title == "Status": self.status_label = v_lbl
        else: self.threat_label = v_lbl
        return card

    # --- HELPER: LOGGING WITH HTML COLORING ---
    def add_log(self, url, reason, is_phishing):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        
        # --- COLOR CODING ---
        time_col = "#555"
        
        if is_phishing:
            tag = "PHISHING LINK"
            tag_col = "#FF453A" # Red
            url_col = "#30D158" # Green (Matches your image)
            reason_col = "#30D158" # Green
        else:
            tag = "SAFE LINK"
            tag_col = "#30D158" # Green
            url_col = "#888"
            reason_col = "#666"

        # HTML Styling for Rich Text inside Table
        html_text = f"""
        <div style="font-family: Consolas; font-size: 13px; line-height: 140%;">
            <span style="color:{time_col};">{timestamp}</span> 
            <span style="color:{tag_col}; font-weight:bold;">{tag}:</span> 
            <span style="color:{url_col};">{url}</span>
            <br>
            <span style="color:{reason_col}; padding-left: 20px;">↳ Reason: {reason}</span>
        </div>
        """
        
        label = QLabel()
        label.setText(html_text)
        label.setStyleSheet("padding: 2px;")
        
        self.history_table.setCellWidget(row, 0, label)
        self.history_table.resizeRowToContents(row)
        self.history_table.scrollToBottom()

    # --- BACKEND LOGIC: REAL-TIME GUARD ---
    def toggle_guard(self):
        sender = self.sender()
        
        if sender == self.btn_start and not self.is_monitoring:
            # START
            self.is_monitoring = True
            
            self.status_label.setText("Active")
            self.status_label.setStyleSheet("color: #30D158; font-size: 32px; font-weight: bold;")
            self.add_log("System Guard", "Real-time protection initialized.", False)
            
            self.thread = QThread()
            self.worker = PhishingMonitor()
            self.worker.moveToThread(self.thread)
            self.worker.alert_signal.connect(self.handle_auto_alert)
            self.thread.started.connect(self.worker.start_monitoring)
            self.thread.start()

        elif sender == self.btn_stop and self.is_monitoring:
            # STOP
            self.is_monitoring = False
            
            self.status_label.setText("Passive")
            self.status_label.setStyleSheet("color: #888; font-size: 32px; font-weight: bold;")
            self.add_log("System Guard", "Real-time protection stopped.", False)

            if self.worker:
                self.worker.stop_monitoring()
                self.thread.quit()
                self.thread.wait()

    def handle_auto_alert(self, title, message, explanation, color):
        self.total_threats += 1
        self.threat_label.setText(str(self.total_threats))
        
        url_detected = message.replace("Suspicious URL: ", "")
        self.add_log(url_detected, explanation, True)

        self.popup.show_message(title, message, explanation, color)

    # --- BACKEND LOGIC: DEEP SCAN ---
    # src/gui_main.py

    def run_deep_scan(self):
        url = self.url_input.text()
        if not url: return
        
        self.lbl_scan_res.setText("Analyzing...")
        self.lbl_scan_res.setStyleSheet("color: #FF9F0A; font-size: 24px; font-weight: bold;")
        QApplication.processEvents()

        # 1. Feature Extraction
        feats = get_url_features(url)
        if not feats:
            self.lbl_scan_res.setText("Invalid URL")
            self.lbl_scan_res.setStyleSheet("color: #888; font-size: 24px;")
            return

        # 2. Local AI Prediction
        df = pd.DataFrame([feats], columns=feature_names)
        prob = self.manual_model.predict_proba(df)[0][1]
        
        # 3. Cloud API Check
        self.lbl_scan_res.setText("Checking URl...")
        QApplication.processEvents()
        
        api_result = check_virustotal(url)
        
        # 4. Final Verdict Logic
        verdict = "Safe"
        color = "#30D158" # Green
        
        if api_result == 2:
            verdict = "CONFIRMED PHISHING "
            color = "#FF453A"
            prob = 0.99 
        elif api_result == 1:
            verdict = "Likely Phishing"
            color = "#FF9F0A"
            prob = max(prob, 0.85)
        elif api_result == -1: 
            # --- NEW: HANDLE API FAILURE ---
            if prob > 0.5:
                verdict = "Danger Detected "
                color = "#FF453A"
            else:
                verdict = "Unverified "
                color = "#888" # Grey, not Green!
        elif prob > 0.5:
            verdict = "Danger Detected "
            color = "#FF453A"
        else:
            verdict = "Safe (Verified)"
            prob = prob

        # 5. Display Result
        self.lbl_scan_res.setText(f"{verdict}")
        self.lbl_scan_res.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold; margin-top: 20px;")
        
        is_threat = prob > 0.5 or api_result >= 1
        self.add_log(url, "Manual Deep Scan Analysis", is_threat)
        
        if is_threat:
            self.total_threats += 1
            self.threat_label.setText(str(self.total_threats))

    # Window Dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.old_pos = event.globalPosition().toPoint()
    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()
    def mouseReleaseEvent(self, event):
        self.old_pos = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhishingApp()
    window.show()
    sys.exit(app.exec())