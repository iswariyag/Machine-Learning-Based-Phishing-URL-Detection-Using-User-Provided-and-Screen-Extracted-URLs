# src/screen_monitor.py
import time
import mss
import numpy as np
import pytesseract
import xgboost as xgb
import pickle
import re
import cv2
import os
from PyQt6.QtCore import QObject, pyqtSignal

# --- IMPORTS ---
from feature_extractor import get_url_features, feature_names
from api_verifier import check_virustotal  # <--- HYBRID VERIFICATION

# --- PATH CONFIGURATION ---
current_file_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_directory)
MODELS_DIR = os.path.join(project_root, 'models')

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PhishingMonitor(QObject):
    # SIGNAL: Sends (Title, Message, Explanation, Color) to GUI
    alert_signal = pyqtSignal(str, str, str, str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.alert_history = {} # Tracks when we last alerted for a specific URL

        print("Loading Real-Time Models...")
        self.url_model = xgb.XGBClassifier()
        self.url_model.load_model(os.path.join(MODELS_DIR, "url_classifier.json"))

    def start_monitoring(self):
        self.running = True
        sct = mss.mss()
        monitor = sct.monitors[1] # Primary monitor

        print("--- Real-Time Hybrid Monitoring Started ---")

        while self.running:
            try:
                # 1. Capture Screen & OCR
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                # --psm 6 assumes a block of text, good for URL lists
                text = pytesseract.image_to_string(gray, config='--psm 6')
                
                # 2. Extract URLs
                urls = re.findall(r'(?:www\.|https?://)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
                
                for url in urls:
                    clean_url = url.strip().lower()
                    
                    # Deduplication: Don't alert the same URL twice in 10 seconds
                    if time.time() - self.alert_history.get(clean_url, 0) < 10: 
                        continue

                    # --- LEVEL 1: LOCAL AI CHECK ---
                    feats = get_url_features(url)
                    if not feats: continue

                    import pandas as pd
                    df = pd.DataFrame([feats], columns=feature_names)
                    local_pred = self.url_model.predict(df)[0]

                    if local_pred == 1: # AI thinks it is Phishing
                        print(f" [?] Local AI flagged {url}. Checking API...")
                        
                        # --- LEVEL 2: CLOUD API CONFIRMATION ---
                        api_result = check_virustotal(url)
                        
                        should_alert = False
                        alert_color = "#FF453A"
                        alert_msg = "Critical Threat Detected"

                        if api_result == 2: # MALICIOUS
                            should_alert = True
                            alert_msg = "CONFIRMED PHISHING (VirusTotal Verified)"
                            alert_color = "#FF0000" # Deep Red

                        elif api_result == 1: # SUSPICIOUS
                            should_alert = True
                            alert_msg = "Suspicious Site Detected"
                            alert_color = "#FF9500" # Orange

                        elif api_result == 0: # CLEAN
                            # API says Safe -> The Local AI was wrong.
                            print(f" [✓] False Positive Supressed: {url}")
                            should_alert = False 

                        else: 
                            # API ERROR / RATE LIMIT (-1)
                            # FIX: If API fails, ONLY alert if strictly malicious keywords exist.
                            # This prevents "web.whatsapp.com" from flagging when API is offline.
                            suspicious_keywords = ['login', 'verify', 'update', 'bank', 'secure', 'account']
                            if any(k in clean_url for k in suspicious_keywords):
                                should_alert = True
                                alert_msg = "Potential Phishing (Offline Detection)"
                                print(f" [!] API Failed. Fallback to AI because keyword found.")
                            else:
                                should_alert = False
                                print(f" [!] API Failed. Suppressing alert for '{url}' (No sensitive keywords).")

                        if should_alert:
                            self.alert_history[clean_url] = time.time()
                            self.alert_signal.emit(
                                "PHISHING DETECTED", 
                                f"{url}",
                                alert_msg,
                                alert_color
                            )

                time.sleep(2) # Scan every 2 seconds
            except Exception as e:
                print(f"Monitor Error: {e}")
                time.sleep(2)

    def stop_monitoring(self):
        self.running = False