# src/api_verifier.py
import requests
import base64

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR ACTUAL VIRUSTOTAL API KEY
API_KEY = '748c3c1e9c5a2c1740ce37b714462ae9391e9aff941779afd983689e0e40ec4a' 

def check_virustotal(url):
    """
    Returns:
        2 = Confirmed Malicious (High Risk)
        1 = Suspicious (Medium Risk)
        0 = Clean (Safe)
        -1 = API Error / Rate Limit (Fallback to Local AI)
    """
    if API_KEY == 'YOUR_VIRUSTOTAL_API_KEY_HERE':
        print("[!] Warning: No VirusTotal API Key configured.")
        return -1

    try:
        # VirusTotal requires URL to be base64 encoded
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        headers = {
            "accept": "application/json",
            "x-apikey": API_KEY
        }
        
        # Query the API
        response = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            stats = response.json()['data']['attributes']['last_analysis_stats']
            malicious = stats['malicious']
            suspicious = stats['suspicious']
            
            print(f" [API] VirusTotal Results -> Malicious: {malicious}, Suspicious: {suspicious}")
            
            if malicious >= 2:
                return 2 # CONFIRMED DANGER
            elif malicious == 1 or suspicious >= 1:
                return 1 # PROBABLY DANGER
            else:
                return 0 # FALSE POSITIVE (Safe)
                
        elif response.status_code == 404:
            # URL not in database yet -> Rely on Local AI
            return -1 
        else:
            return -1

    except Exception as e:
        print(f" [!] API Error: {e}")
        return -1