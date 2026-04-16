# src/feature_extractor.py
import re
from urllib.parse import urlparse
import ipaddress

# Define features globally so other files can import them
feature_names = [
    'url_length', 'hostname_length', 'count_dots', 'count_hyphens',
    'count_at', 'count_qmark', 'count_percent', 'is_ip',
    'is_http', 'dir_depth', 'has_sus_words', 'is_shortened'
]

def get_url_features(url):
    features = []
    
    # --- FIX: Default to HTTPS to avoid false positives on manual entry ---
    # If we force 'http://', the model thinks it's insecure (is_https=0) and flags it.
    # By forcing 'https://', we treat the input as secure by default unless known otherwise.
    if not re.match(r'^https?://', url):
        url = 'https://' + url 
    
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        path = parsed_url.path
    except:
        return None

    # --- 1. Lexical Features ---
    features.append(len(url))
    features.append(len(hostname))
    features.append(hostname.count('.'))
    features.append(hostname.count('-'))
    features.append(url.count('@'))
    features.append(url.count('?'))
    features.append(url.count('%'))

    # --- 2. Structural Features ---
    # Is IP?
    try:
        ipaddress.ip_address(hostname)
        features.append(1)
    except:
        features.append(0)

    # Is HTTPS? (0 = Safe/HTTPS, 1 = Risky/HTTP)
    if parsed_url.scheme == 'https':
        features.append(0) 
    else:
        features.append(1) 

    # Directory Depth
    features.append(path.count('/'))

    # Suspicious Words
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account', 'banking']
    if any(word in url.lower() for word in suspicious_keywords):
        features.append(1)
    else:
        features.append(0)
        
    # Shortener
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 'is.gd', 't.co']
    if any(s in hostname for s in shorteners):
        features.append(1)
    else:
        features.append(0)

    return features