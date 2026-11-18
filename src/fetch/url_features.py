# src/fetch/url_features_enhanced.py
"""
Enhanced URL Feature Extraction for Malicious QR Code Detection
Extracts 68 lexical, structural, and statistical features from URLs

Features are grouped into:
1. Basic structural features (length, components) - 7 features
2. Character distribution features (digits, special chars, entropy) - 17 features
3. Domain-based features (TLD, subdomains, suspicious patterns) - 8 features
4. Path and query features (depth, parameters, encoding) - 6 features
5. Security indicator features (IP usage, obfuscation, brand impersonation) - 8 features
6. Statistical features (entropy, character variety) - 5 features
7. ADVANCED features (homoglyphs, brand similarity, URL shorteners, domain age indicators) - 9 features
8. PATTERN FEATURES (CMS exploits, nested TLDs, phishing paths) - 8 features

NEW FEATURES v2 (8 additional for better FN detection):
- cms_exploit_score: Detects CMS exploitation patterns (Joomla/WordPress/Drupal attacks)
- nested_tld_score: Detects suspicious TLD stacking (paypal.co.info.com.nz.au)
- phishing_path_score: Detects phishing path patterns (/login.php, /webscr.php)
- suspicious_filename_score: Detects suspicious filenames in path
- excessive_subdomain_score: Detects excessive subdomain depth (>4 levels)
- login_form_indicators: Detects login/form-related path components
- data_exfil_pattern_score: Detects data exfiltration URL patterns
- typosquat_distance_score: Improved typosquatting detection with edit distance
"""

from __future__ import annotations

import re
import math
import string
import difflib
import unicodedata
import requests
from urllib.parse import urlparse, parse_qs
from typing import Dict, Optional, Set
from datetime import datetime


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq',  # Freenom free domains
    'xyz', 'top', 'work', 'click', 'link',  # Commonly abused
    'date', 'racing', 'stream', 'download', 'win',
    'bid', 'science', 'party', 'review', 'trade'
}

# Known URL shortener domains
URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'adf.ly', 'bit.do', 'short.io',
    'rebrand.ly', 'cutt.ly', 'bl.ink', 'tiny.cc', 'shorturl.at'
}

# Brand names commonly used in phishing
BRAND_KEYWORDS = {
    'paypal', 'amazon', 'apple', 'microsoft', 'google',
    'facebook', 'netflix', 'instagram', 'twitter', 'linkedin',
    'bank', 'secure', 'account', 'verify', 'update',
    'login', 'signin', 'password', 'confirm', 'suspended'
}

# Suspicious keywords in URLs
SUSPICIOUS_KEYWORDS = {
    'login', 'signin', 'verify', 'update', 'secure', 'account',
    'banking', 'suspended', 'locked', 'confirm', 'urgent',
    'reset', 'restore', 'validate', 'credential', 'activate'
}

# Common file extensions in phishing
SUSPICIOUS_EXTENSIONS = {
    '.exe', '.zip', '.rar', '.scr', '.bat', '.cmd', '.com',
    '.pif', '.vbs', '.js', '.jar', '.apk'
}

# CMS exploitation patterns (Joomla, WordPress, Drupal attacks)
CMS_EXPLOIT_PATTERNS = [
    r'index\.php\?option=com_',  # Joomla component exploitation
    r'wp-admin', r'wp-includes', r'wp-content',  # WordPress
    r'administrator/index\.php',  # Joomla admin
    r'sites/default/files',  # Drupal
    r'xmlrpc\.php',  # WordPress XML-RPC
    r'admin\.php', r'config\.php',  # Generic CMS
    r'phpmyadmin', r'phpMyAdmin',  # Database admin panels
    r'cgi-bin',  # CGI directory
    r'\.htaccess', r'\.htpasswd',  # Apache config files
]

# Common phishing path patterns
PHISHING_PATH_PATTERNS = [
    r'/login\.php', r'/signin\.php', r'/webscr\.php',  # PayPal pattern
    r'/verification/', r'/verify/', r'/validate/',
    r'/secure/', r'/security/', r'/update/',
    r'/billing/', r'/account/', r'/confirm/',
    r'/password/', r'/reset/', r'/restore/',
    r'/banking/', r'/auth/', r'/authentication/',
    r'/suspended/', r'/locked/', r'/limited/',
    r'/invoice/', r'/payment/', r'/checkout/',
    r'/recover/', r'/reactivate/', r'/unlock/',
]

# Suspicious filenames commonly used in phishing
PHISHING_FILENAMES = {
    'login', 'signin', 'logon', 'authenticate', 'auth',
    'verify', 'verification', 'validate', 'confirm',
    'update', 'upgrade', 'renew', 'reactivate',
    'secure', 'security', 'protect', 'safety',
    'account', 'profile', 'settings', 'preferences',
    'billing', 'payment', 'invoice', 'receipt',
    'password', 'credential', 'access', 'unlock',
    'suspended', 'limited', 'restricted', 'blocked',
    'webscr', 'cgi', 'dispatch', 'redirect',
}

# Country-code TLDs (for nested TLD detection)
COUNTRY_CODE_TLDS = {
    'au', 'uk', 'nz', 'ca', 'de', 'fr', 'jp', 'cn', 'br', 'in',
    'ru', 'za', 'mx', 'ar', 'es', 'it', 'nl', 'be', 'ch', 'at',
    'pl', 'se', 'no', 'dk', 'fi', 'ie', 'sg', 'hk', 'tw', 'kr',
    'th', 'ph', 'my', 'id', 'vn', 'pk', 'eg', 'ng', 'ke', 'gh',
}

# Second-level domain patterns (for nested TLD detection)
SECOND_LEVEL_DOMAINS = {
    'co', 'com', 'org', 'net', 'gov', 'edu', 'ac', 'or', 'ne',
    'info', 'biz', 'mil', 'int', 'name', 'pro', 'mobi', 'tel',
}


# ============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ============================================================================

def url_features_enhanced(url: str) -> Dict[str, any]:
    """
    Extract comprehensive features from a URL.

    Args:
        url: The URL string to analyze

    Returns:
        Dictionary containing 60 features (51 original + 9 advanced)
    """
    if not url or not isinstance(url, str):
        return _empty_features()
    
    features = _empty_features()
    
    try:
        parsed = urlparse(url.strip())
        
        # Extract components
        scheme = parsed.scheme or ""
        domain = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""
        full_url = url.strip()
        
        # ===== 1. BASIC STRUCTURAL FEATURES =====
        features.update(_extract_basic_features(full_url, domain, path, query, scheme))
        
        # ===== 2. CHARACTER DISTRIBUTION FEATURES =====
        features.update(_extract_character_features(full_url, domain, path))
        
        # ===== 3. DOMAIN-BASED FEATURES =====
        features.update(_extract_domain_features(domain))
        
        # ===== 4. PATH AND QUERY FEATURES =====
        features.update(_extract_path_query_features(path, query))
        
        # ===== 5. SECURITY INDICATOR FEATURES =====
        features.update(_extract_security_features(full_url, domain, path, scheme))

        # ===== 6. STATISTICAL FEATURES =====
        features.update(_extract_statistical_features(full_url, domain, path))

        # ===== 7. ADVANCED FEATURES (NEW) =====
        features.update(_extract_advanced_features(full_url, domain, path, query))

        # ===== 8. PATTERN FEATURES (v2 - for better FN detection) =====
        features.update(_extract_pattern_features(full_url, domain, path, query))

    except Exception as e:
        # If parsing fails, return empty features
        pass

    return features


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def _extract_basic_features(url: str, domain: str, path: str, query: str, scheme: str) -> Dict:
    """Basic length and component features"""
    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'path_length': len(path),
        'query_length': len(query),
        'scheme_is_https': 1 if scheme == 'https' else 0,
        'has_port': 1 if ':' in domain and not domain.startswith('[') else 0,
        'url_has_fragment': 1 if '#' in url else 0,
    }


def _extract_character_features(url: str, domain: str, path: str) -> Dict:
    """Character distribution and composition features"""
    # Count different character types
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    num_special = sum(not c.isalnum() and c not in '/:.-_' for c in url)
    
    # Ratios (avoid division by zero)
    url_len = len(url)
    digit_ratio = num_digits / url_len if url_len > 0 else 0
    letter_ratio = num_letters / url_len if url_len > 0 else 0
    special_ratio = num_special / url_len if url_len > 0 else 0
    
    # Specific character counts
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_underscores = url.count('_')
    num_slashes = url.count('/')
    num_question_marks = url.count('?')
    num_equals = url.count('=')
    num_ampersands = url.count('&')
    num_at_signs = url.count('@')
    num_percent = url.count('%')
    
    # Domain-specific character features
    domain_num_digits = sum(c.isdigit() for c in domain)
    
    return {
        'num_digits': num_digits,
        'num_letters': num_letters,
        'num_special_chars': num_special,
        'digit_letter_ratio': digit_ratio / (letter_ratio + 1e-10),  # Avoid div by 0
        'digit_ratio': digit_ratio,
        'letter_ratio': letter_ratio,
        'special_char_ratio': special_ratio,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_slashes': num_slashes,
        'num_question_marks': num_question_marks,
        'num_equals': num_equals,
        'num_ampersands': num_ampersands,
        'num_at_signs': num_at_signs,
        'num_percent': num_percent,
        'domain_has_digits': 1 if domain_num_digits > 0 else 0,
    }


def _extract_domain_features(domain: str) -> Dict:
    """Domain structure and reputation features"""
    # Remove port if present
    domain_clean = domain.split(':')[0] if ':' in domain else domain
    
    # TLD extraction
    tld = domain_clean.split('.')[-1] if '.' in domain_clean else ''
    
    # Subdomain count
    parts = domain_clean.split('.')
    num_subdomains = max(0, len(parts) - 2) if len(parts) >= 2 else 0
    
    # IP address detection (IPv4 pattern)
    is_ip_address = bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain_clean))
    
    # Suspicious patterns
    has_suspicious_tld = 1 if tld.lower() in SUSPICIOUS_TLDS else 0
    is_url_shortener = 1 if domain_clean.lower() in URL_SHORTENERS else 0
    
    # Homograph/internationalized domain detection (punycode)
    has_punycode = 1 if 'xn--' in domain_clean.lower() else 0
    
    # Domain contains suspicious patterns
    has_double_slash_in_domain = 1 if '//' in domain else 0
    
    # Vowel ratio in domain (random domains often have low vowel ratios)
    vowels = sum(1 for c in domain_clean.lower() if c in 'aeiou')
    consonants = sum(1 for c in domain_clean.lower() if c.isalpha() and c not in 'aeiou')
    vowel_ratio = vowels / (vowels + consonants) if (vowels + consonants) > 0 else 0
    
    return {
        'tld_length': len(tld),
        'num_subdomains': num_subdomains,
        'is_ip_address': 1 if is_ip_address else 0,
        'has_suspicious_tld': has_suspicious_tld,
        'is_url_shortener': is_url_shortener,
        'has_punycode': has_punycode,
        'domain_vowel_ratio': vowel_ratio,
        'domain_has_double_slash': has_double_slash_in_domain,
    }


def _extract_path_query_features(path: str, query: str) -> Dict:
    """Path and query string features"""
    # Path depth
    path_depth = len([p for p in path.split('/') if p]) if path else 0
    
    # Query parameters
    query_params = parse_qs(query) if query else {}
    num_query_params = len(query_params)
    
    # Suspicious query parameter names
    suspicious_param_names = {'token', 'session', 'id', 'user', 'pass', 'login', 'auth', 'key'}
    has_suspicious_params = 1 if any(
        param.lower() in suspicious_param_names 
        for param in query_params.keys()
    ) else 0
    
    # URL encoding in path/query (excessive encoding is suspicious)
    num_encoded_chars = path.count('%') + query.count('%')
    
    # File extension in path
    has_file_extension = 1 if re.search(r'\.[a-zA-Z0-9]{2,4}$', path) else 0
    has_suspicious_extension = 1 if any(
        path.lower().endswith(ext) for ext in SUSPICIOUS_EXTENSIONS
    ) else 0
    
    return {
        'path_depth': path_depth,
        'num_query_params': num_query_params,
        'has_suspicious_params': has_suspicious_params,
        'num_encoded_chars': num_encoded_chars,
        'has_file_extension': has_file_extension,
        'has_suspicious_extension': has_suspicious_extension,
    }


def _extract_security_features(url: str, domain: str, path: str, scheme: str) -> Dict:
    """Security-related indicators"""
    url_lower = url.lower()
    domain_lower = domain.lower()
    
    # Brand impersonation detection
    has_brand_keyword = 1 if any(brand in url_lower for brand in BRAND_KEYWORDS) else 0
    
    # Suspicious keyword detection
    has_suspicious_keyword = 1 if any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS) else 0
    
    # Obfuscation indicators
    has_hex_encoding = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0
    has_unicode_encoding = 1 if '\\u' in url or '\\x' in url else 0
    
    # Base64-like strings (long alphanumeric sequences)
    has_long_alphanumeric = 1 if re.search(r'[a-zA-Z0-9]{50,}', url) else 0
    
    # @ symbol before domain (URL spoofing technique)
    has_at_in_url = 1 if '@' in url else 0
    
    # Double extensions (e.g., file.pdf.exe)
    has_double_extension = 1 if re.search(r'\.[a-z]{2,4}\.[a-z]{2,4}', path.lower()) else 0
    
    # Mixed case in domain (unusual for legitimate sites)
    has_mixed_case_domain = 1 if domain != domain.lower() and domain != domain.upper() else 0
    
    return {
        'has_brand_keyword': has_brand_keyword,
        'has_suspicious_keyword': has_suspicious_keyword,
        'has_hex_encoding': has_hex_encoding,
        'has_unicode_encoding': has_unicode_encoding,
        'has_long_alphanumeric': has_long_alphanumeric,
        'has_at_in_url': has_at_in_url,
        'has_double_extension': has_double_extension,
        'has_mixed_case_domain': has_mixed_case_domain,
    }


def _extract_statistical_features(url: str, domain: str, path: str) -> Dict:
    """Statistical and entropy-based features"""
    # Shannon entropy (measure of randomness)
    url_entropy = _shannon_entropy(url)
    domain_entropy = _shannon_entropy(domain)
    path_entropy = _shannon_entropy(path) if path else 0

    # Longest consecutive character run
    max_consecutive = _max_consecutive_chars(url)

    # Character variety (unique chars / total chars)
    char_variety = len(set(url)) / len(url) if len(url) > 0 else 0

    return {
        'url_entropy': url_entropy,
        'domain_entropy': domain_entropy,
        'path_entropy': path_entropy,
        'max_consecutive_chars': max_consecutive,
        'char_variety': char_variety,
    }


def _extract_advanced_features(url: str, domain: str, path: str, query: str) -> Dict:
    """Advanced malicious URL detection features (9 features: 6 original + 3 new)"""
    return {
        'homoglyph_score': _extract_homoglyph_score(domain),
        'brand_similarity_score': _extract_brand_similarity_score(domain),
        'subdomain_suspicion_score': _extract_subdomain_suspicion_score(domain),
        'url_complexity_score': _extract_url_complexity_score(url, domain, path, query),
        'redirect_pattern_score': _extract_redirect_pattern_score(url, query, path),
        'domain_impersonation_score': _extract_domain_impersonation_score(domain, path),
        'unicode_homoglyph_score': _extract_unicode_homoglyph_score(domain),
        'url_shortener_risk_score': _extract_url_shortener_risk_score(url, domain),
        'new_domain_risk_score': _extract_new_domain_risk_score(domain),
    }


def _extract_homoglyph_score(domain: str) -> float:
    """
    Detect homoglyph attacks (character substitution).

    Common homoglyphs:
    - 0 (zero) vs O (letter o)
    - 1 (one) vs l (lowercase L) vs I (uppercase i)
    - rn (r+n) vs m
    - vv vs w
    - cl vs d

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_lower = domain.lower()
    score = 0.0

    # Common homoglyph patterns
    homoglyph_pairs = [
        ('0', 'o'),  # zero vs letter o
        ('1', 'l'), ('1', 'i'),  # one vs lowercase L vs uppercase i
        ('5', 's'),
        ('8', 'b'),
        ('rn', 'm'),  # consecutive rn looks like m
        ('vv', 'w'),  # consecutive vv looks like w
        ('cl', 'd'),  # consecutive cl looks like d
    ]

    # Check for suspicious character combinations
    for char1, char2 in homoglyph_pairs:
        if char1 in domain_lower or char2 in domain_lower:
            # If domain contains digits mixed with letters, more suspicious
            if any(c.isdigit() for c in domain) and any(c.isalpha() for c in domain):
                score += 0.15

    # Check for excessive digit-letter mixing (e.g., g00gle, paypa1)
    transitions = 0
    for i in range(len(domain) - 1):
        if domain[i].isdigit() != domain[i+1].isdigit():
            transitions += 1

    # Normalize by domain length
    if len(domain) > 0:
        transition_rate = transitions / len(domain)
        score += min(transition_rate * 2, 0.4)  # Cap at 0.4

    # Check for confusable patterns in brand context
    for brand in ['paypal', 'google', 'amazon', 'apple', 'microsoft']:
        if brand in domain_lower:
            # If brand name has digit substitutions, very suspicious
            if any(c.isdigit() for c in domain_lower):
                score += 0.3
                break

    return min(score, 1.0)


def _extract_brand_similarity_score(domain: str) -> float:
    """
    Detect typosquatting using edit distance to known brands.

    Checks if domain is suspiciously similar to popular brands.
    Uses Levenshtein distance (via difflib.SequenceMatcher).

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    # Extended brand list for typosquatting detection
    major_brands = [
        'google', 'facebook', 'amazon', 'apple', 'microsoft',
        'paypal', 'netflix', 'instagram', 'twitter', 'linkedin',
        'youtube', 'walmart', 'ebay', 'chase', 'wellsfargo',
        'bankofamerica', 'citibank', 'americanexpress', 'discover',
        'dropbox', 'github', 'reddit', 'tiktok', 'snapchat'
    ]

    domain_lower = domain.lower().split(':')[0]  # Remove port
    domain_parts = domain_lower.split('.')

    max_similarity = 0.0

    # Check each part of the domain against brand list
    for part in domain_parts:
        if len(part) < 3:  # Skip very short parts like 'co', 'uk'
            continue

        for brand in major_brands:
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, part, brand).ratio()

            # If very similar but not exact match, suspicious
            if 0.7 <= similarity < 1.0:
                max_similarity = max(max_similarity, similarity)

            # Check for insertion/deletion of 1 character
            if abs(len(part) - len(brand)) == 1 and similarity > 0.8:
                max_similarity = max(max_similarity, 0.9)

    # Convert similarity to suspicion score
    # High similarity to brand but not exact = suspicious
    if max_similarity >= 0.85:
        return min((max_similarity - 0.7) / 0.3, 1.0)  # Map 0.7-1.0 to 0.0-1.0

    return 0.0


def _extract_subdomain_suspicion_score(domain: str) -> float:
    """
    Detect suspicious subdomain patterns.

    Indicators:
    - Excessive subdomain depth (>3 levels)
    - Very long subdomain names (>20 chars)
    - Random-looking subdomains (high entropy, low vowel ratio)
    - Subdomains impersonating domains (e.g., paypal.evil.com)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0]  # Remove port
    parts = domain_clean.split('.')

    if len(parts) < 3:
        return 0.0  # No subdomains

    score = 0.0

    # 1. Excessive subdomain depth
    num_subdomains = len(parts) - 2
    if num_subdomains > 3:
        score += 0.3
    elif num_subdomains > 2:
        score += 0.15

    # 2. Check subdomain lengths
    subdomains = parts[:-2]  # All but last two parts (domain.tld)
    for subdomain in subdomains:
        if len(subdomain) > 20:
            score += 0.2
        elif len(subdomain) > 15:
            score += 0.1

    # 3. Check for random-looking subdomains (high entropy, low vowel ratio)
    for subdomain in subdomains:
        if len(subdomain) > 5:
            entropy = _shannon_entropy(subdomain)
            if entropy > 3.5:  # High randomness
                score += 0.15

            vowels = sum(1 for c in subdomain.lower() if c in 'aeiou')
            vowel_ratio = vowels / len(subdomain)
            if vowel_ratio < 0.2:  # Very low vowel ratio
                score += 0.1

    # 4. Brand impersonation in subdomain (e.g., paypal.evil.com)
    brand_keywords_extended = {
        'paypal', 'amazon', 'google', 'apple', 'microsoft',
        'facebook', 'bank', 'secure', 'login', 'verify'
    }

    for subdomain in subdomains:
        subdomain_lower = subdomain.lower()
        if any(brand in subdomain_lower for brand in brand_keywords_extended):
            # Brand in subdomain but not in main domain = suspicious
            main_domain = '.'.join(parts[-2:]).lower()
            if not any(brand in main_domain for brand in brand_keywords_extended):
                score += 0.3

    return min(score, 1.0)


def _extract_url_complexity_score(url: str, domain: str, path: str, query: str) -> float:
    """
    Aggregate complexity score to distinguish legitimate vs malicious complexity.

    Malicious URLs often have:
    - Very long URLs with random parameters
    - Excessive URL encoding
    - Deep path structures with suspicious patterns
    - Many query parameters with gibberish values

    Returns score 0.0-1.0 (higher = more suspicious complexity)
    """
    if not url:
        return 0.0

    score = 0.0

    # 1. URL length complexity
    url_len = len(url)
    if url_len > 150:
        score += 0.2
    elif url_len > 100:
        score += 0.1

    # 2. Path depth complexity
    path_depth = len([p for p in path.split('/') if p]) if path else 0
    if path_depth > 5:
        score += 0.15
    elif path_depth > 3:
        score += 0.05

    # 3. Query parameter complexity
    num_params = query.count('&') + 1 if query else 0
    if num_params > 10:
        score += 0.2
    elif num_params > 5:
        score += 0.1

    # 4. URL encoding density
    encoding_density = url.count('%') / len(url) if len(url) > 0 else 0
    if encoding_density > 0.1:  # >10% of chars are encoded
        score += 0.2
    elif encoding_density > 0.05:
        score += 0.1

    # 5. Entropy vs length ratio (random long URLs)
    url_entropy = _shannon_entropy(url)
    if url_len > 80 and url_entropy > 4.5:
        score += 0.15

    # 6. Special character density
    special_chars = sum(1 for c in url if not c.isalnum() and c not in '/:.-')
    special_density = special_chars / len(url) if len(url) > 0 else 0
    if special_density > 0.3:
        score += 0.1

    return min(score, 1.0)


def _extract_redirect_pattern_score(url: str, query: str, path: str) -> float:
    """
    Detect redirect patterns without making network calls.

    Indicators:
    - Query parameters like: url=, link=, goto=, redirect=, next=, return=
    - Double slashes in path (protocol confusion)
    - Multiple http/https in URL (nested redirects)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not url:
        return 0.0

    score = 0.0
    url_lower = url.lower()

    # 1. Redirect parameter detection
    redirect_params = ['url=', 'link=', 'goto=', 'redirect=', 'next=',
                      'return=', 'continue=', 'destination=', 'redir=', 'target=']

    for param in redirect_params:
        if param in url_lower:
            score += 0.25

            # Extra suspicious if the parameter contains http/https
            param_start = url_lower.find(param) + len(param)
            if param_start < len(url_lower):
                param_value = url_lower[param_start:param_start+100]
                if 'http' in param_value:
                    score += 0.3  # Redirect to external URL

    # 2. Double slash in path (not protocol ://)
    if '//' in path:
        score += 0.2

    # 3. Multiple protocols in URL (nested redirects)
    http_count = url_lower.count('http://')
    https_count = url_lower.count('https://')
    if (http_count + https_count) > 1:
        score += 0.4  # Very suspicious

    # 4. URL-encoded URL in query (obfuscated redirect)
    if '%2F%2F' in url or '%3A%2F%2F' in url:  # Encoded "//" or "://"
        score += 0.2

    return min(score, 1.0)


def _extract_domain_impersonation_score(domain: str, path: str) -> float:
    """
    Enhanced brand impersonation detection.

    Detects:
    - Brand names in domain with suspicious TLDs
    - Brand names in path/subdomain but not main domain
    - Multiple brand names in one URL (e.g., paypal-amazon-verify.tk)
    - Brand + suspicious keywords (e.g., paypal-login-verify.com)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    score = 0.0
    domain_lower = domain.lower().split(':')[0]  # Remove port
    path_lower = path.lower() if path else ''

    # Major brands that are commonly impersonated
    major_brands = {
        'paypal', 'amazon', 'apple', 'microsoft', 'google',
        'facebook', 'netflix', 'instagram', 'bank', 'chase',
        'wellsfargo', 'citibank', 'discover', 'americanexpress'
    }

    # Count brand mentions in domain
    brand_count = sum(1 for brand in major_brands if brand in domain_lower)

    # 1. Multiple brands in domain (e.g., paypal-amazon.com)
    if brand_count > 1:
        score += 0.5

    # 2. Brand + suspicious TLD
    if brand_count >= 1:
        tld = domain_lower.split('.')[-1] if '.' in domain_lower else ''
        if tld in SUSPICIOUS_TLDS:
            score += 0.4

    # 3. Brand + suspicious keywords in domain
    suspicious_combos = ['login', 'verify', 'secure', 'account', 'update',
                        'confirm', 'validate', 'signin', 'suspended']

    for brand in major_brands:
        if brand in domain_lower:
            for keyword in suspicious_combos:
                if keyword in domain_lower:
                    score += 0.3
                    break

    # 4. Brand in path but not in domain (e.g., evil.com/paypal/login)
    brand_in_path = any(brand in path_lower for brand in major_brands)
    brand_in_domain = any(brand in domain_lower for brand in major_brands)

    if brand_in_path and not brand_in_domain:
        score += 0.25

    # 5. Hyphen-heavy brand mentions (e.g., pay-pal, amaz0n)
    for brand in major_brands:
        if brand in domain_lower:
            # Count hyphens near brand name
            brand_pos = domain_lower.find(brand)
            surrounding = domain_lower[max(0, brand_pos-2):brand_pos+len(brand)+2]
            if surrounding.count('-') >= 2:
                score += 0.2

    return min(score, 1.0)


def _extract_unicode_homoglyph_score(domain: str) -> float:
    """
    Detect Unicode homograph attacks (Cyrillic, Greek, other non-Latin characters).

    Examples of Unicode homograph attacks:
    - paypaⅼ.com (uses Cyrillic 'l' instead of Latin 'l')
    - gооgle.com (uses Cyrillic 'о' instead of Latin 'o')
    - аpple.com (uses Cyrillic 'а' instead of Latin 'a')
    - microsоft.com (uses Cyrillic 'о' instead of Latin 'o')

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0].lower()  # Remove port
    score = 0.0

    # Check for non-ASCII characters
    non_ascii_count = sum(1 for c in domain_clean if ord(c) > 127)

    if non_ascii_count == 0:
        return 0.0  # All ASCII, no Unicode homoglyphs

    # Calculate ratio of non-ASCII to total characters
    non_ascii_ratio = non_ascii_count / len(domain_clean)

    # Suspicious Unicode character ranges
    # Cyrillic: U+0400 to U+04FF
    # Greek: U+0370 to U+03FF
    # Latin lookalikes in other scripts
    suspicious_ranges = [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0370, 0x03FF),  # Greek
        (0x0530, 0x058F),  # Armenian
        (0x10A0, 0x10FF),  # Georgian
    ]

    suspicious_char_count = 0
    for char in domain_clean:
        char_code = ord(char)
        for start, end in suspicious_ranges:
            if start <= char_code <= end:
                suspicious_char_count += 1
                break

    if suspicious_char_count > 0:
        score += 0.5  # Base score for having suspicious Unicode chars

        # If domain contains both ASCII and suspicious Unicode (mixed script)
        ascii_count = sum(1 for c in domain_clean if ord(c) < 128)
        if ascii_count > 0 and suspicious_char_count > 0:
            score += 0.3  # Mixed scripts are highly suspicious

        # Check if domain looks like a brand with Unicode substitution
        # Remove suspicious chars and check if it matches a brand
        ascii_only = ''.join(c for c in domain_clean if ord(c) < 128)
        major_brands = ['paypal', 'google', 'amazon', 'apple', 'microsoft',
                       'facebook', 'netflix', 'bank', 'chase']

        for brand in major_brands:
            if brand in ascii_only or len(ascii_only) > 0:
                # If the domain has Unicode chars and resembles a brand, very suspicious
                similarity = difflib.SequenceMatcher(None, ascii_only, brand).ratio()
                if similarity > 0.6:
                    score += 0.4
                    break

    # Punycode (xn--) is a form of Unicode encoding
    if 'xn--' in domain_clean:
        score += 0.3

        # Try to decode punycode and check for suspicious patterns
        try:
            parts = domain_clean.split('.')
            for part in parts:
                if part.startswith('xn--'):
                    decoded = part.encode('ascii').decode('idna')
                    # Check if decoded contains suspicious scripts
                    for char in decoded:
                        if ord(char) > 127:
                            score += 0.1
                            break
        except Exception:
            # If punycode decoding fails, even more suspicious
            score += 0.2

    return min(score, 1.0)


def _extract_url_shortener_risk_score(url: str, domain: str) -> float:
    """
    Assess risk from URL shorteners without making network calls.

    URL shorteners hide the real destination, making them useful for phishing.
    This function assigns a risk score based on:
    - Whether the URL is a known shortener
    - Additional suspicious patterns in the shortened link

    NOTE: This does NOT expand the URL (no network calls for offline training).
    For real-time expansion, use expand_url_shortener() during inference.

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not url or not domain:
        return 0.0

    domain_clean = domain.split(':')[0].lower()
    score = 0.0

    # Check if it's a known URL shortener
    if domain_clean in URL_SHORTENERS:
        score += 0.3  # Base risk for using a shortener

        # Additional risk factors:
        # 1. Very short path (easier to type-squat)
        parsed = urlparse(url)
        path = parsed.path or ""
        if len(path) <= 5:  # Very short shortened URL
            score += 0.1

        # 2. Path contains suspicious patterns (random-looking)
        if path and len(path) > 1:
            path_clean = path.lstrip('/')
            entropy = _shannon_entropy(path_clean)
            if entropy > 3.5:  # High entropy = random-looking
                score += 0.2

        # 3. Shortener + query parameters (unusual)
        if parsed.query:
            score += 0.15

    # Check for lesser-known or custom shorteners (high risk)
    # Pattern: short domain (≤6 chars) + short path (≤6 chars)
    domain_parts = domain_clean.split('.')
    if len(domain_parts) >= 2:
        domain_name = domain_parts[0]
        if len(domain_name) <= 6:
            parsed = urlparse(url)
            path = (parsed.path or "").lstrip('/')
            if 1 <= len(path) <= 6:
                score += 0.25  # Custom/unknown shortener pattern

    return min(score, 1.0)


def _extract_new_domain_risk_score(domain: str) -> float:
    """
    Assess risk based on domain age patterns WITHOUT making WHOIS queries.

    Since we can't check domain age offline, this function looks for
    indicators that suggest a newly registered domain:
    - Domain registration patterns in the URL
    - Suspicious TLD commonly used for disposable domains
    - Domain structure typical of temporary/throwaway sites

    NOTE: This does NOT query WHOIS (no network calls for offline training).
    For real domain age lookup, use check_domain_age_days() during inference.

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0].lower()
    score = 0.0

    # 1. Suspicious TLDs often used for temporary/newly registered domains
    tld = domain_clean.split('.')[-1] if '.' in domain_clean else ''

    # High-risk TLDs (free/cheap, often newly registered)
    high_risk_tlds = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'win'}
    if tld in high_risk_tlds:
        score += 0.4

    # Medium-risk TLDs
    medium_risk_tlds = {'click', 'link', 'work', 'download', 'bid', 'racing'}
    if tld in medium_risk_tlds:
        score += 0.2

    # 2. Random-looking domain names (likely newly generated)
    domain_parts = domain_clean.split('.')
    if len(domain_parts) >= 2:
        domain_name = domain_parts[0]

        # High entropy = random-looking (e.g., x8k2jf9s.tk)
        if len(domain_name) > 5:
            entropy = _shannon_entropy(domain_name)
            if entropy > 3.8:
                score += 0.3

        # Very long domain names (often randomly generated)
        if len(domain_name) > 20:
            score += 0.2

        # Low vowel ratio (hard to pronounce, likely random)
        vowels = sum(1 for c in domain_name if c in 'aeiou')
        vowel_ratio = vowels / len(domain_name) if len(domain_name) > 0 else 0
        if vowel_ratio < 0.15 and len(domain_name) > 5:
            score += 0.2

    # 3. IP addresses instead of domain names (no domain registration)
    if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain_clean):
        score += 0.3  # Using IP directly, no domain age

    # 4. Subdomain-heavy structure (e.g., lots of subdomains on free hosting)
    num_subdomains = max(0, len(domain_parts) - 2)
    if num_subdomains > 3:
        score += 0.15

    return min(score, 1.0)


# ============================================================================
# PATTERN FEATURES (v2 - for better False Negative detection)
# ============================================================================

def _extract_pattern_features(url: str, domain: str, path: str, query: str) -> Dict:
    """Extract 8 new pattern-based features for better malicious URL detection"""
    return {
        'cms_exploit_score': _extract_cms_exploit_score(url, path, query),
        'nested_tld_score': _extract_nested_tld_score(domain),
        'phishing_path_score': _extract_phishing_path_score(path),
        'suspicious_filename_score': _extract_suspicious_filename_score(path),
        'excessive_subdomain_score': _extract_excessive_subdomain_score(domain),
        'login_form_indicators': _extract_login_form_indicators(url, path, query),
        'data_exfil_pattern_score': _extract_data_exfil_pattern_score(url, query),
        'typosquat_distance_score': _extract_typosquat_distance_score(domain),
    }


def _extract_cms_exploit_score(url: str, path: str, query: str) -> float:
    """
    Detect CMS exploitation patterns (Joomla, WordPress, Drupal attacks).

    These patterns often appear in compromised websites used for phishing.
    Examples:
    - index.php?option=com_user&task=register (Joomla exploit)
    - wp-admin/admin-ajax.php (WordPress)
    - sites/default/files/malicious.php (Drupal)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not url:
        return 0.0

    score = 0.0
    url_lower = url.lower()

    # Check each CMS exploit pattern
    for pattern in CMS_EXPLOIT_PATTERNS:
        if re.search(pattern, url_lower, re.IGNORECASE):
            score += 0.3

    # Additional indicators of CMS exploitation
    # PHP files in query parameters (code injection)
    if '.php' in query.lower():
        score += 0.2

    # Base64-encoded parameters (often used in exploits)
    if re.search(r'[a-zA-Z0-9+/]{50,}={0,2}', query):
        score += 0.15

    # Multiple nested PHP includes
    if url_lower.count('.php') > 2:
        score += 0.2

    # Suspicious parameter names used in CMS exploits
    exploit_params = ['cmd', 'exec', 'system', 'shell', 'eval', 'base64', 'decode']
    for param in exploit_params:
        if param in url_lower:
            score += 0.25

    return min(score, 1.0)


def _extract_nested_tld_score(domain: str) -> float:
    """
    Detect suspicious nested/stacked TLD patterns.

    Attackers create domains like:
    - paypal.co.info.com.nz.au (multiple TLDs stacked)
    - secure-login.com.tk (brand + legitimate TLD + suspicious TLD)
    - amazon.co.uk.verify.xyz (legitimate domain pattern as subdomain)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0].lower()  # Remove port
    parts = domain_clean.split('.')

    if len(parts) < 3:
        return 0.0  # Not enough parts to have nested TLDs

    score = 0.0

    # Count TLD-like patterns in the domain
    tld_count = 0
    for part in parts:
        if part in COUNTRY_CODE_TLDS or part in SECOND_LEVEL_DOMAINS:
            tld_count += 1

    # Multiple TLD-like patterns is suspicious
    if tld_count >= 3:
        score += 0.5
    elif tld_count >= 2 and len(parts) > 4:
        score += 0.3

    # Check for legitimate domain patterns used as subdomains
    # e.g., paypal.com.evil.tk -> "paypal" and "com" appear before real domain
    legitimate_patterns = ['paypal', 'amazon', 'google', 'apple', 'microsoft',
                          'facebook', 'netflix', 'bank', 'secure']

    for i, part in enumerate(parts[:-2]):  # Exclude last two parts (real domain.tld)
        if part in legitimate_patterns:
            # Brand name in subdomain position
            score += 0.3
        if part in ['com', 'org', 'net', 'co']:
            # TLD-like part in middle of domain
            if i < len(parts) - 2:
                score += 0.2

    # Check for country code after generic TLD pattern (co.uk.tk, com.au.xyz)
    for i in range(len(parts) - 2):
        if parts[i] in SECOND_LEVEL_DOMAINS and parts[i+1] in COUNTRY_CODE_TLDS:
            # This looks like a legitimate ccTLD pattern, but there's more after it
            if i + 2 < len(parts) - 1:
                score += 0.4

    return min(score, 1.0)


def _extract_phishing_path_score(path: str) -> float:
    """
    Detect common phishing path patterns.

    Phishing URLs often contain paths like:
    - /login.php, /signin.php, /webscr.php
    - /verification/, /secure/, /update/
    - /billing/, /account/, /suspended/

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not path:
        return 0.0

    score = 0.0
    path_lower = path.lower()

    # Check each phishing path pattern
    pattern_count = 0
    for pattern in PHISHING_PATH_PATTERNS:
        if re.search(pattern, path_lower, re.IGNORECASE):
            pattern_count += 1
            score += 0.2

    # Multiple phishing patterns is very suspicious
    if pattern_count >= 3:
        score += 0.3
    elif pattern_count >= 2:
        score += 0.15

    # Common phishing file extensions with suspicious names
    if re.search(r'/(login|signin|verify|secure|update|account)\.(php|html|asp|aspx|jsp)', path_lower):
        score += 0.25

    # PayPal-specific patterns (common target)
    if 'webscr' in path_lower or 'cgi-bin/webscr' in path_lower:
        score += 0.3

    # Path contains action keywords
    action_keywords = ['submit', 'process', 'validate', 'confirm', 'complete']
    for keyword in action_keywords:
        if keyword in path_lower:
            score += 0.1

    return min(score, 1.0)


def _extract_suspicious_filename_score(path: str) -> float:
    """
    Detect suspicious filenames in the URL path.

    Phishing sites often use files named:
    - login.php, verify.html, secure.aspx
    - update_account.php, confirm_billing.html

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not path:
        return 0.0

    score = 0.0
    path_lower = path.lower()

    # Extract filename from path
    path_parts = path_lower.split('/')
    filename = path_parts[-1] if path_parts else ''

    # Remove extension for checking
    filename_base = re.sub(r'\.[a-z]{2,4}$', '', filename)

    # Check for suspicious filename keywords
    suspicious_count = 0
    for keyword in PHISHING_FILENAMES:
        if keyword in filename_base:
            suspicious_count += 1
            score += 0.15

    # Multiple suspicious keywords in filename
    if suspicious_count >= 2:
        score += 0.2

    # Check for suspicious patterns in full filename
    # e.g., login_verify_secure.php
    if filename_base.count('_') >= 2:
        parts = filename_base.split('_')
        phishing_parts = sum(1 for p in parts if p in PHISHING_FILENAMES)
        if phishing_parts >= 2:
            score += 0.25

    # Numeric suffix on phishing filename (login1.php, secure2.html)
    if re.search(r'(login|verify|secure|account|update)\d+\.(php|html|asp)', filename):
        score += 0.2

    # Random-looking filename with suspicious extension
    if len(filename_base) > 10:
        entropy = _shannon_entropy(filename_base)
        if entropy > 3.5 and filename.endswith(('.php', '.html', '.asp', '.aspx')):
            score += 0.15

    return min(score, 1.0)


def _extract_excessive_subdomain_score(domain: str) -> float:
    """
    Detect excessive subdomain depth.

    Legitimate sites rarely have more than 3-4 subdomain levels.
    Phishing sites often use deep subdomains to hide the real domain.
    e.g., secure.login.paypal.verify.evil.com

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0]  # Remove port
    parts = domain_clean.split('.')

    # Calculate subdomain depth (total parts - domain - TLD)
    subdomain_depth = max(0, len(parts) - 2)

    score = 0.0

    # Score based on depth
    if subdomain_depth >= 5:
        score += 0.5
    elif subdomain_depth >= 4:
        score += 0.3
    elif subdomain_depth >= 3:
        score += 0.15

    # Check for brand/keyword abuse in subdomains
    if subdomain_depth >= 2:
        subdomains = parts[:-2]
        suspicious_keywords = ['secure', 'login', 'verify', 'account', 'update',
                              'paypal', 'amazon', 'google', 'bank', 'billing']

        keyword_count = 0
        for subdomain in subdomains:
            for keyword in suspicious_keywords:
                if keyword in subdomain.lower():
                    keyword_count += 1

        if keyword_count >= 2:
            score += 0.3
        elif keyword_count >= 1:
            score += 0.15

    # Very long total domain length with many subdomains
    if len(domain_clean) > 50 and subdomain_depth >= 3:
        score += 0.2

    return min(score, 1.0)


def _extract_login_form_indicators(url: str, path: str, query: str) -> float:
    """
    Detect indicators of credential harvesting/login form phishing.

    These URLs often contain:
    - Form action parameters (action=login, submit=true)
    - Session/token parameters
    - Email/username fields in query string

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not url:
        return 0.0

    score = 0.0
    url_lower = url.lower()
    query_lower = query.lower() if query else ''

    # Form-related parameters
    form_params = ['action=', 'submit=', 'form=', 'cmd=', 'process=']
    for param in form_params:
        if param in url_lower:
            score += 0.15

    # Credential-related parameters
    cred_params = ['user', 'email', 'login', 'pass', 'pwd', 'credential',
                   'account', 'username', 'userid', 'member']
    for param in cred_params:
        if param + '=' in query_lower:
            score += 0.2

    # Session/token parameters (often used in phishing to track victims)
    session_params = ['token=', 'session=', 'sid=', 'ssid=', 'auth=', 'key=']
    for param in session_params:
        if param in query_lower:
            score += 0.1

    # Return URL parameter (for redirect after "login")
    if 'return=' in query_lower or 'redirect=' in query_lower or 'next=' in query_lower:
        score += 0.15

    # Suspicious path + query combination
    if any(kw in path.lower() for kw in ['login', 'signin', 'auth']) and query:
        score += 0.2

    # Email-like pattern in query (collecting emails)
    if re.search(r'[a-zA-Z0-9._%+-]+%40[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', query):
        score += 0.25  # URL-encoded email

    return min(score, 1.0)


def _extract_data_exfil_pattern_score(url: str, query: str) -> float:
    """
    Detect URL patterns commonly used for data exfiltration.

    Phishing sites often encode stolen data in URL parameters:
    - Base64-encoded data
    - Very long query strings
    - Suspicious parameter patterns

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not url:
        return 0.0

    score = 0.0

    # Very long query string (potential data exfiltration)
    if query and len(query) > 200:
        score += 0.2
    if query and len(query) > 500:
        score += 0.2

    # Base64-like patterns in query
    if query:
        # Look for long base64-like strings
        base64_pattern = re.search(r'[a-zA-Z0-9+/]{40,}={0,2}', query)
        if base64_pattern:
            score += 0.25

    # Multiple equals signs in URL (many parameters)
    equals_count = url.count('=')
    if equals_count > 10:
        score += 0.2
    elif equals_count > 5:
        score += 0.1

    # Hex-encoded data in URL
    if re.search(r'(%[0-9A-Fa-f]{2}){10,}', url):
        score += 0.2

    # Suspicious parameter names for exfiltration
    exfil_params = ['data=', 'payload=', 'info=', 'details=', 'content=',
                   'body=', 'message=', 'text=', 'encoded=', 'b64=']
    for param in exfil_params:
        if param in url.lower():
            score += 0.15

    # Very high entropy query string (random/encoded data)
    if query and len(query) > 50:
        query_entropy = _shannon_entropy(query)
        if query_entropy > 4.5:
            score += 0.2

    return min(score, 1.0)


def _extract_typosquat_distance_score(domain: str) -> float:
    """
    Improved typosquatting detection using Levenshtein distance.

    Detects domains that are suspiciously close to known brands:
    - Single character substitution (g00gle, amaz0n)
    - Character insertion/deletion (paypall, amazn)
    - Character transposition (amazno, goolge)

    Returns score 0.0-1.0 (higher = more suspicious)
    """
    if not domain:
        return 0.0

    domain_clean = domain.split(':')[0].lower()

    # Extract main domain (remove TLD)
    parts = domain_clean.split('.')
    if len(parts) < 2:
        return 0.0

    # Get the registrable domain (e.g., "google" from "www.google.com")
    # Check all parts except TLD
    check_parts = parts[:-1] if len(parts) >= 2 else parts

    # High-value targets for typosquatting
    target_brands = {
        'paypal': ['paypa1', 'paypaI', 'paypai', 'paypall', 'payp4l', 'peypal', 'paypol'],
        'amazon': ['amaz0n', 'amazom', 'amazn', 'amezon', 'amazone', 'amazonn'],
        'google': ['g00gle', 'googel', 'gogle', 'googIe', 'g0ogle', 'gooogle'],
        'apple': ['app1e', 'appIe', 'aple', 'applle', 'aplle'],
        'microsoft': ['micros0ft', 'microsft', 'mircosoft', 'micosoft', 'microsoftt'],
        'facebook': ['faceb00k', 'facebok', 'faceboook', 'facebbook'],
        'netflix': ['netf1ix', 'netfiix', 'netflx', 'netflixx'],
        'instagram': ['instagr4m', 'instgram', 'instagrom', 'lnstagram'],
        'twitter': ['twiter', 'twtter', 'tw1tter', 'twitterr'],
        'linkedin': ['linkedln', 'linkdin', 'linkedinn', 'l1nkedin'],
        'chase': ['chas3', 'chese', 'chas', 'chasee'],
        'wellsfargo': ['wel1sfargo', 'wellsfarg0', 'welsfargo'],
        'bankofamerica': ['bankofamer1ca', 'bankofamerca'],
    }

    score = 0.0

    for part in check_parts:
        if len(part) < 4:
            continue

        # Check against known typosquat patterns
        for brand, typos in target_brands.items():
            # Exact typosquat match
            if part in typos:
                score += 0.8
                break

            # Close edit distance (1-2 edits)
            similarity = difflib.SequenceMatcher(None, part, brand).ratio()

            # Very close but not exact (typosquatting zone)
            if 0.75 <= similarity < 1.0 and len(part) >= len(brand) - 1:
                # Higher score for closer matches
                if similarity >= 0.9:
                    score += 0.6
                elif similarity >= 0.8:
                    score += 0.4
                else:
                    score += 0.2
                break

            # Check for digit substitution (0 for o, 1 for l/i)
            normalized = part.replace('0', 'o').replace('1', 'l').replace('1', 'i')
            if normalized == brand and normalized != part:
                score += 0.7
                break

    return min(score, 1.0)


# ============================================================================
# OPTIONAL INFERENCE-TIME FUNCTIONS (Network-Dependent)
# ============================================================================
# These functions require network calls and should ONLY be used during
# inference, NOT during training. They are provided for Stage 2 enrichment.

def expand_url_shortener(url: str, timeout: int = 5, max_redirects: int = 10) -> Optional[str]:
    """
    Expand shortened URL to get final destination.

    IMPORTANT: This makes network calls and should ONLY be used during
    inference/Stage 2 enrichment, NOT during training.

    Args:
        url: The potentially shortened URL
        timeout: Request timeout in seconds (default: 5)
        max_redirects: Maximum redirects to follow (default: 10)

    Returns:
        Final destination URL, or None if expansion fails

    Example:
        >>> expand_url_shortener("https://bit.ly/3xK2jF9")
        "https://www.example.com/real-destination"
    """
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        return response.url if response.url != url else url
    except Exception:
        # If expansion fails, return None (could be network error, invalid URL, etc.)
        return None


def check_domain_age_days(domain: str, timeout: int = 5) -> Optional[int]:
    """
    Get domain age in days since registration using WHOIS lookup.

    IMPORTANT: This makes network calls and should ONLY be used during
    inference/Stage 2 enrichment, NOT during training.

    Args:
        domain: The domain name to check (e.g., "example.com")
        timeout: Lookup timeout in seconds (default: 5)

    Returns:
        Age in days since registration, or None if lookup fails

    Example:
        >>> check_domain_age_days("google.com")
        9131  # Google.com is 25+ years old
        >>> check_domain_age_days("newphishing.tk")
        5  # Newly registered domain (HIGH RISK)

    NOTE: Requires 'python-whois' package:
        pip install python-whois
    """
    try:
        import whois

        domain_clean = domain.split(':')[0]  # Remove port if present
        w = whois.whois(domain_clean)

        # Get creation date
        creation_date = w.creation_date

        # Handle list of dates (some domains return multiple)
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            age = (datetime.now() - creation_date).days
            return age

        return None

    except Exception:
        # WHOIS lookup failed (domain doesn't exist, network error, etc.)
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string"""
    if not s:
        return 0.0
    
    # Calculate character frequency
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # Calculate entropy
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    
    return entropy


def _max_consecutive_chars(s: str) -> int:
    """Find the maximum number of consecutive identical characters"""
    if not s:
        return 0
    
    max_count = 1
    current_count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    
    return max_count


def _empty_features() -> Dict[str, any]:
    """Return a dictionary with all features set to default values"""
    return {
        # Basic structural (7)
        'url_length': 0,
        'domain_length': 0,
        'path_length': 0,
        'query_length': 0,
        'scheme_is_https': 0,
        'has_port': 0,
        'url_has_fragment': 0,
        
        # Character distribution (17)
        'num_digits': 0,
        'num_letters': 0,
        'num_special_chars': 0,
        'digit_letter_ratio': 0.0,
        'digit_ratio': 0.0,
        'letter_ratio': 0.0,
        'special_char_ratio': 0.0,
        'num_dots': 0,
        'num_hyphens': 0,
        'num_underscores': 0,
        'num_slashes': 0,
        'num_question_marks': 0,
        'num_equals': 0,
        'num_ampersands': 0,
        'num_at_signs': 0,
        'num_percent': 0,
        'domain_has_digits': 0,
        
        # Domain-based (8)
        'tld_length': 0,
        'num_subdomains': 0,
        'is_ip_address': 0,
        'has_suspicious_tld': 0,
        'is_url_shortener': 0,
        'has_punycode': 0,
        'domain_vowel_ratio': 0.0,
        'domain_has_double_slash': 0,
        
        # Path and query (6)
        'path_depth': 0,
        'num_query_params': 0,
        'has_suspicious_params': 0,
        'num_encoded_chars': 0,
        'has_file_extension': 0,
        'has_suspicious_extension': 0,
        
        # Security indicators (8)
        'has_brand_keyword': 0,
        'has_suspicious_keyword': 0,
        'has_hex_encoding': 0,
        'has_unicode_encoding': 0,
        'has_long_alphanumeric': 0,
        'has_at_in_url': 0,
        'has_double_extension': 0,
        'has_mixed_case_domain': 0,
        
        # Statistical (5)
        'url_entropy': 0.0,
        'domain_entropy': 0.0,
        'path_entropy': 0.0,
        'max_consecutive_chars': 0,
        'char_variety': 0.0,

        # Advanced features (9: 6 original + 3 new)
        'homoglyph_score': 0.0,
        'brand_similarity_score': 0.0,
        'subdomain_suspicion_score': 0.0,
        'url_complexity_score': 0.0,
        'redirect_pattern_score': 0.0,
        'domain_impersonation_score': 0.0,
        'unicode_homoglyph_score': 0.0,
        'url_shortener_risk_score': 0.0,
        'new_domain_risk_score': 0.0,

        # Pattern features (8 new for FN detection)
        'cms_exploit_score': 0.0,
        'nested_tld_score': 0.0,
        'phishing_path_score': 0.0,
        'suspicious_filename_score': 0.0,
        'excessive_subdomain_score': 0.0,
        'login_form_indicators': 0.0,
        'data_exfil_pattern_score': 0.0,
        'typosquat_distance_score': 0.0,
    }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def url_features(url: str) -> dict:
    """
    Backward compatible function name.
    Calls the enhanced version.
    """
    return url_features_enhanced(url)


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/admin/login.php",
        "https://paypal-secure-login.tk/verify?token=abc123",
        "http://bit.ly/2x9f8Ks",
        "https://xn--80akhbyknj4f.xn--p1ai/",  # punycode
        "http://legitimate-bank.com/update-account.exe",
    ]
    
    print("=" * 80)
    print("URL FEATURES EXTRACTION TEST")
    print("=" * 80)
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = url_features_enhanced(url)
        
        # Print key indicators
        print(f"  Length: {features['url_length']}")
        print(f"  Entropy: {features['url_entropy']:.2f}")
        print(f"  Suspicious TLD: {features['has_suspicious_tld']}")
        print(f"  Brand keyword: {features['has_brand_keyword']}")
        print(f"  IP address: {features['is_ip_address']}")
        print(f"  URL shortener: {features['is_url_shortener']}")
        print(f"  Total features: {len(features)}")
    
    print("\n" + "=" * 80)
    print(f"Total features extracted per URL: {len(_empty_features())}")
    print("=" * 80)