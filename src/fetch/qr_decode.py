# src/fetch/qr_decode.py
"""
Enhanced QR Code Decoder with Improved URL Extraction

Key improvements:
1. Better URL extraction - removes metadata/artifacts
2. Multiple decoding engines (OpenCV, Pyzbar, ZXing)
3. Smart URL pattern matching
4. Removes number prefixes and "url" suffixes
"""

from __future__ import annotations
import io
import re
from typing import Optional, Tuple

# === QR Decoding Engines ===

# 1) OpenCV (best on Windows, no external DLLs)
_cv_ok = False
try:
    import cv2  # type: ignore
    _cv_ok = True
except Exception:
    _cv_ok = False

# 2) Optional fallback: pyzbar (requires zbar DLL)
_pyzbar_ok = False
try:
    from PIL import Image
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
    _pyzbar_ok = True
except Exception:
    _pyzbar_ok = False

# 3) Optional: zxingcpp (high accuracy)
_zxing_ok = False
try:
    import zxingcpp  # type: ignore
    _zxing_ok = True
except Exception:
    _zxing_ok = False


# === URL Pattern Matching ===
# This regex will match http:// or https:// URLs
URL_PATTERN = re.compile(
    r'(?:https?://)'  # Must start with http:// or https://
    r'[^\s<>"{}|\\^`\[\]]+'  # Valid URL characters
)


def _extract_url_from_text(text: str) -> Optional[str]:
    """
    Extract a clean URL from decoded QR text that may contain metadata.
    
    Examples of problematic input:
    - "383259 http://accenti.mx/nryojp url"
    - "392227http://maagmen.pl/new/url"
    - "http://example.com _url"
    
    This function will extract just the URL part.
    """
    if not text:
        return None
    
    # Find all URLs in the text
    urls = URL_PATTERN.findall(text)
    
    if urls:
        # Take the first URL found
        url = urls[0]
        
        # Clean up any trailing artifacts
        # Remove common suffixes like '_url', ',url', etc.
        url = re.sub(r'[_,\s]url$', '', url, flags=re.IGNORECASE)
        
        # Remove trailing punctuation that's not part of URLs
        url = url.rstrip(',;')
        
        return url
    
    # If no URL pattern found, check if it looks like a URL without protocol
    # (some QR codes might omit http://)
    text_cleaned = text.strip()
    
    # If it starts with something that looks like a domain
    if re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9.]+\.[a-zA-Z]{2,}', text_cleaned):
        # Add http:// prefix
        return 'http://' + text_cleaned
    
    return None


def _clean_decoded_text(text: str) -> str:
    """
    Remove artifacts and extraneous text from QR decoding.
    Then extract the actual URL.
    """
    if not text:
        return ""
    
    # Remove pandas/numpy artifacts
    bad_tokens = ["Name:", "dtype:", "object", "Series", "Index", "NaN"]
    for token in bad_tokens:
        text = text.replace(token, "")
    
    # Remove newlines and carriage returns
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Extract the URL from the cleaned text
    url = _extract_url_from_text(text)
    
    return url.strip() if url else ""


# === OpenCV QR Decoder ===
def _decode_with_opencv(img_bytes: bytes) -> Optional[str]:
    import numpy as np
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    qr = cv2.QRCodeDetector()
    data, points, _ = qr.detectAndDecode(img)
    if not data:
        return None

    return _clean_decoded_text(data)


# === Pyzbar QR Decoder ===
def _decode_with_pyzbar(img_bytes: bytes) -> Optional[str]:
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except Exception:
        return None

    try:
        results = zbar_decode(im)
    except Exception:
        return None

    for d in results:
        txt = d.data.decode("utf-8", errors="ignore").strip()
        if txt:
            url = _clean_decoded_text(txt)
            if url:
                return url
    return None


# === ZXing QR Decoder ===
def _decode_with_zxing(img_bytes: bytes) -> Optional[str]:
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except Exception:
        return None
    
    try:
        result = zxingcpp.read_barcode(im)
        if result and result.format == "QRCode" and result.text:
            return _clean_decoded_text(result.text)
    except Exception:
        pass
    
    return None


# === Public Function ===
def decode_qr_bytes(img_bytes: bytes) -> Tuple[Optional[str], str]:
    """
    Attempts to decode a QR code from bytes and extract the URL.
    Returns: (decoded_url, method_used)
    
    The decoded_url will be cleaned of metadata/artifacts and contain only the URL.
    """
    # Try ZXing first (most accurate when available)
    if _zxing_ok:
        try:
            out = _decode_with_zxing(img_bytes)
            if out:
                return out, "zxing"
        except Exception:
            pass
    
    # Try OpenCV
    if _cv_ok:
        try:
            out = _decode_with_opencv(img_bytes)
            if out:
                return out, "opencv"
        except Exception:
            pass

    # Fallback to Pyzbar
    if _pyzbar_ok:
        try:
            out = _decode_with_pyzbar(img_bytes)
            if out:
                return out, "pyzbar"
        except Exception:
            pass

    return None, "no_decoder_or_no_qr"


# === Testing Function ===
def test_url_extraction():
    """Test cases for URL extraction"""
    test_cases = [
        ("383259 http://accenti.mx/nryojp url", "http://accenti.mx/nryojp"),
        ("392227http://maagmen.pl/new/url", "http://maagmen.pl/new/url"),
        ("http://example.com _url", "http://example.com"),
        ("https://google.com", "https://google.com"),
        ("12345 https://test.com/path?query=1 url", "https://test.com/path?query=1"),
        ("example.com", "http://example.com"),
    ]
    
    print("Testing URL extraction:")
    for input_text, expected in test_cases:
        result = _extract_url_from_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {input_text!r}")
        print(f"  Expected: {expected!r}")
        print(f"  Got:      {result!r}")
        print()


if __name__ == "__main__":
    test_url_extraction()