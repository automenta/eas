#!/usr/bin/env python3
"""causal_checker.py - Zero-model causal claim validation"""

import re

CAUSAL_WORDS = ["causes", "leads to", "results in", "produces"]
CORRELATION_WORDS = ["correlates", "associated", "linked", "related"]
STRONG_EVIDENCE = ["experiment", "randomized", "controlled", "clinical trial"]

def check_claim(text: str) -> dict:
    text_lower = text.lower()
    
    is_causal = any(w in text_lower for w in CAUSAL_WORDS)
    is_correlation = any(w in text_lower for w in CORRELATION_WORDS)
    has_evidence = any(w in text_lower for w in STRONG_EVIDENCE)
    
    if is_causal and not has_evidence:
        validity = "⚠️ WEAK - Causal claim without experimental evidence"
    elif is_causal and has_evidence:
        validity = "✅ STRONG - Causal claim with experimental support"
    elif is_correlation:
        validity = "ℹ️ NEUTRAL - Correlation claim (not causal)"
    else:
        validity = "❓ UNKNOWN - No clear causal structure"
    
    return {"validity": validity, "is_causal": is_causal, "has_evidence": has_evidence}

# Usage
print(check_claim("Coffee causes cancer according to surveys."))
# {'validity': '⚠️ WEAK - Causal claim without experimental evidence', ...}
