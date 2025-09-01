#!/usr/bin/env python
"""
Production Run Script for Insurance Premium Analytics Platform
Videbimus AI - https://www.videbimusai.com
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from dashboard import app, server

if __name__ == '__main__':
    print("\n" + "="*70)
    print("                      VIDEBIMUS AI")
    print("           Insurance Premium Analytics Platform")
    print("="*70)
    print("\n   Production Server Starting...")
    print("   URL: http://127.0.0.1:8050")
    print("   Press CTRL+C to stop\n")
    print("="*70 + "\n")
    
    # Run the app
    app.run(debug=False, host='127.0.0.1', port=8050)