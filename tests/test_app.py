#!/usr/bin/env python3
"""
Test script for MedNeutral Flask app
Verifies that the web application can start and process requests
"""

import sys
import os
import time
import requests
import subprocess
import threading

def test_app_functionality():
    """Test the Flask app functionality"""
    print("Testing MedNeutral Flask app...")
    
    # Start the Flask app in a separate thread
    def start_app():
        try:
            # Change to app directory
            app_dir = os.path.join(os.getcwd(), 'app')
            os.chdir(app_dir)
            
            # Start Flask app
            subprocess.run([sys.executable, 'app.py'], 
                         capture_output=True, text=True)
        except Exception as e:
            print(f"Error starting app: {e}")
    
    # Start app in background
    app_thread = threading.Thread(target=start_app, daemon=True)
    app_thread.start()
    
    # Wait for app to start
    print("Waiting for Flask app to start...")
    time.sleep(3)
    
    try:
        # Test the stats endpoint
        response = requests.get('http://localhost:5001/stats', timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats endpoint working: {stats}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
        
        # Test the process endpoint
        test_data = {
            "text": "Patient was adamant about their symptoms and refused to comply with treatment."
        }
        
        response = requests.post('http://localhost:5001/process', 
                               json=test_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Process endpoint working: {result['num_changes']} changes made")
            return True
        else:
            print(f"❌ Process endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app")
        return False
    except Exception as e:
        print(f"❌ Error testing app: {e}")
        return False

def main():
    """Run app tests"""
    print("MedNeutral Flask App Test")
    print("=" * 40)
    
    try:
        success = test_app_functionality()
        
        if success:
            print("\n" + "=" * 40)
            print("🎉 Flask app is working correctly!")
            print("You can now access it at: http://localhost:5001")
        else:
            print("\n" + "=" * 40)
            print("❌ Flask app test failed")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
