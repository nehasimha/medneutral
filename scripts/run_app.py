#!/usr/bin/env python3
"""
Launcher script for MedNeutral Flask app
Run this from the medneutral directory to start the web application
"""

import os
import sys
import subprocess

def main():
    """Launch the Flask app with correct working directory"""
    
    # Get the current directory (should be medneutral/)
    current_dir = os.getcwd()
    
    # Check if we're in the right directory
    if not os.path.exists('app/app.py'):
        print("Error: Please run this script from the medneutral directory")
        print("Current directory:", current_dir)
        print("Expected to find: app/app.py")
        sys.exit(1)
    
    # Change to app directory and run Flask
    app_dir = os.path.join(current_dir, 'app')
    os.chdir(app_dir)
    
    print("Starting MedNeutral Flask app...")
    print("Web interface will be available at: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
