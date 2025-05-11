import os
import sys
import subprocess
import time
import json

def check_files():
    # Check if required model file exists
    if not os.path.exists('tsunami_prediction_model.pkl'):
        print("Error: tsunami_prediction_model.pkl not found!")
        print("Make sure to run tsunami_model.py first to generate the model.")
        return False
    
    # Create emails.txt if it doesn't exist
    if not os.path.exists('emails.txt'):
        default_emails = ["speakup189@gmail.com", "ankity1892003@gmail.com"]
        with open('emails.txt', 'w') as f:
            f.write(json.dumps(default_emails))
        print("Created default emails.txt file.")
    
    return True

def start_dashboard():
    print("Starting TsunamiWatch AI Dashboard...")
    dashboard_process = subprocess.Popen([sys.executable, 'tsunami_dashboard.py'])
    return dashboard_process

def start_email_alert_system():
    print("Starting email alert system...")
    alert_process = subprocess.Popen([sys.executable, 'email_alert.py'])
    return alert_process

def main():
    print("=" * 70)
    print("TsunamiWatch AI - Earthquake Monitoring and Tsunami Alert System")
    print("=" * 70)
    
    # Check if all required files are present
    if not check_files():
        sys.exit(1)
    
    # Start the dashboard
    dashboard_process = start_dashboard()
    print("Dashboard started! Access it at: http://localhost:8050")
    
    # Start the email alert system
    alert_process = start_email_alert_system()
    print("Email alert system started!")
    
    print("\nBoth systems are now running. Press Ctrl+C to stop all processes.")
    
    try:
        # Keep the script running
        while True:
            # Check if processes are still running
            if dashboard_process.poll() is not None:
                print("Dashboard process terminated unexpectedly. Restarting...")
                dashboard_process = start_dashboard()
            
            if alert_process.poll() is not None:
                print("Email alert system terminated unexpectedly. Restarting...")
                alert_process = start_email_alert_system()
                
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down TsunamiWatch AI system...")
        # Terminate child processes
        dashboard_process.terminate()
        alert_process.terminate()
        print("All processes terminated. Goodbye!")

if __name__ == "__main__":
    main()