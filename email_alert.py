# email_alert.py - Fixed version
import smtplib
import ssl
import json
import time
import datetime
import os
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
required_env_vars = ['EMAIL_USERNAME', 'EMAIL_PASSWORD', 'EMAIL_SERVER', 'EMAIL_PORT']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please create a .env file with the following variables:")
    print("EMAIL_USERNAME=your_email@gmail.com")
    print("EMAIL_PASSWORD=your_app_password")  # For Gmail, use App Password
    print("EMAIL_SERVER=smtp.gmail.com")
    print("EMAIL_PORT=587")
    print("\nNOTE: If using Gmail, you need to:")
    print("1. Enable 2-factor authentication")
    print("2. Create an App Password at https://myaccount.google.com/apppasswords")
    print("3. Use that App Password in the .env file instead of your regular password")
    # Continue with default values for testing or exit based on your preference
    # exit(1)  # Uncomment to force exit if environment variables are missing

# Email configuration
EMAIL_USERNAME = os.getenv('speakup189@gmail.com', '')
EMAIL_PASSWORD = os.getenv('pssz fkbk baya fqwo', '')
EMAIL_SERVER = os.getenv('EMAIL_SERVER', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))

# Load the model
# Load the full pipeline model
try:
    pipeline = joblib.load('tsunami_prediction_model.pkl')
    print("Pipeline model loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline model: {e}")
    pipeline = None

# Define expected raw input fields for prediction
expected_fields = [
    'magnitude', 'cdi', 'mmi', 'alert', 'sig', 'net',
    'nst', 'dmin', 'gap', 'magType', 'depth', 'latitude', 'longitude',
    'continent', 'country'
]

def preprocess_input(properties):
    input_data = {}
    for field in expected_fields:
        input_data[field] = properties.get(field, None)
    df_input = pd.DataFrame([input_data])
    return df_input
print("Loading tsunami prediction model...")
try:
    model = joblib.load('tsunami_prediction_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Will continue without model and use threshold-based alerts")
    model = None

# Load the list of features used in the model
features = []
try:
    with open('model_features.txt', 'r') as f:
        features = f.read().splitlines()
    print(f"Loaded {len(features)} features: {features}")
except Exception as e:
    print(f"Error loading feature list: {e}")
    print("Using default feature list")
    features = ['magnitude', 'depth', 'latitude', 'longitude', 'sig', 'gap', 'dmin', 'mmi']

# USGS Earthquake API URL
USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
# Backup API in case primary fails
BACKUP_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"

def load_email_recipients():
    """Load email recipients from emails.txt file"""
    try:
        with open('emails.txt', 'r') as f:
            emails = json.load(f)
            # Validate email format
            valid_emails = [email for email in emails if '@' in email and '.' in email]
            if len(valid_emails) != len(emails):
                print(f"Warning: Filtered out {len(emails) - len(valid_emails)} invalid email addresses")
            return valid_emails
    except Exception as e:
        print(f"Error loading email recipients: {e}")
        # Return default email list
        default_emails = ["speakup189@gmail.com", "ankity1892003@gmail.com"]
        print(f"Using default email list: {default_emails}")
        # Write default emails to file for future use
        try:
            with open('emails.txt', 'w') as f:
                json.dump(default_emails, f)
        except Exception as write_err:
            print(f"Error writing default emails to file: {write_err}")
        return default_emails

def fetch_earthquake_data():
    """Fetch recent earthquake data from USGS API"""
    try:
        response = requests.get(USGS_API_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Primary API failed with status code {response.status_code}")
            # Try backup API
            backup_response = requests.get(BACKUP_API_URL, timeout=10)
            if backup_response.status_code == 200:
                print("Using backup API data")
                return backup_response.json()
            else:
                print(f"Backup API also failed with status code {backup_response.status_code}")
                return None
    except Exception as e:
        print(f"Error fetching earthquake data: {e}")
        return None

def prepare_prediction_data(earthquake):
    """Prepare data for prediction model"""
    try:
        # Extract properties
        props = earthquake['properties']
        coords = earthquake['geometry']['coordinates']
        
        # Create a dataframe with the required features
        data = {
            'magnitude': props.get('mag', 0),
            'depth': coords[2] if len(coords) > 2 else 0,
            'latitude': coords[1] if len(coords) > 1 else 0,
            'longitude': coords[0] if len(coords) > 0 else 0,
            'sig': props.get('sig', 0),
            'gap': props.get('gap', 0),
            'dmin': props.get('dmin', 0),
            'mmi': props.get('mmi', 0)
        }
        
        # Create dataframe with only the features the model expects
        df = pd.DataFrame([data])
        
        # Ensure all model features are present
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features used by the model
        return df[features]
    except Exception as e:
        print(f"Error preparing prediction data: {e}")
        return None

def predict_tsunami_risk(earthquake):
    """Predict tsunami risk for a given earthquake"""
    # If there's already a tsunami flag in the data, use it
    if earthquake['properties'].get('tsunami') == 1:
        return 1.0, "Official tsunami alert already issued"
    
    # If magnitude is very high, flag as high risk regardless of model
    magnitude = earthquake['properties'].get('mag', 0)
    if magnitude >= 7.5:
        return 0.9, f"Very high magnitude ({magnitude}) - high tsunami risk"
    
    # If we have a model, use it for prediction
    if model is not None:
        try:
            prediction_data = prepare_prediction_data(earthquake)
            if prediction_data is not None:
                # Make prediction
                probability = model.predict_proba(prediction_data)[0, 1]
                return probability, f"Model prediction: {probability:.2f} probability of tsunami"
        except Exception as e:
            print(f"Error during model prediction: {e}")
    
    # Fallback to simple heuristics if model fails or isn't available
    if magnitude >= 6.5 and earthquake['geometry']['coordinates'][2] < 50:
        return 0.7, "Shallow large earthquake - moderate tsunami risk"
    elif magnitude >= 6.0:
        return 0.3, "Moderate earthquake - low tsunami risk"
    else:
        return 0.1, "Low magnitude - very low tsunami risk"

def format_email_subject(earthquake, probability):
    """Format email subject based on risk level"""
    magnitude = earthquake['properties'].get('mag', 0)
    place = earthquake['properties'].get('place', 'Unknown location')
    
    risk_level = "LOW"
    if probability >= 0.7:
        risk_level = "HIGH"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
    
    return f"TSUNAMI ALERT [{risk_level}]: M{magnitude} Earthquake near {place}"

def format_email_body(earthquake, probability, reason):
    """Format email body with earthquake details and tsunami risk assessment"""
    props = earthquake['properties']
    coords = earthquake['geometry']['coordinates']
    
    # Convert timestamp to readable format
    time_ms = props.get('time', 0)
    event_time = datetime.datetime.fromtimestamp(time_ms/1000).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Generate Google Maps link
    lat = coords[1] if len(coords) > 1 else 0
    lon = coords[0] if len(coords) > 0 else 0
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    
    # Calculate risk level description
    risk_level = "Low"
    risk_color = "#1cc88a"  # Green
    if probability >= 0.7:
        risk_level = "High"
        risk_color = "#e74a3b"  # Red
    elif probability >= 0.4:
        risk_level = "Medium"
        risk_color = "#f6c23e"  # Yellow
    
    # Format the HTML email body
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: {risk_color}; color: white; padding: 10px; text-align: center; }}
            .content {{ padding: 20px; }}
            .footer {{ font-size: 12px; text-align: center; margin-top: 20px; color: #888; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .risk-badge {{ display: inline-block; padding: 5px 10px; background-color: {risk_color}; color: white; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Tsunami Risk Alert</h1>
            </div>
            <div class="content">
                <h2>Earthquake Details</h2>
                <table>
                    <tr><th>Magnitude</th><td>{props.get('mag', 'N/A')}</td></tr>
                    <tr><th>Location</th><td>{props.get('place', 'Unknown')}</td></tr>
                    <tr><th>Time</th><td>{event_time}</td></tr>
                    <tr><th>Depth</th><td>{coords[2] if len(coords) > 2 else 'N/A'} km</td></tr>
                    <tr><th>Coordinates</th><td>Lat: {lat}, Lon: {lon} <a href="{maps_link}">(View on Map)</a></td></tr>
                </table>
                
                <h2>Tsunami Risk Assessment</h2>
                <p>Risk Level: <span class="risk-badge">{risk_level}</span></p>
                <p>Probability: {probability:.2f}</p>
                <p>Assessment: {reason}</p>
                
                <p><strong>Note:</strong> This is an automated alert generated by the TsunamiWatch AI system. 
                Please consult official emergency management sources for validated information.</p>
                
                <p>For more information, visit:</p>
                <ul>
                    <li><a href="https://www.tsunami.gov/">NOAA Tsunami Warning System</a></li>
                    <li><a href="https://earthquake.usgs.gov/earthquakes/map/">USGS Earthquake Map</a></li>
                </ul>
            </div>
            <div class="footer">
                <p>This alert was generated automatically by TsunamiWatch AI at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
                <p>To unsubscribe from these alerts, reply with "UNSUBSCRIBE" in the subject line.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Also create a plain text version for email clients that don't support HTML
    plain_text = f"""
    TSUNAMI RISK ALERT
    
    Earthquake Details:
    - Magnitude: {props.get('mag', 'N/A')}
    - Location: {props.get('place', 'Unknown')}
    - Time: {event_time}
    - Depth: {coords[2] if len(coords) > 2 else 'N/A'} km
    - Coordinates: Lat: {lat}, Lon: {lon}
    - Map: {maps_link}
    
    Tsunami Risk Assessment:
    - Risk Level: {risk_level}
    - Probability: {probability:.2f}
    - Assessment: {reason}
    
    Note: This is an automated alert generated by the TsunamiWatch AI system.
    Please consult official emergency management sources for validated information.
    
    For more information, visit:
    - NOAA Tsunami Warning System: https://www.tsunami.gov/
    - USGS Earthquake Map: https://earthquake.usgs.gov/earthquakes/map/
    
    This alert was generated automatically by TsunamiWatch AI at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
    To unsubscribe from these alerts, reply with "UNSUBSCRIBE" in the subject line.
    """
    
    return html, plain_text

def send_email_alert(recipients, subject, html_content, text_content):
    """Send email alert to recipients"""
    # Check if credentials are available
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("ERROR: Email credentials not set. Cannot send alert.")
        return False
    
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = EMAIL_USERNAME
        message["To"] = ", ".join(recipients)
        
        # Add plain text and HTML parts
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        
        # The email client will try to render the last part first
        message.attach(part1)
        message.attach(part2)
        
        # Create secure connection and send message
        context = ssl.create_default_context()
        
        # Connect to server
        print(f"Connecting to email server {EMAIL_SERVER}:{EMAIL_PORT}...")
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            
            print(f"Logging in with username: {EMAIL_USERNAME}")
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            
            print(f"Sending email to {len(recipients)} recipients...")
            server.sendmail(EMAIL_USERNAME, recipients, message.as_string())
            print("Email sent successfully!")
        return True
    
    except smtplib.SMTPAuthenticationError as auth_error:
        print(f"SMTP Authentication Error: {auth_error}")
        print("Check your username and password. If using Gmail, make sure you're using an App Password.")
        return False
    except smtplib.SMTPServerDisconnected as disconnect_error:
        print(f"SMTP Server Disconnected: {disconnect_error}")
        print("Check your internet connection and email server settings.")
        return False
    except smtplib.SMTPException as smtp_error:
        print(f"SMTP Error: {smtp_error}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def process_earthquakes():
    """Process recent earthquakes and send alerts if needed"""
    print(f"Fetching earthquake data at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    earthquake_data = fetch_earthquake_data()
    
    if not earthquake_data:
        print("No earthquake data available. Will try again later.")
        return
    
    print(f"Found {len(earthquake_data['features'])} earthquakes")
    
    # Load email recipients
    recipients = load_email_recipients()
    if not recipients:
        print("No email recipients configured. Cannot send alerts.")
        return
    
    # Track if we've sent any alerts
    alerts_sent = 0
    
    # Process each earthquake
    for earthquake in earthquake_data['features']:
        try:
            # Skip if already processed (you would need to implement a tracking mechanism)
            # For simplicity, here we'll just check for recent events (last 30 minutes)
            time_ms = earthquake['properties'].get('time', 0)
            event_time = datetime.datetime.fromtimestamp(time_ms/1000)
            now = datetime.datetime.now()
            
            # Skip events older than 30 minutes
            if (now - event_time).total_seconds() > 1800:  # 30 minutes in seconds
                continue
            
            # Predict tsunami risk
            probability, reason = predict_tsunami_risk(earthquake)
            
            # Send alert if probability is above threshold
            if probability >= 0.4:  # Medium or high risk
                subject = format_email_subject(earthquake, probability)
                html_content, text_content = format_email_body(earthquake, probability, reason)
                
                print(f"Sending alert for M{earthquake['properties'].get('mag', 0)} earthquake near {earthquake['properties'].get('place', 'Unknown')}")
                success = send_email_alert(recipients, subject, html_content, text_content)
                
                if success:
                    alerts_sent += 1
        except Exception as e:
            print(f"Error processing earthquake: {e}")
    
    print(f"Processed earthquakes. Sent {alerts_sent} alerts.")

def main():
    """Main function to run the alert system"""
    print("Starting Tsunami Email Alert System...")
    print(f"Using email server: {EMAIL_SERVER}:{EMAIL_PORT}")
    print(f"Using username: {EMAIL_USERNAME}")
    
    # Check if email configuration is available
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("WARNING: Email credentials not set. Alerts cannot be sent.")
        print("Please set EMAIL_USERNAME and EMAIL_PASSWORD environment variables.")
    
    # Run forever with a sleep interval
    check_interval = int(os.getenv('CHECK_INTERVAL_MINUTES', 15)) * 60  # Convert to seconds
    print(f"Will check for new earthquakes every {check_interval//60} minutes")
    
    try:
        while True:
            process_earthquakes()
            print(f"Sleeping for {check_interval//60} minutes...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print("Tsunami Email Alert System stopped by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
        print("Tsunami Email Alert System stopped due to an error.")

if __name__ == "__main__":
    main()

    import argparse
    parser = argparse.ArgumentParser(description='Email Alert System')
    parser.add_argument('--test', action='store_true', help='Send a test email')
    args = parser.parse_args()
    
    if args.test:
        # Sample test data
        test_data = {
            "magnitude": 7.5,
            "depth": 10.0,
            "latitude": 35.681,
            "longitude": 139.767,
            "location": "Tokyo, Japan",
            "tsunami_probability": 0.85
        }
        
        # Load email recipients
        recipients = []
        if os.path.exists('emails.txt'):
            with open('emails.txt', 'r') as f:
                recipients = json.loads(f.read())
        else:
            recipients = ["meliodasfury16@gmail.com"]  # Fallback test recipient
        
        # Send test alert
        subject = "TEST ALERT: TsunamiWatch AI System"
        message = f"""
        This is a TEST alert from TsunamiWatch AI System.
        
        A significant earthquake has been detected:
        - Magnitude: {test_data['magnitude']}
        - Depth: {test_data['depth']} km
        - Location: {test_data['latitude']}, {test_data['longitude']} ({test_data['location']})
        - Tsunami Probability: {test_data['tsunami_probability'] * 100:.1f}%
        
        THIS IS ONLY A TEST. No action required.
        """
        
        send_alert_email(recipients, subject, message)
        print("Test email sent!")