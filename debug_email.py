import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))

# Print debug info (remove sensitive info before sharing with others)
print(f"Username: {EMAIL_USERNAME}")
print(f"Password: {'*' * len(EMAIL_PASSWORD) if EMAIL_PASSWORD else 'Not set'}")
print(f"Server: {EMAIL_SERVER}")
print(f"Port: {EMAIL_PORT}")

# Test recipient - use your email address
recipient = "meliodasfury16@gmail.com"  # Change this to your email
subject = "DEBUG TEST: TsunamiWatch System"
message_text = "This is a debug test email from the TsunamiWatch system."

# Create message
msg = MIMEMultipart()
msg["From"] = EMAIL_USERNAME
msg["To"] = recipient
msg["Subject"] = subject

# Add message body
msg.attach(MIMEText(message_text, "plain"))

# Attempt to send email with detailed error reporting
try:
    # Create secure connection
    context = ssl.create_default_context()
    
    print("Connecting to SMTP server...")
    with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
        print("Connected to server, starting TLS...")
        server.starttls(context=context)
        
        print("Logging in...")
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        
        print("Sending email...")
        server.sendmail(EMAIL_USERNAME, recipient, msg.as_string())
        
        print("Email sent successfully!")

except Exception as e:
    print(f"ERROR: Failed to send email: {str(e)}")
    # Print more detailed error information
    import traceback
    traceback.print_exc()
