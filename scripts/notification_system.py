"""
Notification System Module

This module handles email notifications for training completion and other alerts.
"""

import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    # Load environment variables if dotenv is available
    load_dotenv()
except ImportError:
    # Silently continue if dotenv is not available
    pass

try:
    from flask_mail import Mail, Message
    flask_mail_available = True
except ImportError:
    flask_mail_available = False

# Email configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "noreply@finport.app")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@finport.app")
EMAIL_RECIPIENTS = []

# Initialize Flask-Mail for the app
mail = None

def init_mail(app):
    """Initialize Flask-Mail with the app"""
    global mail
    
    if not flask_mail_available:
        print("Flask-Mail not available, email will be sent using smtplib instead")
        return
        
    try:
        app.config['MAIL_SERVER'] = SMTP_SERVER
        app.config['MAIL_PORT'] = SMTP_PORT
        app.config['MAIL_USE_TLS'] = True
        app.config['MAIL_USERNAME'] = EMAIL_SENDER
        app.config['MAIL_PASSWORD'] = EMAIL_PASSWORD
        app.config['MAIL_DEFAULT_SENDER'] = EMAIL_SENDER
        mail = Mail(app)
        print("Flask-Mail initialized successfully")
    except Exception as e:
        print(f"Error initializing Flask-Mail: {str(e)}")
        mail = None

# Initialize logging
logger = logging.getLogger("notification_system")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def register_email(email):
    """Register an email address to receive notifications"""
    if email and email not in EMAIL_RECIPIENTS:
        EMAIL_RECIPIENTS.append(email)
        return True
    return False

def unregister_email(email):
    """Unregister an email address from notifications"""
    if email in EMAIL_RECIPIENTS:
        EMAIL_RECIPIENTS.remove(email)
        return True
    return False

def send_email(subject, message, recipients=None, attachment_path=None):
    """
    Send an email notification
    
    Args:
        subject (str): Email subject
        message (str): Email body in HTML format
        recipients (list, optional): List of recipient email addresses
        attachment_path (str, optional): Path to a file to attach
        
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    if not EMAIL_PASSWORD:
        logger.warning("Email password not configured. Cannot send notification.")
        return False
    
    # If no specific recipients, use registered emails or admin
    if not recipients:
        recipients = EMAIL_RECIPIENTS if EMAIL_RECIPIENTS else [ADMIN_EMAIL]
    
    # Try using Flask-Mail if initialized
    if mail and flask_mail_available:
        try:
            msg = Message(
                subject=subject,
                recipients=recipients,
                html=message
            )
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as file:
                    msg.attach(
                        filename=os.path.basename(attachment_path),
                        content_type="application/pdf",
                        data=file.read()
                    )
            
            mail.send(msg)
            logger.info(f"Email notification sent to {recipients} using Flask-Mail")
            return True
        except Exception as e:
            logger.error(f"Failed to send email via Flask-Mail: {str(e)}")
            # Fall back to direct SMTP
    
    # Direct SMTP approach as fallback
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        
        # Add message body
        msg.attach(MIMEText(message, 'html'))
        
        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as file:
                attachment = MIMEApplication(file.read(), _subtype=os.path.splitext(attachment_path)[1][1:])
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email notification sent to {recipients} using direct SMTP")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email notification: {str(e)}")
        return False

def send_training_completion_notification(ticker, metrics, plot_path=None):
    """
    Send notification when model training is completed
    
    Args:
        ticker (str): The ticker symbol
        metrics (dict): Performance metrics
        plot_path (str, optional): Path to performance plot
        
    Returns:
        bool: True if notification was sent successfully
    """
    subject = f"FinPort Training Completed: {ticker}"
    
    # Format performance metrics
    return_pct = metrics.get("return_pct", 0) * 100
    benchmark_return_pct = metrics.get("benchmark_return_pct", 0) * 100
    sharpe = metrics.get("sharpe_ratio", 0)
    max_drawdown = metrics.get("max_drawdown", 0) * 100
    algorithm = metrics.get("algorithm", "Unknown")
    
    # Create email message
    message = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4a86e8; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; }}
                .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
                .performance {{ font-size: 18px; font-weight: bold; color: {'green' if return_pct > 0 else 'red'}; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Model Training Completed</h2>
                </div>
                <div class="content">
                    <p>The model training for <strong>{ticker}</strong> has been completed successfully using {algorithm}.</p>
                    
                    <div class="metrics">
                        <h3>Performance Metrics</h3>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Total Return</td>
                                <td class="performance">{return_pct:.2f}%</td>
                            </tr>
                            <tr>
                                <td>Benchmark Return</td>
                                <td>{benchmark_return_pct:.2f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{sharpe:.2f}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td>{max_drawdown:.2f}%</td>
                            </tr>
                        </table>
                    </div>
                    
                    <p>You can view full results in the FinPort application.</p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from FinPort AI Portfolio Trainer.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    return send_email(subject, message, attachment_path=plot_path)

def send_backtest_notification(email, ticker, subject, report_path=None):
    """
    Send notification with backtest results
    
    Args:
        email (str): Email address to send to
        ticker (str): The ticker symbol
        subject (str): Email subject
        report_path (str, optional): Path to PDF report
        
    Returns:
        bool: True if notification was sent successfully
    """
    # Create email message
    message = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4a86e8; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; }}
                .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Backtest Results</h2>
                </div>
                <div class="content">
                    <p>The backtest for model <strong>{ticker}</strong> has been completed.</p>
                    
                    <p>A detailed PDF report is attached to this email.</p>
                    <p>You can also view full results in the FinPort application.</p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from FinPort AI Portfolio Trainer.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    return send_email(subject, message, recipients=[email], attachment_path=report_path)

def send_custom_metric_alert(ticker, metric_name, value, threshold, is_above_threshold):
    """
    Send an alert when a custom metric crosses a threshold
    
    Args:
        ticker (str): The ticker symbol
        metric_name (str): Name of the metric
        value (float): Current value of the metric
        threshold (float): Threshold value
        is_above_threshold (bool): True if alert is for exceeding threshold, False for falling below
        
    Returns:
        bool: True if notification was sent successfully
    """
    subject = f"FinPort Metric Alert: {ticker} - {metric_name}"
    condition = "exceeded" if is_above_threshold else "fallen below"
    
    message = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #e84a4a; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; }}
                .alert {{ font-size: 18px; font-weight: bold; color: #e84a4a; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Metric Alert</h2>
                </div>
                <div class="content">
                    <p>This is an automated alert for <strong>{ticker}</strong>.</p>
                    <p class="alert">The {metric_name} has {condition} the configured threshold.</p>
                    <p><strong>Current Value:</strong> {value}</p>
                    <p><strong>Threshold:</strong> {threshold}</p>
                    <p>Please check the FinPort application for more details.</p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from FinPort AI Portfolio Trainer.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    return send_email(subject, message) 