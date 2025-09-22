import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Optional, Dict, Union, Tuple, Any
from pathlib import Path
from lpm_kernel.configs.config import Config
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """Email service for sending notifications"""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.config = Config.from_env()
        # Get email configuration from environment variables
        self.smtp_host = self.config.get('SMTP_HOST')
        self.smtp_port = int(self.config.get('SMTP_PORT', '587'))
        self.smtp_user = self.config.get('SMTP_USER')
        self.smtp_password = self.config.get('SMTP_PASSWORD')
        self.sender_email = self.config.get('SENDER_EMAIL')
        self.use_tls = self.config.get('SMTP_USE_TLS', 'True').lower() == 'true'
        
        # Email templates directory
        self.templates_dir = self.config.get('EMAIL_TEMPLATES_DIR', os.path.join(os.path.dirname(__file__), 'email_templates'))
        
        # Retry configuration
        self.max_retries = int(self.config.get('EMAIL_MAX_RETRIES', '3'))
        self.retry_delay = int(self.config.get('EMAIL_RETRY_DELAY_SECONDS', '5'))
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)

    def send_email(self, 
                  subject: str, 
                  body: str, 
                  to_emails: List[str],
                  is_html: bool = False,
                  attachments: Optional[List[str]] = None,
                  cc_emails: Optional[List[str]] = None,
                  bcc_emails: Optional[List[str]] = None) -> bool:
        """Send email to recipients with enhanced functionality

        Args:
            subject: Email subject
            body: Email body content
            to_emails: List of recipient email addresses
            is_html: Whether the body content is HTML format
            attachments: Optional list of file paths to attach
            cc_emails: Optional list of CC recipient email addresses
            bcc_emails: Optional list of BCC recipient email addresses

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        # Validate email configuration
        if not self._validate_email_config():
            return False
            
        # Initialize retry counter
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Create message container
                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = ', '.join(to_emails)
                msg['Subject'] = subject
                
                # Add CC and BCC if provided
                if cc_emails:
                    msg['Cc'] = ', '.join(cc_emails)
                if bcc_emails:
                    msg['Bcc'] = ', '.join(bcc_emails)

                # Attach body with appropriate content type
                content_type = 'html' if is_html else 'plain'
                msg.attach(MIMEText(body, content_type))

                # Add attachments if provided
                if attachments:
                    for file_path in attachments:
                        self._add_attachment(msg, file_path)

                # Combine all recipients for sending
                all_recipients = to_emails.copy()
                if cc_emails:
                    all_recipients.extend(cc_emails)
                if bcc_emails:
                    all_recipients.extend(bcc_emails)

                # Connect to SMTP server and send email
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    if self.use_tls:
                        server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)

                logger.info(f"Email sent successfully to {', '.join(to_emails)}")
                return True

            except Exception as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    logger.warning(f"Email sending attempt {retry_count} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to send email after {self.max_retries} attempts: {str(e)}")
                    return False
        
        return False
        
    def _validate_email_config(self) -> bool:
        """Validate that all required email configuration is present
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_configs = {
            'SMTP Host': self.smtp_host,
            'SMTP Port': self.smtp_port,
            'SMTP User': self.smtp_user,
            'SMTP Password': self.smtp_password,
            'Sender Email': self.sender_email
        }
        
        missing_configs = [key for key, value in required_configs.items() if not value]
        
        if missing_configs:
            logger.error(f"Email configuration is incomplete. Missing: {', '.join(missing_configs)}")
            return False
        return True
        
    def _add_attachment(self, msg: MIMEMultipart, file_path: str) -> None:
        """Add an attachment to the email message
        
        Args:
            msg: The email message to attach to
            file_path: Path to the file to attach
        """
        try:
            with open(file_path, 'rb') as file:
                part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                msg.attach(part)
        except Exception as e:
            logger.error(f"Failed to attach file {file_path}: {str(e)}")
            
    def _load_template(self, template_name: str) -> Optional[str]:
        """Load an email template from the templates directory
        
        Args:
            template_name: Name of the template file
            
        Returns:
            str: Template content or None if template not found
        """
        template_path = os.path.join(self.templates_dir, template_name)
        try:
            if os.path.exists(template_path):
                with open(template_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                logger.warning(f"Email template not found: {template_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load email template {template_name}: {str(e)}")
            return None

    def send_training_start_notification(self,
                                        model_name: str,
                                        to_email: str,
                                        training_params: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification when training process starts

        Args:
            model_name: Name of the model being trained
            to_email: Recipient's email address
            training_params: Optional dictionary containing training parameters

        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        subject = f"Training Process Started - {model_name}"
        
        # Build the email body
        body = f"""<h2>Training Process Started</h2>
                 <p>The training process for model <strong>{model_name}</strong> has been initiated.</p>"""

        # Add training parameters if provided
        if training_params:
            body += """<h3>Training Parameters:</h3>
                      <ul>"""
            for key, value in training_params.items():
                body += f"<li><strong>{key}:</strong> {value}</li>"
            body += "</ul>"
            
        body += "<p>You will receive updates on the training progress and completion.</p>"

        return self.send_email(subject, body, [to_email], is_html=True)
        
    def send_training_progress_notification(self,
                                           model_name: str,
                                           to_email: str,
                                           progress_percentage: float,
                                           current_step: str,
                                           estimated_completion_time: Optional[str] = None) -> bool:
        """Send notification about training progress

        Args:
            model_name: Name of the model being trained
            to_email: Recipient's email address
            progress_percentage: Current progress percentage (0-100)
            current_step: Current training step description
            estimated_completion_time: Optional estimated completion time

        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        subject = f"Training Progress Update - {model_name} ({progress_percentage:.1f}%)"
        
        # Build the email body
        body = f"""<h2>Training Progress Update</h2>
                 <p>The training process for model <strong>{model_name}</strong> is in progress.</p>
                 
                 <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                    <p><strong>Current Progress:</strong> {progress_percentage:.1f}%</p>
                    <div style="background-color: #e0e0e0; border-radius: 5px; height: 20px; width: 100%; margin-top: 5px;">
                        <div style="background-color: #4CAF50; border-radius: 5px; height: 20px; width: {min(progress_percentage, 100)}%;"></div>
                    </div>
                 </div>
                 
                 <p><strong>Current Step:</strong> {current_step}</p>"""
                 
        if estimated_completion_time:
            body += f"<p><strong>Estimated Completion Time:</strong> {estimated_completion_time}</p>"
            
        return self.send_email(subject, body, [to_email], is_html=True)

    def send_training_error_notification(self, 
                                       model_name: str, 
                                       error_message: str,
                                       to_email: str,
                                       log_file_path: Optional[str] = None,
                                       error_context: Optional[Dict[str, Any]] = None,
                                       max_log_lines: int = 50) -> bool:
        """Send notification when training process encounters an error

        Args:
            model_name: Name of the model being trained
            error_message: Error message to include in notification
            to_email: Recipient's email address
            log_file_path: Optional path to log file to attach
            error_context: Optional dictionary with additional error context information
            max_log_lines: Maximum number of log lines to include in the email body

        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        subject = f"Training Error Alert - {model_name}"
        
        # Build the email body with improved styling
        body = f"""<h2 style="color: #d32f2f;">Training Process Error</h2>
                 <p>An error occurred during the training process for model: <strong>{model_name}</strong></p>
                 <p><strong>Error Details:</strong></p>
                 <pre style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; overflow-x: auto;">{error_message}</pre>"""
        
        # Add error context if provided
        if error_context and isinstance(error_context, dict):
            body += """<h3>Error Context:</h3>
                     <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
                     <tr style="background-color: #f2f2f2;">
                       <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Parameter</th>
                       <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
                     </tr>"""
                     
            for key, value in error_context.items():
                body += f"""<tr>
                         <td style="border: 1px solid #ddd; padding: 8px;">{key}</td>
                         <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                       </tr>"""
            body += "</table>"
        
        # Add recent log entries if log file exists
        if log_file_path and os.path.exists(log_file_path):
            try:
                recent_logs = self._extract_recent_logs(log_file_path, max_log_lines)
                if recent_logs:
                    body += f"""<h3>Recent Log Entries:</h3>
                             <pre style="background-color: #f5f5f5; color: #333; padding: 10px; border-radius: 5px; font-size: 12px; overflow-x: auto; max-height: 300px; overflow-y: auto;">{recent_logs}</pre>"""
            except Exception as e:
                logger.error(f"Failed to extract recent logs from {log_file_path}: {str(e)}")
        
        body += """<p>Please check the complete training logs for more information.</p>
                  <p>If you need assistance, please contact the system administrator.</p>"""

        # Prepare attachments if log file is provided and exists
        attachments = None
        if log_file_path and os.path.exists(log_file_path):
            attachments = [log_file_path]

        return self.send_email(subject, body, [to_email], is_html=True, attachments=attachments)
        
    def _extract_recent_logs(self, log_file_path: str, max_lines: int = 50) -> Optional[str]:
        """Extract the most recent lines from a log file
        
        Args:
            log_file_path: Path to the log file
            max_lines: Maximum number of lines to extract
            
        Returns:
            str: Recent log entries or None if extraction failed
        """
        try:
            # Use a deque to efficiently keep only the last max_lines
            from collections import deque
            recent_lines = deque(maxlen=max_lines)
            
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as file:
                for line in file:
                    recent_lines.append(line.rstrip())
            
            # Join the lines with HTML line breaks for display in email
            return "\n".join(recent_lines)
        except Exception as e:
            logger.error(f"Error extracting recent logs: {str(e)}")
            return None

    def send_training_completion_notification(self, 
                                            model_name: str,
                                            to_email: str,
                                            training_stats: Optional[Dict[str, Any]] = None,
                                            model_path: Optional[str] = None,
                                            training_duration: Optional[str] = None) -> bool:
        """Send notification when training process completes successfully

        Args:
            model_name: Name of the model that completed training
            to_email: Recipient's email address
            training_stats: Optional dictionary containing training statistics
            model_path: Optional path where the model is saved
            training_duration: Optional string representing the total training duration

        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        subject = f"Training Completed Successfully - {model_name}"
        
        # Build the email body with improved styling
        body = f"""<h2 style="color: #2e7d32;">Training Process Completed Successfully</h2>
                 <p>The training process for model <strong>{model_name}</strong> has completed successfully.</p>"""

        if training_duration:
            body += f"<p><strong>Total Training Duration:</strong> {training_duration}</p>"
            
        if model_path:
            body += f"<p><strong>Model Saved At:</strong> {model_path}</p>"

        # Add training statistics if provided with improved styling
        if training_stats:
            body += """<h3>Training Statistics:</h3>
                      <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
                      <tr style="background-color: #f2f2f2;">
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
                      </tr>"""
                      
            for key, value in training_stats.items():
                body += f"""<tr>
                          <td style="border: 1px solid #ddd; padding: 8px;">{key}</td>
                          <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                        </tr>"""
            body += "</table>"
            
        body += """<p>You can now use this model for inference.</p>
                  <p>Thank you for using our training service!</p>"""

        return self.send_email(subject, body, [to_email], is_html=True)
        
    def send_custom_notification(self,
                               subject: str,
                               template_name: Optional[str],
                               to_email: str,
                               template_vars: Optional[Dict[str, Any]] = None,
                               attachments: Optional[List[str]] = None) -> bool:
        """Send a custom notification using a template

        Args:
            subject: Email subject
            template_name: Name of the template file to use (if None, template_vars must contain 'body')
            to_email: Recipient's email address
            template_vars: Dictionary of variables to replace in the template
            attachments: Optional list of file paths to attach

        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        # If template name is provided, load the template
        body = None
        if template_name:
            body = self._load_template(template_name)
            if not body:
                # If template not found but template_vars contains 'body', use that instead
                if template_vars and 'body' in template_vars:
                    body = template_vars['body']
                else:
                    logger.error(f"Template {template_name} not found and no fallback body provided")
                    return False
        elif template_vars and 'body' in template_vars:
            # Use body from template_vars if no template name provided
            body = template_vars['body']
        else:
            logger.error("No template or body content provided for custom notification")
            return False
            
        # Replace template variables if provided
        if template_vars:
            for key, value in template_vars.items():
                if key != 'body':  # Skip 'body' key as it's handled separately
                    placeholder = f"{{{{{key}}}}}"
                    body = body.replace(placeholder, str(value))
                    
        return self.send_email(subject, body, [to_email], is_html=True, attachments=attachments)