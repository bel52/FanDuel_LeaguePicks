"""
Email sender utility.

This module provides a simple helper to send plain‑text emails using the SMTP
protocol.  It reads mail configuration from environment variables.  If
mandatory variables are missing, the send operation is skipped silently.

Environment variables used:
  MAIL_SERVER      – hostname of the SMTP server (required)
  MAIL_PORT        – port number as integer (defaults to 587)
  MAIL_USERNAME    – username for SMTP auth (optional)
  MAIL_PASSWORD    – password for SMTP auth (optional)
  MAIL_FROM        – from address (required)
  MAIL_TO          – destination address (required)

Usage:
    from src.emailer import send_email
    send_email("DFS Update", "Your lineup has been updated.")

The function logs its actions via the shared util.logger but does not raise
exceptions on failure.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Optional

from .util import logger


def send_email(subject: str, body: str) -> None:
    """Send a plain‑text email notification.

    Parameters
    ----------
    subject : str
        Subject of the email.
    body : str
        Body text of the email.  Plain text only; HTML content is not
        currently supported.

    Notes
    -----
    If mandatory mail configuration is missing, this function logs a warning
    and returns without sending.
    """
    server_host = os.getenv("MAIL_SERVER")
    mail_from = os.getenv("MAIL_FROM")
    mail_to = os.getenv("MAIL_TO")
    if not server_host or not mail_from or not mail_to:
        logger.warning(
            "Email configuration incomplete. Skipping email send."
        )
        return
    try:
        port = int(os.getenv("MAIL_PORT", "587"))
    except ValueError:
        port = 587
    username: Optional[str] = os.getenv("MAIL_USERNAME") or None
    password: Optional[str] = os.getenv("MAIL_PASSWORD") or None
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)
    try:
        with smtplib.SMTP(server_host, port) as smtp:
            # Attempt to upgrade the connection to TLS
            try:
                smtp.starttls()
            except Exception:
                # If TLS fails, proceed with plain connection
                pass
            if username and password:
                smtp.login(username, password)
            smtp.send_message(msg)
        logger.info(f"Sent email to {mail_to} with subject '{subject}'")
    except Exception as exc:
        logger.warning(f"Failed to send email: {exc}")
