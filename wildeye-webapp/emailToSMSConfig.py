# emailToSMSConfig.py
"""
Email to SMS Gateway - Configuration
"""
import os

# Email credentials from environment variables or set default values here
senderEmail = os.environ.get('EMAIL_USERNAME', 'your-email@gmail.com')

# App password for Gmail (NOT your regular password)
# For Gmail, you need to create an App Password: https://myaccount.google.com/apppasswords
appKey = os.environ.get('EMAIL_PASSWORD', 'your-app-password')

# Default gateway address - will be dynamically generated in the SMS service
# Format: 1112223333@carrier-gateway.com
gatewayAddress = "default-placeholder"

# Carrier SMS gateways lookup
# Carrier SMS gateways lookup
CARRIER_GATEWAYS = {
    # Indian Carriers
    'jio': '@jiomail.com',
    'airtel': '@airtelkk.com',
    'vodafoneidea': '@vimail.in',
    'bsnl': '@bsnlmobile.in',
    'mtnl': '@mtnlmail.in',
    # US Carriers
    'verizon': '@vtext.com',
    'tmobile': '@tmomail.net',
    'sprint': '@messaging.sprintpcs.com',
    'at&t': '@txt.att.net',
    'boost': '@sms.myboostmobile.com',
    'cricket': '@sms.cricketwireless.net',
    'uscellular': '@email.uscc.net',
    # International carriers
    'vodafone': '@vodafone.net',
    'orange': '@orange.net',
    # Default to Airtel if carrier not specified
    'default': '@airtelkk.com'
}