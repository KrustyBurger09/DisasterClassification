import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
 
from .email_config import gmail_pass, user, host, port

def send_email_w_attachment(to, subject, body, filename = None):
    # create message object
    message = MIMEMultipart()
 
    # add in header
    message['From'] = Header(user)
    message['To'] = Header(to)
    message['Subject'] = Header(subject)
 
    # attach message body as MIMEText
    message.attach(MIMEText(body, 'plain', 'utf-8'))

    # locate and attach desired attachments
    if filename != None:

        att_name = os.path.basename(filename)
        _f = open(filename, 'rb')
        att = MIMEApplication(_f.read(), _subtype="jpg")
        _f.close()
        att.add_header('Content-Disposition', 'attachment', filename=att_name)
        message.attach(att)
 
    # setup email server
    server = smtplib.SMTP_SSL(host, port)
    server.login(user, gmail_pass)
 
    # send email and quit server
    server.sendmail(user, to, message.as_string())
    server.quit()
