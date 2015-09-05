#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/PySpark_PI/")

from FlaskApp import app as application
application.secret_key = 'password'
