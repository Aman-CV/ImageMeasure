"""
WSGI config for Measure project.

It exposes the WSGI callable as a module-level variable named ``application``.
"""

import os
import warnings

from django.core.wsgi import get_wsgi_application

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Measure.settings')

application = get_wsgi_application()