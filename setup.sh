#!/bin/bash

# Check if db.sqlite3 exists
if [ ! -f ./db.sqlite3 ]; then
    # If database file does not exist, perform initial setup
    python manage.py makemigrations
    python manage.py migrate
fi

# Start Django development server
python manage.py runserver 0.0.0.0:8000