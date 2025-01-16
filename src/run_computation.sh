#!/bin/bash

# Ensure the script exits on errors
set -e

# Path to the virtual environment (relative path)
VENV_PATH="./comput_main_env"

# Check if the virtual environment is already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment already activated."
fi

# Create the LOG folder if it doesn't exist
if [ ! -d "LOG" ]; then
    mkdir LOG
    echo "LOG folder created."
fi

# Set the log file name with a timestamp for uniqueness
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="Comput_main_$TIMESTAMP.log"

# Log the start time
echo "Running Comput_main.py at $(date)" | tee "$LOGFILE"

# Run the Python program and redirect output (stdout and stderr) to the log file and terminal
python Comput_main.py 2>&1 | tee -a "$LOGFILE"

# Log the end time
echo "Completed at $(date)" | tee -a "$LOGFILE"

# Move the log file into the LOG directory
mv "$LOGFILE" LOG/
echo "Log file saved to LOG/$LOGFILE"

