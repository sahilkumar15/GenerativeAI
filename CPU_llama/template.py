import os  # Import the os module to handle directory and file operations
from pathlib import Path  # Import the Path class from pathlib module to work with file paths in an object-oriented way
import logging  # Import the logging module to log messages

# Configure the logging module to log messages with INFO level and specify the format of the log messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define a list of file paths that we want to create
list_of_files = [
    "src/__init__.py",
    "src/run_local.py",
    "src/run_local_2.py",
    "src/helper.py",
    "model/instruction.txt",
    "requirements.txt",
    "setup.py",
    "main.py",
    "research/trials.ipynb",
    r"research\trials2.ipynb",
    "templates/index.html",
    "static/jquery.min.js",
    
]

# Loop through each file path in the list
for filepath in list_of_files:
    # Convert the file path string to a Path object
    filepath = Path(filepath)
    
    # Split the file path into directory and file name components
    filedir, filename = os.path.split(filepath)
    
    # If the directory is not empty (i.e., a directory is specified)
    if filedir != "":
        # Create the directory if it does not exist
        os.makedirs(filedir, exist_ok=True)
        # Log a message indicating that the directory is being created
        logging.info(f"Creating directory : {filedir} for the file: {filename}")
        
    # Check if the file does not exist or if it exists but is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Create an empty file and open it for writing
        with open(filepath, "w") as f:
            # No content is written to the file
            # Log a message indicating that the empty file is being created
            logging.info(f"Creating empty file: {filepath}")
    else:
        # Log a message indicating that the file already exists
        logging.info(f"{filename} already exists.")
