#!/bin/sh

# Install gdown if not already installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing..."
    pip install gdown
fi

# Download the zip file from Google Drive
gdown "1cLEwlNrvxqDL84_kQoEpaaaw3mKjQdRf" -O data.zip

# https://drive.google.com/file/d/1cLEwlNrvxqDL84_kQoEpaaaw3mKjQdRf/view?usp=sharing

# Unzip the downloaded file
unzip data.zip -d ./

# Remove the zip file
rm data.zip