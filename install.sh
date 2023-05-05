#!/bin/bash
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install opencv-python --user
pip install -r requirements.txt
