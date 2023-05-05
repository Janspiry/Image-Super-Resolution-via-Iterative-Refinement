#!/bin/bash

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
apt-get update && apt-get install -y libgl1-mesa-dev

pip install -r requirements.txt
