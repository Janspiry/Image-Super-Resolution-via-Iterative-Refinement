#!/bin/bash

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

pip install -r requirements.txt
