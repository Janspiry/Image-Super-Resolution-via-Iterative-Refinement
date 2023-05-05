#!/bin/bash

apt-get update && apt-get install libgl1
pip install opencv-python
pip install -r requirements.txt
