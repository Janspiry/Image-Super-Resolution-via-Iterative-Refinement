#!/bin/bash
RUN apt-get update && apt-get install -y libgl1-mesa-dev
RUN apt-get update && apt-get install -y libxxf86vm1

pip install -r requirements.txt
