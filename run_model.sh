#!/bin/sh
hostname
source ~/.bashrc
micromamba activate agenda
pwd

# Train model
python3 ./scripts/5_train.py --config config.ini

