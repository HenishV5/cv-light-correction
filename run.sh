#!/bin/bash
echo "Activating Python environment..."
source ~/.bashrc # Update with your environment activation command if needed (e.g., conda activate <env-name>)


export CUDA_VISIBLE_DEVICES=0

Z_DCE_MODEL="z_dce_network_ZESR.pth"
SR_CNN_MODEL="sr_cnn_ZESR.pth"

if [[ ! -f $Z_DCE_MODEL ]]; then
  echo "Error: Model file $Z_DCE_MODEL not found!"
  exit 1
fi

if [[ ! -f $SR_CNN_MODEL ]]; then
  echo "Error: Model file $SR_CNN_MODEL not found!"
  exit 1
fi

echo "Running test.py..."
python test.py

echo "Test script execution completed!"