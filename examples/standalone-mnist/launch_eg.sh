#!/bin/bash
# cd examples/standalone-mnist
# source .venv/bin/activate
# cd /home/xavier002/FedLab_main/examples/standalone-mnist
python3 standalone.py --total_clients 100 --com_round 10 --sample_ratio 0.4 --batch_size 128 --epochs 3 --lr 0.01
