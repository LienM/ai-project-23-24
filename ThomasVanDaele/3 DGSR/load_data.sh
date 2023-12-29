#!/bin/bash
cd ../DGSR
python -u  new_data.py \
 --data=transactions_train \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=3
