#!/usr/bin/bash

cd /eos/user/c/cahampso/lq-asym/
source ./.venv/bin/activate
python3 ./friend_ntuples/produce_small.py --file "$1"
