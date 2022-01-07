#!/bin/bash

DATA_DIR=$(realpath ../data)
CODE_DIR=$(realpath .)

echo "Data dir: $DATA_DIR"
echo "Code dir: $CODE_DIR"

docker run -it --runtime=nvidia \
               --mount type=bind,source=$DATA_DIR,target=/root/mount/R2R/data \
               --volume $CODE_DIR:/root/mount/R2R/code \
               kxnguyen/pytorch:pytorch17-transformers 
