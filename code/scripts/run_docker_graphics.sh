#!/bin/bash

DATA_DIR=$(realpath ../data)
CODE_DIR=$(realpath .)

echo "Matterport3D data: $MATTERPORT_DATA_DIR"
echo "Data dir: $DATA_DIR"                                                         
echo "Code dir: $CODE_DIR"

xhost + 
docker run -it --runtime=nvidia \
               -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
               --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/R2R/code/data/v1,readonly \
               --mount type=bind,source=$DATA_DIR,target=/root/mount/R2R/data \
               --volume $CODE_DIR:/root/mount/R2R/code \
               kxnguyen/pytorch:pytorch17-transformers
