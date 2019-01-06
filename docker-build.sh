#!/usr/bin/env bash
if [ $# = 3 ]; then
    echo "3 argument are required" 1>&2
    echo "doucker-build.sh dockerimage dockerfilename" 1>&2
    exit 1
fi

echo "building dockefile..."
filename=$2
fileimage=$1
docker build -t $fileimage -f $filename . #${@:3}
#docker build --build-arg  HOST_UID=$(id -u) -t $fileimage -f $filename . 
