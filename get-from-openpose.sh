#!/usr/bin/env bash
if [ $# = 3 ]; then
    echo "2 argument are required" 1>&2
    echo "video_path keypoint_path(abs)" 1>&2
    exit 1
fi

video_path=$1
keypoint_path=$2

cd /media/junkado/hdd3tb/openpose
./build/examples/openpose/openpose.bin --video $video_path --write_keypoint_json ${keypoint_path} --no_display