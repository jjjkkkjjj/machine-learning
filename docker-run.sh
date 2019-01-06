#!/usr/bin/env bash
if [ $# = 3 ]; then
    echo "3 argument are required" 1>&2
    echo "doucker-build.sh dockerimage dockerfilename" 1>&2
    exit 1
fi

filename=$2
fileimage=$1
bash ./docker-build.sh $fileimage $filename

# run
#docker run --rm -it --name=kado -v $(pwd):/workdir -w /workdir $fileimage "$@"
docker run --rm -it --name=kado -v $(pwd):/workdir -w /workdir $fileimage #${@:3}

