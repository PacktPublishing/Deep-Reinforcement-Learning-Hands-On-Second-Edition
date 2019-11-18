#!/usr/bin/env bash
for i in `seq 0 7`; do
    p1=$((5900+i))
    p2=$((15900+i))
    docker run -d -p $p1:5900 -p $p2:15900 --privileged \
        --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
done
