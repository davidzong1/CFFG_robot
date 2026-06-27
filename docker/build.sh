#!/bin/bash

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PARENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
VERSION=$(cd "$PARENT_DIR" && python setup.py --version 2>/dev/null || echo "unknown")
IMAGENAME="mjlab_env:$VERSION"

cd "$SCRIPT_DIR" || exit
# cp ../requirements.txt .
# docker build --build-arg HOST_USER=$(whoami) --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) \
#  -t $IMAGENAME . --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
# rm requirements.txt


DIR_HASH=$(echo "$PARENT_DIR" | md5sum | cut -c1-8)
CONTAINER_NAME="${DIR_HASH}-${VERSION}"
COMMAND=rl
echo "在创建控制脚本..."
# 检查并删除已存在的 ~/.rlscript
[ -f ~/.rlscript ] && rm ~/.rlscript
cat > ~/.rlscript << EOF
CONTAINER_NAME="$CONTAINER_NAME"
PARENT_DIR="$PARENT_DIR"
xhost +local: >> /dev/null
echo "请输入指令控制unitreelab: 重启(r) 进入(e) 启动(s) 关闭(c) 删除(d) 测试(t) 创建容器(g):"
read choose
case \$choose in
s) docker start $CONTAINER_NAME;;
r) docker restart $CONTAINER_NAME;;
e) docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=/home/$(whoami)/.Xauthority $CONTAINER_NAME /bin/bash;;
c) docker stop $CONTAINER_NAME;;
d) docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME && sudo rm -rf /home/$(whoami)/.fishros/bin/$CONTAINER_NAME;;
t) docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=/home/$(whoami)/.Xauthority $CONTAINER_NAME /bin/bash;;
g) docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --privileged \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e QT_X11_NO_MITSHM=1 \
  -e QT_QPA_PLATFORM=xcb \
  -e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY" \
  -e "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
  -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
  -e NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR" \
  -v "$XAUTHORITY:$XAUTHORITY" \
  -v "/dev/dri:/dev/dri" \
  -v $PARENT_DIR:/workspace \
  $IMAGENAME tail -f /dev/null;;
esac
newgrp docker
EOF
if ! grep -Fxq "alias $COMMAND='bash ~/.rlscript'" ~/.bashrc; then
    echo "alias $COMMAND='bash ~/.rlscript'" >> ~/.bashrc
fi
echo -e "\033[33m控制脚本创建完成，source ~/.bashrc 后使用命令 $COMMAND 来进入容器操作脚本\033[0m"