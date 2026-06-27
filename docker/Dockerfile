FROM nvidia/opengl:1.2-glvnd-devel-ubuntu22.04

# 确保以root用户身份运行包管理命令
USER root
ENV DEBIAN_FRONTEND=noninteractive

# GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV TZ=Asia/Shanghai
# ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



RUN apt-get update  && \
    apt-get install -y --no-install-recommends \
    sudo vim git \
    && apt-get install -y gnome-terminal \
    dbus-x11 \
    libcanberra-gtk-module \ 
    libcanberra-gtk3-module wget curl \
# other tools
    && apt install -y nano htop tmux \
    && apt install -y python3-tk \
    && apt install -y glpk-utils libglpk-dev \
# Python 3.11 and pip
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y python3.11 python3.11-distutils python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN if id -u ubuntu > /dev/null 2>&1; then \
        # 如果用户已存在，修改其 UID 和 GID
        groupmod -g ${GROUP_ID} ubuntu && \
        usermod -u ${USER_ID} -g ${GROUP_ID} ubuntu; \
    else \
        # 如果不存在则创建
        groupadd -g ${GROUP_ID} ubuntu && \
        useradd -l -u ${USER_ID} -g ubuntu -m -s /bin/bash ubuntu; \
    fi && \
    # 修复家目录权限
    chown -R ubuntu:ubuntu /home/ubuntu

# 1. 确保 ubuntu 用户在 sudo 组中
RUN usermod -aG sudo ubuntu

# 2. 创建 sudoers.d 配置文件（这是最稳妥的方法）
# 使用 tee 命令写入，确保权限和格式正确
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu
# 3. (可选) 如果你希望所有 sudo 组的用户都免密（包括未来可能创建的用户）
RUN echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


RUN cat <<'EOF' > /usr/local/bin/init_venv.sh && chmod +x /usr/local/bin/init_venv.sh
#!/bin/bash
MARKER="/tmp/.dev_install_done"
if [ ! -f "$MARKER" ]; then
    if [ -f setup.py ] || [ -f pyproject.toml ]; then
        echo ">> First shell in container: installing package in editable mode..."
        pip install -e . && touch "$MARKER"
        clear
        echo ">> Package installed in editable mode. You can now use the package directly."
    fi
fi
EOF
RUN echo 'source /usr/local/bin/init_venv.sh' >> /etc/bash.bashrc

# >>> Tools <<<
ENV HOME=/home/ubuntu


RUN git config --global --add safe.directory /home/ubuntu/cffg_robot && \
    # 修复家目录权限
    chown -R ubuntu:ubuntu /home/ubuntu

RUN apt clean && apt autoclean


USER ubuntu
WORKDIR /workspace
# 设置环境变量
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# pip install cffg_robot in editable mode
COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install --no-cache-dir -r /workspace/requirements.txt


RUN rm -rf /workspace/requirements.txt
# 设置终端颜色和提示符
ENV TERM=xterm-256color
RUN echo 'PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc
RUN echo 'alias python=python3' >> ~/.bashrc
# 设置入口点
CMD ["bash"]

