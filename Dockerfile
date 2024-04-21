FROM 134.100.39.10:32000/isaacsim2023.1.1
RUN apt-get update && apt-get install -y build-essential tmux zsh wget git python3 python3-pip vim kmod nvidia-driver-535-server
COPY ./NVIDIA-Linux-x86_64-525.85.05.run /data/.
RUN chmod +x /data/NVIDIA-Linux-x86_64-525.85.05.run
RUN /data/NVIDIA-Linux-x86_64-525.85.05.run --accept-license --ui=none --no-questions --no-kernel-module

