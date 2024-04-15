FROM 134.100.39.10:32000/isaacsim:2023.1.1
RUN apt-get update && apt-get install -y build-essential tmux zsh wget git python3 python3-pip vim
