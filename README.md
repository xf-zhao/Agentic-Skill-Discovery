
# Install Everything Inside Isaac Sim Docker

## Install necessary tools
```bash
apt-get update
apt-get install -y tmux zsh wget git python3 python3-pip vim
sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

## Configure Gitub authorization
```bash
ssh-keygen -t ed25519 -C "xfz.zhao@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copy the last line and add to SSH keys: https://github.com/settings/keys.

## Install orbit and zero-hero
```bash
cd orbit
export ISAACSIM_PATH=/isaac-sim
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ORBIT_ROOT_DIR=$(pwd)
export ZEROHERO_ROOT_DIR=$ORBIT_ROOT_DIR/zero-hero
export PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export OPENAI_API_KEY=

 /isaac-sim/kit/python/bin/python3 -m pip install "usd-core<24.00,>=21.11"
./orbit.sh --install
./orbit.sh --extra rsl_rl
ln -s ${ISAACSIM_PATH} _isaac_sim

cd zero-hero
git pull
pip3 install -r requirements.txt

```

## Minimal Test 

Launch `train.py` under `zero-hero` directory to examine whether oribt is successfully installed.
```bash
cd zero-hero
tmux
../orbit.sh -p rsl_rl/train.py --task Franka_Table --headless --num_envs 4096 --max_iterations 10
```

Normally, the training starts with a ~50k frames/s on a A100 gpu card.

# Minimal Training Scripts

## The `learn.py` to learn a single task with LLM+RL.
```bash
python3 learn.py task="Reach cube A." num_envs=4096 memory_requirement=32 min_gpu=90 temperature=0.7
```