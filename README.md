
# Install Everything Inside Isaac Sim Docker

## Test whether IsaacSim docker works

### Local docker test
1. Pull docker image and run
```
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" 134.100.39.10:32000/isaacsim:2023.1.1
```

Reference: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim

2. Inside the launched docker container, execute
```bash
cd /isaac-sim
./runheadless.native.sh
```
If normal, no obvious error.

Extra information:
```text
NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4
```

### k8s Pod Test

1. Create `yaml` with 
```bash
./mlpod.py --yaml isaacsim2.yaml --user gaede --pod isaacsim --image 134.100.39.10:32000/isaacsim:2023.1.1 --gpumem 60 --env ACCEPT_EULA=Y PRIVACY_CONSENT=Y -- /bin/bash
```
2. Edit `isaacsim2.yaml` by replacing the two `Y` with `"Y"` (as string instead of as bool value).

3. Create pod with the `yaml` file
```bash
alias kubectl k
k create -f isaacsim2.yaml
k attach -it isaacsim2
```

4. Inside the attached pod, run
```bash
cd /isaac-sim
./runheadless.native.sh
```

May run into errors:
```text
2024-04-16 08:45:58 [1,012ms] [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
2024-04-16 08:45:58 [1,032ms] [Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
2024-04-16 08:45:58 [1,032ms] [Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
2024-04-16 08:45:58 [1,032ms] [Error] [gpu.foundation.plugin] carb::graphics::createInstance failed.
2024-04-16 08:45:58 [1,608ms] [Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
2024-04-16 08:45:58 [1,608ms] [Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
2024-04-16 08:45:58 [1,608ms] [Error] [gpu.foundation.plugin] carb::graphics::createInstance failed.
2024-04-16 08:45:59 [2,155ms] [Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode.
```

Extra information:
```text
NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4
```

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
export ZEROHERO_ROOT_DIR=$ORBIT_ROOT_DIR/zero_hero
export PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export OPENAI_API_KEY=

 /isaac-sim/kit/python/bin/python3 -m pip install "usd-core<24.00,>=21.11"
./orbit.sh --install
./orbit.sh --extra rsl_rl
ln -s ${ISAACSIM_PATH} _isaac_sim

cd zero_hero
git pull
pip3 install -r requirements.txt

```

## Minimal Test 

Launch `train.py` under `zero_hero` directory to examine whether oribt is successfully installed.
```bash
../orbit.sh -p rsl_rl/train.py --task Franka_Table --headless --num_envs 4096 --max_iterations 10
```

Normally, the training starts with a ~50k frames/s on a A100 gpu card.

# Minimal Training Scripts

## The `learn.py` to learn a single task with LLM+RL.
```bash
python3 learn.py task="Reach cube A." num_envs=4096 memory_requirement=32 min_gpu=90 temperature=0.7
```