
# Install Inside Isaac Sim Docker

```bash
apt-get update
apt-get install -y tmux zsh wget git python3 python3-pip vim
sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

cd orbit
export ISAACSIM_PATH=/isaac-sim
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ORBIT_ROOT_DIR=$(pwd)
export ZEROHERO_ROOT_DIR=$ORBIT_ROOT_DIR/zero-hero
export PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export OPENAI_API_KEY=

./orbit.sh --install
./orbit.sh --extra rsl_rl
ln -s ${ISAACSIM_PATH} _isaac_sim

cd zero-hero
pip3 install -r requirements.txt

```