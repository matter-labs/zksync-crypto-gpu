#!/bin/sh
export USERNAME=$(whoami)
DEV_HOME=$HOME/dev/matter-labs
BASH_VARS_FILE=~/.bash_vars
echo "export DEV_HOME=\$DEV_HOME" >> ~/.bashrc

mkdir -pv $DEV_HOME

echo username: $USERNAME
echo dev home: $DEV_HOME

sudo apt update
sudo apt install libclang-dev build-essential -y
sudo snap install cmake --classic

# rust
curl https://sh.rustup.rs -sSf | sh -s -- -y

. ~/.bashrc
cargo --version


# bellman-cuda
cd ~/dev/matter-labs
git clone https://github.com/matter-labs/era-bellman-cuda --branch rr/permutation-3-cols
echo "export BELLMAN_CUDA_DIR=~/dev/matter-labs/era-bellman-cuda" >> $BASH_VARS_FILE
cd era-bellman-cuda
git submodule update --init --recursive
# devices other than L4
#cmake -B./buildd -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake -B./buildd -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --bulid ./buildd
./buildd/tests/tests

 
# cuda
cd ~/
wget -nc https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run
sudo sh cuda_12.6.1_560.35.03_linux.run --silent --driver --toolkit --override --ui=none --no-questions --accept-license
echo "PATH=\$PATH:/usr/local/cuda/bin" >> $BASH_VARS_FILE
echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> $BASH_VARS_FILE


# CRS
cd $DEV_HOME
git clone https:://github.com/matter-labs/zksync-crypto-gpu --branch proof-compression
cd proof-compression
IGNITION_TRANSCRIPT_PATH=$PWD/data cargo test download_crs --release -- --nocapture
IGNITION_TRANSCRIPT_PATH=$PWD/data cargo test transform_crs --release -- --nocapture
rm -rf transcript**