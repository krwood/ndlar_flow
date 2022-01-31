#!/bin/bash
# Demonstrates how to install module0_flow into an environment for a CORI login node
# (must be run from a directory containing h5flow and module0_flow)
#
# Usage:
#   source install_module0_flow-login_node.sh <conda environment to create>
#

#ENV_NAME=$1
ENV_NAME=ndlar_flow
CWD=`pwd`

H5FLOW_INSTALL='/global/homes/k/kwood/dunend/software/h5flow'
NDLARFLOW_INSTALL=$CWD

conda deactivate

echo "Installing h5flow..."
cd $H5FLOW_INSTALL
conda env create -n $ENV_NAME -f environment-nompi.yml
conda activate $ENV_NAME
pip install .

echo "Installing ndlar_flow..."
cd $NDLARFLOW_INSTALL
conda env update -n $ENV_NAME -f env-nompi.yaml
pip install .

echo "Done!"
cd $CWD

