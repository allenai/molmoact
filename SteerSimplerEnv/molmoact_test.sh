#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Usage check
# -----------------------------------------------------------------------------
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <ckpt_path> <policy_model>"
  exit 1
fi

# -----------------------------------------------------------------------------
# Positional arguments
# -----------------------------------------------------------------------------
ckpt="$1"
policy="$2"

gpu_id=0
# env_name=PlayWithObjectsCustomInScene-v2
# env_name=PlayWithCubesCustomInSceneSteer-v0
# env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
# env_name=PlaceIntoClosedDrawerCustomInScene-v0
env_name=PlayWithCubesCustomInScene-v1
scene_name=google_pick_coke_can_1_v4
# coke_can_option="lr_switch=True"


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy} --ckpt-path ${ckpt} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 10000 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.235 -0.235 1 --obj-init-y 0.42 0.42 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 