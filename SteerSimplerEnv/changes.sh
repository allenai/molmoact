#!/bin/bash

echo "=== Updating SimplerEnv Files ==="

##########################################
# 1. Replace main_inference.py
##########################################

SRC_MAIN="molmoact/SteerSimplerEnv/main_inference.py"
DST_MAIN="SimplerEnv/simpler_env/main_inference.py"

if [ ! -f "$SRC_MAIN" ]; then
    echo "ERROR: Source file not found: $SRC_MAIN"
    exit 1
fi

cp "$SRC_MAIN" "$DST_MAIN"
echo "✔ Replaced: $DST_MAIN"


##########################################
# 2. Copy maniskill2_evaluator_steer.py
##########################################

SRC_EVAL="molmoact/SteerSimplerEnv/maniskill2_evaluator_steer.py"
DST_EVAL_DIR="SimplerEnv/simpler_env/evaluation/"
DST_EVAL="$DST_EVAL_DIR/maniskill2_evaluator_steer.py"

mkdir -p "$DST_EVAL_DIR"

if [ ! -f "$SRC_EVAL" ]; then
    echo "ERROR: Source file not found: $SRC_EVAL"
    exit 1
fi

cp "$SRC_EVAL" "$DST_EVAL"
echo "✔ Copied evaluator to: $DST_EVAL"


##########################################
# 3. Copy molmoact_model_test.py
##########################################

SRC_MODEL="molmoact/SteerSimplerEnv/molmoact_model_test.py"
DST_MODEL_DIR="SimplerEnv/simpler_env/policies/molmoact/"
DST_MODEL="$DST_MODEL_DIR/molmoact_model_test.py"

mkdir -p "$DST_MODEL_DIR"

if [ ! -f "$SRC_MODEL" ]; then
    echo "ERROR: Source file not found: $SRC_MODEL"
    exit 1
fi

cp "$SRC_MODEL" "$DST_MODEL"
echo "✔ Copied model test to: $DST_MODEL"


##########################################
# 4. Copy molmoact_test.sh
##########################################

SRC_SCRIPT="molmoact/SteerSimplerEnv/molmoact_test.sh"
DST_SCRIPT_DIR="SimplerEnv/scripts/"
DST_SCRIPT="$DST_SCRIPT_DIR/molmoact_test.sh"

mkdir -p "$DST_SCRIPT_DIR"

if [ ! -f "$SRC_SCRIPT" ]; then
    echo "ERROR: Source file not found: $SRC_SCRIPT"
    exit 1
fi

cp "$SRC_SCRIPT" "$DST_SCRIPT"
chmod +x "$DST_SCRIPT"   # Make script executable
echo "✔ Copied test script to: $DST_SCRIPT"


echo "=== All updates completed successfully! ==="
