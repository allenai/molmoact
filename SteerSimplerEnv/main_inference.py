import os

import numpy as np

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.evaluation.maniskill2_evaluator_steer import maniskill2_evaluator_steer

TEST = False

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


try:
    from simpler_env.policies.molmoact.molmoact_model import MolmoActInference
    from simpler_env.policies.molmoact.molmoact_model_vllm import MolmoActInferenceVLLM
    # ++++++
    from simpler_env.policies.molmoact.molmoact_model_test import MolmoActInferenceTEST
except ImportError as e:
    print("MolmoAct is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # CRITICAL FIX: For molmoact-vllm models, we need to initialize PyTorch/vLLM BEFORE TensorFlow
    # Otherwise TensorFlow fails to register GPU devices and leaves CUDA in a bad state
    if "molmoact" in args.policy_model and "vllm" in args.policy_model:
        # Initialize PyTorch CUDA first
        import torch
        if torch.cuda.is_available():
            print(f"Initializing PyTorch CUDA first: {torch.cuda.device_count()} device(s) available")
            _ = torch.cuda.current_device()
    
    # Now import TensorFlow
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif "molmoact" in args.policy_model:
        # ++++++++
        if "test" in args.policy_model:
            model = MolmoActInferenceTEST(
                saved_model_path = args.ckpt_path,
                policy_setup = args.policy_setup,
            )
        elif "hf" in args.policy_model or "vllm" not in args.policy_model:
            model = MolmoActInference(
                saved_model_path = args.ckpt_path,
                policy_setup = args.policy_setup,
            )
        elif "vllm" in args.policy_model:
            model = MolmoActInferenceVLLM(
                saved_model_path = args.ckpt_path,
                policy_setup = args.policy_setup,
            )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    if TEST :
        success_arr = maniskill2_evaluator_steer(model, args)
    else :
        success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
