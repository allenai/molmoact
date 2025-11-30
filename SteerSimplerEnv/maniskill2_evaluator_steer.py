"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import time
import numpy as np
from transforms3d.euler import quat2euler
import cv2
import glob
# import keyboard
# import torch
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

import multiprocessing as mp
from simpler_env.utils.traj_interface import RemoteTrajServer
if mp.current_process().name == "MainProcess":
    remote = RemoteTrajServer(port=5002)   # <-- only once
else:
    remote = None

import os
# if os.environ.get("RANK", "0") == "0":
#     import debugpy
#     # Listen on port 5678 (adjust if needed)
#     debugpy.listen(("172.17.0.1", 5678))
#     print("Debugger is listening on port 5678. Waiting for client to attach...")
#     debugpy.wait_for_client()
import faulthandler
faulthandler.enable()

def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
   
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
        

    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    # print("Reset env with options:", env_reset_options)
    obs, _ = env.reset(options=env_reset_options)
    # print("Reset success")

    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.unwrapped.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.unwrapped.get_language_instruction()
        
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    annotated_frames = []
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    traj = None

    # Create directory for saving frames
    # frames_dir = "/weka/oe-training-default/jiafeid/SimplerEnv/simpler_env/frames_steer" +++++++++
    frames_dir = "./frames_steer"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Clear existing frames in the directory (optional)
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)

    # Step the environment
    while not (predicted_terminated or truncated):
        if timestep % 10 == 0 and remote is not None:
            try:
                ans = input(f"[t={timestep}] Draw trajectory remotely? [y/n] ").strip().lower()
            except EOFError:                     # non-interactive run
                ans = "n"

            if ans == "y":
                if traj:
                    try:
                        ans_2 = input(f"[t={timestep}] Reuse previous trajectory? [y/n] ").strip().lower()
                    except EOFError:                     # non-interactive run
                        ans_2 = "n"
                
                if not traj or ans_2 == "n":
                    # blocks until the browser posts points or timeout hits
                    traj = remote.request_traj(image, timeout=120)
                    print("Received traj:", traj)
                    print(type(traj))
            
            else:
                traj = None
            
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        try:
            raw_action, action, annotated_image = model.step(image, task_description, traj)
        except Exception as e:
            print(f"[Fail-safe triggered] model.step() failed at timestep {timestep}: {e}")
            
            # Fail-safe defaults
            raw_action = {
                "world_vector": np.array([0.0, 0.0, 0.0]),
                "rotation_delta": np.array([0.0, 0.0, 0.0]),
                "open_gripper": np.array([1.0]),
            }
            annotated_image = image.copy()

            action = {
                "world_vector": raw_action["world_vector"],
                "rot_axangle": np.array([0.0, 0.0, 0.0]),
                "gripper": np.array([1.0]),
                "terminate_episode": np.array([0.0]),
            }
        # predicted_actions.append(raw_action) ++++++++++DEBUG MARK
        predicted_actions.append(action)


        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.unwrapped.advance_to_next_subtask()

        annotated_frames.append(annotated_image)
        
        # Save frame with timestep number
        frame_filename = f"frame_{timestep:04d}.png"
        frame_path = os.path.join(frames_dir, frame_filename)

        ##################################


        (h, w) = annotated_image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 0, 0)  
        thickness = 1
        
        if not traj:
            text1 = "model reasoning"
        else:
            text1 = "human steering"

        # font settings
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1               # small size
        thickness  = 1
        color      = (255, 255, 255)   # white
        line_type  = cv2.LINE_AA

        # positions (x, y) in pixels
        x, y0 = 5, 30
        line_height = 20

        # draw the two lines using the text strings
        cv2.putText(annotated_image, text1, (x, y0),               font, font_scale, color, thickness, line_type)

        ##################################


        cv2.imwrite(
            frame_path,
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )
        print(f"Saved frame {timestep} to {frame_path}")

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.unwrapped.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.unwrapped.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    annotated_video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/annotated_{video_name}"
    annotated_video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, annotated_frames, fps=5)
    # write_video(annotated_video_path, annotated_frames, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    
    # try:

    #     torch.cuda.empty_cache()
    #     print("Torch GPU cache cleared.")
    # except Exception as e:
    #     print("Error clearing torch cache:", e)
        
    # import gc
    # gc.collect()
    
    
    # try:
    #     del predicted_actions, images, annotated_frames
    # except Exception as e:
    #     print("Error deleting variables:", e)
        
        
        
    # try:
    #     if hasattr(env, "close"):
    #         env.close()
    #     if hasattr(model, "close"):
    #         model.close()
    # except Exception as cleanup_error:
    #     print(f"Error during cleanup: {cleanup_error}")
    
    # # Delay to allow asynchronous cleanup routines to finish
    # time.sleep(4)
    
    return success == "success"


# def maniskill2_evaluator(model, args):
def maniskill2_evaluator_steer(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # strip trailing slash & take final segment
    ckpt_path_basename = args.ckpt_path.rstrip("/").split("/")[-1]
    additional_kwargs = args.additional_env_build_kwargs or {}

    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:

                # === exactly how run_maniskill2_eval_single_episode builds env_save_name
                env_save_name = args.env_name
                for k, v in additional_kwargs.items():
                    env_save_name += f"_{k}_{v}"
                if args.additional_env_save_tags is not None:
                    env_save_name += f"_{args.additional_env_save_tags}"

                # rgb-overlay tag
                if args.rgb_overlay_path is not None:
                    rgb_overlay_path_str = os.path.splitext(
                        os.path.basename(args.rgb_overlay_path)
                    )[0]
                else:
                    rgb_overlay_path_str = "None"

                # robot orientation
                r, p, y = quat2euler(robot_init_quat)

                # === the folder into which your videos land
                video_dir = os.path.join(
                    args.logging_dir,
                    ckpt_path_basename,
                    args.scene_name,
                    control_mode,
                    env_save_name,
                    f"rob_{robot_init_x}_{robot_init_y}"
                    f"_rot_{r:.3f}_{p:.3f}_{y:.3f}"
                    f"_rgb_overlay_{rgb_overlay_path_str}"
                )

                # ——— XY variation mode ———
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            # build the two patterns
                            success_pat = os.path.join(
                                video_dir,
                                f"success_obj_{obj_init_x}_{obj_init_y}_*.mp4"
                            )
                            failure_pat = os.path.join(
                                video_dir,
                                f"failure_obj_{obj_init_x}_{obj_init_y}_*.mp4"
                            )

                            print(f"[CHECK] {success_pat}")
                            print(f"[CHECK] {failure_pat}")

                            matches = glob.glob(success_pat) + glob.glob(failure_pat)
                            if matches:
                                for m in matches:
                                    print(f"[SKIP] Found existing video: {m}")
                                continue

                            # if we get here, no existing video → actually run
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    model=model,
                                    ckpt_path=args.ckpt_path,
                                    robot_name=args.robot,
                                    env_name=args.env_name,
                                    scene_name=args.scene_name,
                                    robot_init_x=robot_init_x,
                                    robot_init_y=robot_init_y,
                                    robot_init_quat=robot_init_quat,
                                    control_mode=control_mode,
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    additional_env_build_kwargs=additional_kwargs,
                                    rgb_overlay_path=args.rgb_overlay_path,
                                    obs_camera_name=args.obs_camera_name,
                                    control_freq=args.control_freq,
                                    sim_freq=args.sim_freq,
                                    max_episode_steps=args.max_episode_steps,
                                    instruction=None,
                                    enable_raytracing=args.enable_raytracing,
                                    additional_env_save_tags=args.additional_env_save_tags,
                                    logging_dir=args.logging_dir,
                                )
                            )

                # ——— episode-ID variation mode ———
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_pat = os.path.join(
                            video_dir,
                            f"success_obj_episode_{obj_episode_id}_*.mp4"
                        )
                        failure_pat = os.path.join(
                            video_dir,
                            f"failure_obj_episode_{obj_episode_id}_*.mp4"
                        )

                        print(f"[CHECK] {success_pat}")
                        print(f"[CHECK] {failure_pat}")

                        matches = glob.glob(success_pat) + glob.glob(failure_pat)
                        if matches:
                            for m in matches:
                                print(f"[SKIP] Found existing video: {m}")
                            continue

                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                model=model,
                                ckpt_path=args.ckpt_path,
                                robot_name=args.robot,
                                env_name=args.env_name,
                                scene_name=args.scene_name,
                                robot_init_x=robot_init_x,
                                robot_init_y=robot_init_y,
                                robot_init_quat=robot_init_quat,
                                control_mode=control_mode,
                                obj_episode_id=obj_episode_id,
                                additional_env_build_kwargs=additional_kwargs,
                                rgb_overlay_path=args.rgb_overlay_path,
                                obs_camera_name=args.obs_camera_name,
                                control_freq=args.control_freq,
                                sim_freq=args.sim_freq,
                                max_episode_steps=args.max_episode_steps,
                                instruction=None,
                                enable_raytracing=args.enable_raytracing,
                                additional_env_save_tags=args.additional_env_save_tags,
                                logging_dir=args.logging_dir,
                            )
                        )

                else:
                    raise NotImplementedError(
                        f"Unknown obj_variation_mode: {args.obj_variation_mode}"
                    )

    return success_arr