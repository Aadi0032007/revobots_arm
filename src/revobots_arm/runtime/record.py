"""Timed recording, ``lerobot-record`` style (alternative to ``teleop.py``).

Same arguments as lerobot-record:

    python -m revobots_arm.runtime.record --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader \\
                         --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.single_task="Pick the cube" \\
                         --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \\
                         --display_data=true
    python -m revobots_arm.runtime.record --preset so101 --resume            # append to the latest run

Keys:  → or n  next episode     ← or r  re-record     Esc or q  stop

Drives LeRobot's real ``record_loop`` with a ``recorder.SessionWriter`` as the sink, so
the output is raw sessions (convert them with ``convert.py``), not a LeRobotDataset.
Cameras come from ``cameras.py`` through ``RobotWithCameras``, exactly as in ``teleop.py``.
"""

from __future__ import annotations


import logging
from pathlib import Path

from revobots_arm.cameras.manager import CamerasManager, RobotWithCameras
from revobots_arm.config import ArmConfig
from revobots_arm.config.devices import (
    add_bool_flag,
    describe,
    load_config,
    make_parser,
    make_robot,
    make_teleop,
)
from revobots_arm.recording.session import SessionWriter, next_session_dir
from revobots_arm.runtime.teleoperation import build_features, resolve_run_dir

logger = logging.getLogger(__name__)


def record(cfg: ArmConfig, run_dir: Path) -> list[Path]:
    if cfg.ik.enabled:
        raise SystemExit("Use revobots-arm-teleop --ik=true --record.continuous=true for leader IK recording")
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.keyboard_input import init_keyboard_listener
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import (
        init_visualization,
        shutdown_visualization,
    )

    rec = cfg.record
    fps = cfg.fps
    task = cfg.task
    play_sounds = rec.play_sounds
    compress_images = rec.display_compressed_images or (
        rec.display_data and rec.display_ip is not None and rec.display_port is not None
    )

    cameras = CamerasManager(cfg)
    cameras.start()
    robot = RobotWithCameras(make_robot(cfg), cameras, stale_s=rec.frame_stale_s)
    teleop = make_teleop(cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    features = build_features(robot)

    base_meta = {
        "robot_type": robot.name,
        "robot_id": robot.id,
        "teleop_type": cfg.teleop_type,
        "task": task,
        "preset": cfg.name,
    }
    if rec.display_data:
        init_visualization(
            rec.display_mode,
            session_name="revobots_arm_record",
            ip=rec.display_ip,
            port=rec.display_port,
        )

    saved: list[Path] = []
    listener = None
    try:
        robot.connect()
        teleop.connect()
        if getattr(
            teleop, "needs_seed", False
        ):  # gamepad jog starts from the arm's current pose
            teleop.seed_from_observation(robot.get_observation())
        listener, events = init_keyboard_listener()

        recorded = 0
        while recorded < rec.num_episodes and not events["stop_recording"]:
            session_dir = next_session_dir(run_dir)
            session = SessionWriter(
                session_dir,
                fps=fps,
                features=features,
                meta={**base_meta, "session_name": session_dir.name},
                codec=rec.codec,
            )
            logging.info("Recording to %s", session_dir)
            log_say(f"Recording {session_dir.name}", play_sounds)
            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=session,
                    teleop=teleop,
                    control_time_s=rec.episode_time_s,
                    single_task=task,
                    display_data=rec.display_data,
                    display_mode=rec.display_mode,
                    display_compressed_images=compress_images,
                )
            except BaseException:
                session.close(discard=True)
                raise

            if not events["stop_recording"] and (
                recorded < rec.num_episodes - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment", play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=None,
                    teleop=teleop,
                    control_time_s=rec.reset_time_s,
                    single_task=task,
                    display_data=rec.display_data,
                    display_mode=rec.display_mode,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                session.close(discard=True)
                continue

            session.close()
            saved.append(session_dir)
            recorded += 1
            logging.info("Saved %d frames -> %s", session.num_frames, session_dir)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        log_say("Stop recording", play_sounds, blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        cameras.stop()
        if listener is not None:
            listener.stop()
        if rec.display_data:
            shutdown_visualization(rec.display_mode)

    return saved


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument(
        "--run_dir", default=None, help="Append sessions to this run folder"
    )
    add_bool_flag(
        parser, "--resume", "Append sessions to the most recent run under sessions_root"
    )
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides, deployment=args.deployment)

    from lerobot.utils.utils import init_logging

    init_logging()
    logging.info("\n%s", describe(cfg))

    run_dir = resolve_run_dir(cfg, args.run_dir, args.resume)
    saved = record(cfg, run_dir=run_dir)

    if saved:
        print(f"\n{len(saved)} session(s) saved under {run_dir}")
        print(
            f'Next: python -m revobots_arm.recording.conversion --sessions_dir "{run_dir}" --dataset.repo_id={cfg.dataset.repo_id}'
        )
    else:
        print("\nNo recordings saved.")


if __name__ == "__main__":
    main()
