# Revobots Arm

Core package for arm control, gamepad/leader teleoperation, cameras, RTSP streaming,
recording, replay, dataset conversion, and inference.

## Install

Requires Python 3.10 or newer:

```sh
python -m pip install -e ".[gamepad]"
python -m pip install -e <compatible-lerobot-checkout>
```

Install vendor SDKs for the selected robot. RealSense support is available with
`python -m pip install -e ".[realsense]"`. RTSP requires system GStreamer/PyGObject
on Linux/Jetson.

## Run

```sh
revobots-arm-teleop --preset nero_gamepad
revobots-arm-gamepad-client --ip=192.168.1.20 --port=5001
revobots-arm-cameras --help
revobots-arm-record --help
revobots-arm-replay --help
revobots-arm-convert --help
revobots-arm-inference --help
```

Use installed `revobots-arm-*` commands or `python -m revobots_arm.runtime.teleoperation`.
The old `src.*` imports and commands have been removed.

## Depth distance telemetry for Porter / Scout

RealSense always captures RGB. Set `use_depth=true` to add depth. When
`--stream.enabled=true` is also set, the camera owner automatically samples each
running RealSense depth bus at **3 Hz**. RGB goes to RTSP at the configured FPS.
For example, use a RealSense for RGB and depth alongside a USB RGB camera:

```sh
revobots-arm-teleop --preset nero_gamepad \
  --cameras.eye.source=/dev/video0 \
  --cameras.wrist2D.type=intelrealsense --cameras.wrist2D.source=CAMERA_SERIAL \
  --cameras.wrist2D.use_depth=true --stream.enabled=true \
  --stream.depth_robot_id=XAVIER_PORTER_001
```

With RTSP disabled, you can run the standalone worker alongside a camera owner
that has `use_depth=true` and `publish_frames=true`:

```sh
revobots-arm-depth --camera wrist2D --robot-id XAVIER_PORTER_001
```

The default destination is UDP `127.0.0.1:57110`. Automatic telemetry uses the
arm's configured ID by default. For Porter, set `--stream.depth_robot_id` (or
standalone `--robot-id`) to the **base's fleet identity**, enable
`arm_depth_enabled`, and match `arm_depth_camera` to the camera name. The base's
existing publisher sends the scalar result with its normal dashboard WebSocket
and Azure IoT messages under the base identity. This worker needs no Azure keys
and does not open a second fleet connection. Scout requires the equivalent
scalar receiver/snapshot integration; the reference SCOUT-XT tree is unchanged.

The measurement is the median of valid pixels in an 11x11 **depth-image centre**
patch. Zero and 65535 are excluded. The bus already contains millimetres, so no
raw-device depth scale is applied again. The arm depth is unaligned: this is
not a colour-crosshair measurement or a Euclidean range. Invalid samples clear
the distance immediately; no advancing frames for 1.5 seconds makes it stale.
New telemetry packets are sent at 3 Hz even when the bus is missing/stale, and
old frames are skipped rather than queued. The worker reattaches after a bus
restart. Colour/depth copies and resource contention still need target latency
measurement; zero additional system load is not guaranteed.

On different computers, the worker runs on the **camera-owning computer**
(shared memory is local). Set `--stream.depth_host` (or standalone `--host`) to
the base's private IPv4 address and
configure the receiver's bind address/network access. Keep clocks synchronized;
packets carry capture/sample timestamps. Do not expose the unauthenticated scalar
UDP port to public networks. Each receiver accepts one configured camera/identity.

Offline checks (synthetic arrays/shared memory; no cameras/cloud):

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Configuration

Schemas: `src/revobots_arm/config/schema.py`.
Presets: `src/revobots_arm/config/presets.py`.

Override settings with arguments such as `--robot.channel=can-nero`. For a deployment,
pass `--deployment /path/to/robot.json`, containing dotted keys such as
`{"robot.channel":"can-nero","fps":30}`. The environment variable
`REVOBOTS_ARM_DEPLOYMENT` can also select that file. CLI values override deployment
values, which override the preset.

The `nero_gamepad` and `so101_gamepad` presets start at 1.5× jog speed. Set
`--teleop.initial_speed_level=2` to start at 0.5×.

The Nero gamepad preset uses SocketCAN `can-nero`, firmware profile `v121`, startup
rest positioning, X/Y/B pick/place/look poses, LB open and RB close. Saved-pose
moves continue through controller dropout. Hardware behavior requires verification
on the target robot before deployment.

## Extend

Register native driver config/class import paths with `robots.registry.robots`;
register inputs with `teleoperators.registry.teleoperators`. Existing LeRobot types
use LeRobot's registry. Add robot presets in `config/presets.py` and declare joint
order, units and URDF links explicitly. Runtime orchestration remains shared.

```text
src/revobots_arm/
  assets/               Runtime URDFs
  config/               Schemas, presets, deployment and CLI loading
  robots/               Drivers, contracts and registration
  teleoperators/        Gamepad and leader inputs
  cameras/              OpenCV, RealSense and observation management
  streaming/            RTSP
  transport/            UDP and shared-memory frame bus
  inverse_kinematics/    Solver and robot profiles
  recording/            Session I/O and dataset conversion
  runtime/              Control, recording, replay and inference
  utils/                Shared helpers
```

Robot-specific integration modules are explicit:

```text
robots/koch/follower.py
robots/so101/follower.py
robots/nero/driver.py       # native Nero control
robots/nero/follower.py     # LeRobot retargeting follower
teleoperators/leader/koch.py
teleoperators/leader/so101.py
```

Koch and SO101 modules delegate calibration and motor I/O to LeRobot. Each leader
module exposes local and remote adapters; the shared leader client routes through
these modules. SO101 supports both combined (`so_follower`/`so_leader`) and separate
(`so101_follower`/`so101_leader`) LeRobot package layouts. The direct joint-copy
remote mode uses the corresponding LeRobot remote leader extension.

## Leader-to-follower Cartesian IK

Enable with `--ik=true` (also `--ik.enabled=true`). Leader joint angles go through
leader FK, and the change in tool pose goes through follower IK. The first
paired reading anchors both tools at their current poses. Nero uses the native
seven-joint driver in this mode and skips the startup rest move. The same
solver can use all seven joints, and the motion it chooses depends on the target
pose. The gripper holds its measured setting when the follower exposes one.

For a local SO-101 leader driving Nero:

```sh
revobots-arm-teleop --preset nero --ik=true \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=aadi_so101_leader
```

For a remote SO-101 leader, run the following on the follower and leader PCs,
respectively. Use the follower PC's IP and the leader's calibration ID:

```sh
revobots-arm-teleop --preset nero --ik=true --teleop.type=so101_leader_remote --teleop.port=5002
revobots-arm-leader-client --preset nero --ik=true --teleop.type=so101_leader_remote \
  --ip 192.168.1.20 --port 5002 --com /dev/ttyACM1 --id aadi_so101_leader
```

Koch v1.1 has a bundled URDF with the jaw pivot as the tool point. For a local
Koch leader driving Nero:

```sh
revobots-arm-teleop --preset nero --ik=true \
  --teleop.type=koch_leader --teleop.port=/dev/ttyACM0 --teleop.id=rahul \
  --ik.leader_calibration=rahul
```

For remote Koch, use the updated package on both PCs. Run these commands on
the Nero and leader PCs respectively:

```sh
revobots-arm-teleop --preset nero --ik=true
revobots-arm-leader-client --preset nero --ik=true --ik.leader_calibration=rahul \
  --ip 192.168.1.20 --port 5001 --com /dev/ttyACM0 --id rahul
```

Use `--ik.leader_calibration=rahul` for that measured leader. For a different
Koch calibration, configure the joint zeros and directions on the leader PC.
The client applies those settings and sends calibrated URDF joint degrees.
Koch-to-Nero uses the nominal Nero gripper tip, aligns the two starting tool
headings, and defaults to 2.75× XYZ displacement. This mode anchors to current
feedback and sends targets through the regular Nero driver.

For other rotary serial arms, supply the relevant `--ik.leader_urdf` or
`--ik.follower_urdf` and `*_tip_link`. Specify `*_base_link` if the chain starts
below the URDF root. If the driver labels differ from the URDF joint names,
set `*_motors` to the driver names in base-to-tip order. `*_invert_joints` and
`*_joint_offsets_deg` align each arm's joint feedback with its own URDF. For a
remote leader, pass the `leader_*` direction and offset settings to
`revobots-arm-leader-client`. A custom driver must use calibrated degrees for
arm joints.

Other pairs default to aligned base frames and 1:1 translation. Set
`--ik.frame_rpy_deg=0,0,0` to select aligned axes explicitly, or use
`--ik.frame_rpy_deg=0,0,180` when a 180° base rotation is needed. Set
`--ik.position_scale=0.5` for half-size motions. `--ik.orientation_weight=0`
follows XYZ only. For continuous data
capture, add `--record.continuous=true` to the teleop command.

## Continuous recording and episode annotation

Record immediately after teleoperation becomes ready, without pressing Space:

```sh
revobots-arm-teleop --preset nero_gamepad --record.continuous=true --run_dir /path/to/continuous_run
```

In continuous mode Space and redo are ignored. Esc/q or Ctrl-C finalizes the capture.
Every recorded control tick includes observations and camera/depth frames. During
controller dropout, the recorded action is the measured hold pose and `command_sent`
is false; no new robot command is transmitted. Timestamps retain actual capture timing,
while preview playback uses the session's nominal FPS, as the existing format does.

After the recording is finalized:

```sh
revobots-arm-episode-annotate /path/to/continuous_run
# A single session folder or its data.jsonl path also works.
revobots-arm-episode-annotate /path/to/continuous_run/session_0 --output /path/to/curated --port 8765
```

The CLI opens a local browser dashboard. Use `--no-browser` to open its printed URL
manually, or `--port 0` to choose a free port. It binds only to 127.0.0.1 and requires
the per-launch token in that URL. No account or additional web framework is needed.

- Drag/click the timeline to scrub. Left/right step a frame; Shift + arrows jump one second.
- Space marks start, then the last included frame; the on-screen button does the same.
- K or Play toggles playback. Esc cancels an unfinished mark.
- Select a camera or enable all-camera preview; depth is colorized only for display.
- Edit episode boundaries and task labels; changes autosave to `annotations.json` beside
  the source metadata (in the run folder when opened as a run). Undo reverses edits.
  Overlapping ranges are rejected. Export waits for pending edits to save.
- **Export sessions** writes a new directory containing `session_0`, `session_1`, etc.
  Sources stay intact. Every camera is exported; depth maps are copied losslessly.
  Color videos are decoded/re-encoded as MP4. Rows retain source frame indices and
  timestamps; episode-local indices start at zero. Internal ranges are [start, end).
- **Export + convert to LeRobot** calls the existing converter with the entered dataset
  ID, saving a local dataset. It requires the compatible LeRobot environment. It never
  uploads to Hugging Face. Conversion failure leaves the exported sessions available.

By default, exports go into a sibling `<source>_annotated` directory, with a new
unique subdirectory per export. You can also convert exported sessions separately:

```sh
revobots-arm-convert --sessions_dir /path/to/exported/session_directory --dataset.repo_id=username/dataset-name
```

Opening a recording validates metadata/video/depth frame counts. Finish recording
before annotation; modified sources are rejected rather than silently reusing old marks.

Annotation exports use `session_<YYYYmmdd_HHMMSS>_<suffix>/`, containing the selected
`session_0/`, `session_1/`, etc. Continuous teleoperation prints a ready-to-run
`revobots-arm-episode-annotate "<saved-session-path>"` command after saving.
