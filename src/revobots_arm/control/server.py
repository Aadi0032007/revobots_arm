"""Open the local browser console for the arm and its cameras."""

import logging
import os
from pathlib import Path
import webbrowser

from revobots_arm.config.devices import load_config, make_parser
from revobots_arm.config.presets import PRESET_ENV_VAR
from revobots_arm.grasp.server import create_server

from .session import ArmConsole


def main():
    parser = make_parser(__doc__)
    parser.epilog = ("Pass dotted overrides such as --robot.channel=can0 or --cameras.wrist.source=0. "
                     "Set leader and gamepad UDP ports in Setup or use --teleop.port and --teleop.udp_port.")
    parser.add_argument("--execute", action="store_true", help="Allow live robot commands")
    parser.add_argument("--port", type=int, default=8766, help="Loopback port (0 selects a free port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL only")
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        preset = args.preset or os.environ.get(PRESET_ENV_VAR) or "nero_gamepad"
        cfg = load_config(preset, overrides, args.deployment, require_lerobot=False)
        if not 0 <= args.port <= 65535:
            raise ValueError("Use port 0–65535")
        session = ArmConsole(cfg, live=args.execute)
        try:
            with create_server(session, args.port, console_assets=Path(__file__).with_name("web")) as server:
                url = f"http://127.0.0.1:{server.server_port}/#token={session.token}"
                print(f"Arm control ({'live' if args.execute else 'preview'}): {url}", flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            session.close()
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Arm control: {exc}\n")


if __name__ == "__main__":
    main()
