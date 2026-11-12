from __future__ import annotations
import logging
import threading
import time
from typing import Any, Optional
import cv2
from revobots_arm.config import ArmConfig, CameraConfig

logger = logging.getLogger(__name__)


class RtspServer:
    """GstRtspServer on a background GLib thread (the AGV ``LAB/cameras.py`` ``RtspServer``).

    Mounts come from ``ArmConfig.stream_mounts()``:

        /<stream.mount>   ``stream.camera`` (default the first camera)   ← MediaMTX pulls this
        /<name>           every other colour camera

    Each mount is ``appsrc → (x264enc | nvv4l2h264enc) → rtph264pay``, shared by all
    clients and only built while a client pulls it. Its feeder (``_BusAppsrcFeeder``)
    reads the camera's frame bus, so streaming never touches the capture thread.
    """

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self._stream = cfg.stream
        self._mounts: dict[str, CameraConfig] = {}
        self._feeders: set[_BusAppsrcFeeder] = set()
        self._feeders_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Any = None  # GLib.MainLoop
        self._server: Any = None  # GstRtspServer.RTSPServer
        self._stop = threading.Event()

    @property
    def mounts(self) -> dict[str, CameraConfig]:
        return dict(self._mounts)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the server thread and the CLOSE_WAIT monitor. False when the config
        cannot be streamed (logged); a missing GStreamer is logged by the server thread."""
        try:
            self._mounts = self._cfg.stream_mounts()
        except ValueError as exc:
            logger.error("rtsp: %s; stream disabled", exc)
            return False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rtsp-server"
        )
        self._thread.start()
        threading.Thread(
            target=self._watchdog_loop, daemon=True, name="rtsp-watchdog"
        ).start()
        return True

    def stop(self) -> None:
        self._stop.set()
        with self._feeders_lock:
            feeders = list(self._feeders)
            self._feeders.clear()
        for feeder in feeders:
            feeder.stop()
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:  # noqa: BLE001
                pass

    # ── GLib main loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import GLib, Gst, GstRtspServer
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rtsp: GStreamer RTSP server unavailable (%s); stream disabled. "
                "It needs PyGObject and gst-rtsp-server (Linux / Jetson).",
                exc,
            )
            return

        s = self._stream
        Gst.init(None)
        # RTSPServer(), not RTSPServer.new(): the AGV construction known to work with MediaMTX pulls.
        self._server = GstRtspServer.RTSPServer()
        if s.bind and s.bind not in ("0.0.0.0", ""):
            self._server.set_address(s.bind)
        self._server.set_service(str(s.port))
        mount_points = self._server.get_mount_points()

        encoder = "nvv4l2h264enc" if s.hw_encode else "x264enc"
        for mount, cam in self._mounts.items():
            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(self._pipeline(cam))
            factory.set_shared(True)
            factory.connect("media-configure", self._on_media_configure, mount, cam)
            mount_points.add_factory(f"/{mount}", factory)
            logger.info(
                "rtsp: mount /%s ← camera %s (frame bus lab_%s, %s %s kbps)",
                mount,
                cam.name,
                cam.name,
                encoder,
                s.bitrate_kbps,
            )

        if not self._server.attach(None):
            logger.error(
                "rtsp: cannot listen on %s:%s (port in use?); stream disabled",
                s.bind,
                s.port,
            )
            return
        logger.info("rtsp: listening on %s:%s", s.bind, s.port)

        # GstRtspServer does not expire dead sessions on its own.
        GLib.timeout_add_seconds(20, self._cleanup_sessions)

        self._loop = GLib.MainLoop()
        if self._stop.is_set():
            return
        try:
            self._loop.run()
        except Exception as exc:  # noqa: BLE001
            logger.error("rtsp: main loop crashed: %s", exc)

    def _cleanup_sessions(self) -> bool:
        try:
            self._server.get_session_pool().cleanup()
        except Exception:  # noqa: BLE001
            pass
        return True

    # ── pipeline ─────────────────────────────────────────────────────────────

    @staticmethod
    def _caps(cam: CameraConfig) -> str:
        h, w, _ = cam.frame_shape
        return f"video/x-raw,format=BGR,width={w},height={h},framerate={cam.fps}/1"

    def _pipeline(self, cam: CameraConfig) -> str:
        """The AGV USB pipeline: SPS/PPS every second (``config-interval=1``) so MediaMTX
        can join mid-stream, byte-stream caps so h264parse / rtph264pay never renegotiate,
        and a leaky queue so a slow client cannot back-pressure the encoder."""
        s = self._stream
        if s.hw_encode:
            # Jetson NVENC: BGR → NV12 → NVMM. insert-sps-pps=1 puts SPS/PPS on every IDR.
            encoder = (
                "videoconvert ! video/x-raw,format=NV12 ! "
                "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                f"nvv4l2h264enc bitrate={int(s.bitrate_kbps) * 1000} "
                "preset-level=1 profile=0 insert-sps-pps=1 iframeinterval=30 "
            )
        else:
            encoder = (
                "videoconvert ! "
                "x264enc tune=zerolatency speed-preset=superfast "
                f"bitrate={int(s.bitrate_kbps)} key-int-max=30 byte-stream=true ! "
                "video/x-h264,profile=baseline,stream-format=byte-stream,alignment=au "
            )
        return (
            "appsrc name=bus_src is-live=true format=time do-timestamp=true block=false "
            f"! {self._caps(cam)} "
            f"! {encoder}"
            "! h264parse config-interval=1 "
            "! queue leaky=downstream max-size-buffers=4 "
            "! rtph264pay name=pay0 pt=96 config-interval=1"
        )

    def _on_media_configure(
        self, _factory: Any, media: Any, mount: str, cam: CameraConfig
    ) -> None:
        """Start a feeder whenever a media pipeline is built (shared: one per pull, rebuilt
        after the last client leaves) and stop it when that pipeline is torn down. The caps
        also go on the appsrc element: some GStreamer versions need them there."""
        appsrc = media.get_element().get_by_name("bus_src")
        if appsrc is None:
            logger.error("rtsp /%s: appsrc not found in the pipeline", mount)
            return
        try:
            from gi.repository import Gst

            appsrc.set_property("caps", Gst.Caps.from_string(self._caps(cam)))
            appsrc.set_property("format", Gst.Format.TIME)
            appsrc.set_property("is-live", True)
            appsrc.set_property("do-timestamp", True)
            appsrc.set_property("block", False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rtsp /%s: appsrc property set failed: %s", mount, exc)

        feeder = _BusAppsrcFeeder(mount, cam, appsrc, self._stream.max_repeated_frames)
        with self._feeders_lock:
            if self._stop.is_set():
                return
            self._feeders.add(feeder)
        feeder.start()

        def _unprepared(_media: Any) -> None:
            with self._feeders_lock:
                self._feeders.discard(feeder)
            feeder.stop()

        media.connect("unprepared", _unprepared)

    # ── CLOSE_WAIT monitor ───────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """The AGV CLOSE_WAIT check, logging only (see ``StreamConfig.close_wait_max``)."""
        s = self._stream
        warned = False
        while not self._stop.wait(timeout=max(1.0, float(s.close_wait_interval_s))):
            stuck = _count_close_wait(int(s.port))
            over = stuck > s.close_wait_max
            if over and not warned:
                logger.error(
                    "rtsp: %d CLOSE_WAIT connections on :%s, the server may be wedged. "
                    "Restart teleop between sessions to clear it.",
                    stuck,
                    s.port,
                )
            warned = over


class _BusAppsrcFeeder:
    """Pushes one camera's frame bus into a mount's appsrc at the camera's fps (the AGV
    ``_UsbAppsrcFeeder``, reading ``lab_<name>`` instead of the in-process holder).

    Every tick sends the newest frame. When the camera dropped or skipped one,
    ``HoldLastReader`` hands back the previous frame so the stream keeps its rate; after
    ``max_repeats`` repeats in a row without a fresh frame the feeder stops and logs the
    camera and frame bus. appsrc is non-blocking: when the encoder falls behind the frame
    is dropped there, and capture never waits.
    """

    def __init__(
        self, mount: str, cam: CameraConfig, appsrc: Any, max_repeats: int
    ) -> None:
        self.mount = mount
        self._cam = cam
        self._appsrc = appsrc
        self._max_repeats = int(max_repeats)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"rtsp-feed-{self.mount}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._appsrc.emit("end-of-stream")
        except Exception:  # noqa: BLE001
            pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            from gi.repository import Gst
        except Exception:  # noqa: BLE001
            return
        from revobots_arm.transport.frame_bus import FrameBusStalled, HoldLastReader

        cam = self._cam
        h, w, _ = cam.frame_shape
        period = 1.0 / max(1.0, float(cam.fps))
        bus = HoldLastReader(cam.name, self._max_repeats)
        started = time.monotonic()
        waiting_logged = pushed_logged = push_error_logged = resize_logged = False
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    _, frame = bus.read()
                except FrameBusStalled as exc:
                    logger.error("rtsp /%s: %s Stream stopped.", self.mount, exc)
                    return
                if frame is None:
                    if not waiting_logged and t0 - started > 5.0:
                        logger.warning(
                            "rtsp /%s: no frame on frame bus %s yet (camera %s)",
                            self.mount,
                            bus.region,
                            cam.name,
                        )
                        waiting_logged = True
                else:
                    if frame.shape[:2] != (h, w):
                        if not resize_logged:
                            logger.warning(
                                "rtsp /%s: frame bus %s is %sx%s, stream is %sx%s; resizing",
                                self.mount,
                                bus.region,
                                frame.shape[1],
                                frame.shape[0],
                                w,
                                h,
                            )
                            resize_logged = True
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    try:
                        ret = self._appsrc.emit(
                            "push-buffer", Gst.Buffer.new_wrapped(frame.tobytes())
                        )
                        if ret == Gst.FlowReturn.OK:
                            if not pushed_logged:
                                logger.info("rtsp /%s: first frame pushed", self.mount)
                                pushed_logged = True
                        elif not push_error_logged and not self._stop.is_set():
                            # Encoder or client in trouble: the frame is dropped; say so once.
                            logger.warning(
                                "rtsp /%s: push-buffer returned %s",
                                self.mount,
                                ret.value_nick,
                            )
                            push_error_logged = True
                    except Exception as exc:  # noqa: BLE001
                        if not push_error_logged:
                            logger.warning(
                                "rtsp /%s: push-buffer failed: %s", self.mount, exc
                            )
                            push_error_logged = True
                self._stop.wait(timeout=max(0.0, period - (time.monotonic() - t0)))
        finally:
            bus.close()


def _count_close_wait(port: int) -> int:
    """Sockets in CLOSE_WAIT on ``port`` (Linux /proc; 0 elsewhere). Same as the AGV."""
    port_hex = f"{port:04X}"
    count = 0
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_file) as f:
                next(f)
                for line in f:
                    fields = line.split()
                    local_port = fields[1].split(":")[1]
                    state = fields[3]
                    if local_port == port_hex and state == "08":
                        count += 1
        except Exception:  # noqa: BLE001
            continue
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  CamerasManager
# ══════════════════════════════════════════════════════════════════════════════
