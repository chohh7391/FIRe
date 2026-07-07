import os
import time
import json
import queue
import threading
import zmq
import base64
import cv2
import numpy as np
import logging
import contextlib
import argparse

# ---------------------------------------------------------------------------
# LeRobot 카메라 임포트
# ---------------------------------------------------------------------------
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode

# lerobot 카메라 임포트가 root 로거를 WARNING으로 먼저 설정하므로,
# force=True 로 INFO 레벨/핸들러를 강제 재설정한다 (없으면 INFO 로그가 억제됨).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("VisionServer")
logger.setLevel(logging.INFO)

# ===========================================================================
# 1. 카메라 팩토리 (Camera Factory)
# ===========================================================================
def get_webcam_path_by_id(serial_id: str):
    base_dir = "/dev/v4l/by-id/"
    if os.path.exists(base_dir):
        for f in os.listdir(base_dir):
            if serial_id in f and "video-index0" in f:
                return os.path.join(base_dir, f)
    return 0

def create_camera(serial_id: str, width: int = 640, height: int = 480, fps: int = 30):
    if serial_id.isdigit() and len(serial_id) == 12:
        logger.info(f"Creating RealSenseCamera for serial: {serial_id}")
        config = RealSenseCameraConfig(
            serial_number_or_name=serial_id, fps=fps, width=width, height=height, color_mode=ColorMode.RGB
        )
        return RealSenseCamera(config)
    else:
        dev_path = get_webcam_path_by_id(serial_id)
        logger.info(f"Creating OpenCVCamera for ID: {serial_id} (Path: {dev_path})")
        config = OpenCVCameraConfig(
            index_or_path=dev_path, fps=fps, width=width, height=height, color_mode=ColorMode.RGB
        )
        return OpenCVCamera(config)


# ===========================================================================
# 2. Vision Server 클래스
# ===========================================================================
class VisionServer:
    def __init__(
        self,
        camera_configs: dict,
        port: int = 5555,
        width_size: int = 256,
        height_size: int = 256,
    ) -> None:
        self.port = port
        self.width_size = width_size
        self.height_size = height_size
        self.cameras = {}

        # 팩토리를 이용한 카메라 초기화
        for name, serial in camera_configs.items():
            self.cameras[name] = create_camera(serial)

        # ZMQ PUB 소켓 설정
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        # 캡처-발행 파이프라인 큐 (maxsize=1: 항상 최신 프레임만 유지)
        self._raw_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._capture_thread = None
        self._publish_thread = None

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Publish images as contiguous uint8 RGB arrays with shape (height, width, 3)."""
        prepared = np.asarray(image)
        if prepared.ndim == 2:
            prepared = np.repeat(prepared[:, :, None], 3, axis=2)
        if prepared.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {prepared.shape}.")
        if prepared.shape[2] == 4:
            prepared = prepared[:, :, :3]
        if prepared.shape[2] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {prepared.shape}.")
        if prepared.shape[:2] != (self.height_size, self.width_size):
            prepared = cv2.resize(
                prepared,
                (self.width_size, self.height_size),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(prepared, dtype=np.uint8)

    @staticmethod
    def _encode_image(image: np.ndarray, quality: int = 85) -> str:
        """RGB 이미지를 Base64 JPEG 문자열로 인코딩 (색상 변환 없음, LeRobot 호환)"""
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")

    def _wait_for_frames(self, name, cam, timeout: float = 5.0) -> bool:
        """카메라 연결 후 실제 프레임(데이터)이 들어오는지 검증한다."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                frame = cam.read_latest(max_age_ms=1000)
                if frame is not None:
                    h, w = np.asarray(frame).shape[:2]
                    logger.info(
                        f"[DATA OK] Camera '{name}' streaming — frame received ({w}x{h})."
                    )
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        logger.error(
            f"[NO DATA] Camera '{name}' connected but no frame received within {timeout:.0f}s."
        )
        return False

    def start(self):
        logger.info("Connecting to all cameras...")
        connected = []
        failed = []
        for name, cam in self.cameras.items():
            try:
                cam.connect()
                logger.info(f"[OK] Camera '{name}' connected. Verifying data stream...")
                if self._wait_for_frames(name, cam):
                    connected.append(name)
                else:
                    failed.append(name)
                    with contextlib.suppress(Exception):
                        cam.disconnect()
            except Exception as e:
                failed.append(name)
                logger.error(f"[FAIL] Camera '{name}' failed to connect: {e}")

        # 스트리밍이 확인된 카메라만 유지 (실패 카메라는 캡처 루프에서 제외)
        for name in failed:
            self.cameras.pop(name, None)

        if not connected:
            logger.error("No cameras streaming data. Aborting start.")
            self.stop()
            return

        logger.info(
            f"{len(connected)}/{len(self.cameras)} camera(s) connected & streaming: {connected}"
        )

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._capture_thread.start()
        self._publish_thread.start()

        logger.info("=" * 60)
        logger.info(f"VisionServer READY — publishing on tcp://*:{self.port}")
        logger.info(f"  cameras : {list(self.cameras.keys())}")
        logger.info(f"  size    : {self.width_size}x{self.height_size}")
        logger.info("  press Ctrl+C to stop")
        logger.info("=" * 60)
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def _capture_loop(self):
        """카메라 프레임을 지속적으로 캡처해 큐에 넣는 스레드."""
        while self._running:
            try:
                raw_frames = {}
                for name, cam in self.cameras.items():
                    try:
                        raw_frames[name] = cam.read_latest(max_age_ms=1000)
                    except Exception as e:
                        logger.warning(f"Could not get frame from {name}: {e}")

                if not raw_frames:
                    time.sleep(0.01)
                    continue

                # 큐가 꽉 찼으면 낡은 프레임 버리고 최신 프레임으로 교체
                if self._raw_queue.full():
                    try:
                        self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._raw_queue.put_nowait(raw_frames)
            except Exception as e:
                logger.error(f"Capture error: {e}")

    def _publish_loop(self):
        """큐에서 프레임을 꺼내 ZMQ로 전송하는 스레드."""
        while self._running:
            try:
                raw_frames = self._raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()

                message = {"timestamps": {}, "images": {}}
                for name, frame in raw_frames.items():
                    frame = self._prepare_image(frame)
                    message["timestamps"][name] = time.time()
                    message["images"][name] = self._encode_image(frame)

                if message["images"]:
                    with contextlib.suppress(zmq.Again):
                        self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                # 발행 속도 제한 (최대 30Hz)
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, (1.0 / 30.0) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Publish error: {e}")

    def stop(self):
        logger.info("Shutting down VisionServer...")
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=5.0)
        for cam in self.cameras.values():
            cam.disconnect()
        self.socket.close()
        self.context.term()
        logger.info("Disconnected safely.")
