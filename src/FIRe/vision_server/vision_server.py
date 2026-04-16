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

# ---------------------------------------------------------------------------
# SAM3 임포트
# ---------------------------------------------------------------------------
from PIL import Image
import torch
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.model.utils.misc import copy_data_to_device
from sam3.eval.postprocessors import PostProcessImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisionServer")

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
# 2. SAM3 헬퍼 함수들
# ===========================================================================

# 물체별 마스크 색상 팔레트 (RGB)
_MASK_COLORS = [
    (  0, 255,   0),   # 초록  – 첫 번째 물체
    (255, 100,   0),   # 주황  – 두 번째 물체
    (  0, 150, 255),   # 파랑  – 세 번째 물체
    (255,   0, 200),   # 분홍  – 네 번째 물체
    (255, 255,   0),   # 노랑  – 다섯 번째 물체
]


def _apply_color_mask(rgb, mask_np, color=(0, 255, 0), alpha=0.2):
    overlay = rgb.copy()
    overlay[mask_np] = (
        (1 - alpha) * rgb[mask_np] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay

def _create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])

def _set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def _add_text_prompt(datapoint, text_query, counter):
    assert len(datapoint.images) == 1, "set_image를 먼저 호출하세요"
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query, image_id=0, object_ids_output=[],
            is_exhaustive=True, query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=counter, original_image_id=counter,
                original_category_id=1, original_size=[w, h], object_id=0, frame_index=0,
            ),
        )
    )
    return counter, counter + 1


# ===========================================================================
# 3. Vision Server 클래스
# ===========================================================================
class VisionServer:
    def __init__(self, camera_configs: dict, port: int = 5555,
                 use_sam3: bool = False, text_prompts: dict = None, device: str = "cuda"):
        self.port = port
        self.use_sam3 = use_sam3
        self.device = device
        # str은 list[str]로 정규화 (단일 물체도 리스트로 통일)
        raw_prompts = text_prompts or {}
        self.text_prompts = {
            k: ([v] if isinstance(v, str) else list(v))
            for k, v in raw_prompts.items()
        }
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

        # 캡처-추론 파이프라인 큐 (maxsize=1: 항상 최신 프레임만 유지)
        self._raw_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._capture_thread = None
        self._infer_thread = None

        # SAM3 초기화
        self._id_counter = 1
        if self.use_sam3:
            logger.info("Initializing SAM3 model...")
            self._model = build_sam3_image_model(
                bpe_path=None, device=self.device, eval_mode=True, load_from_HF=True
            )
            self._transform = ComposeAPI(transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self._postprocessor = PostProcessImage(
                max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
                use_original_sizes_mask=True, convert_mask_to_rle=False,
                detection_threshold=0.5, to_cpu=True
            )

    @staticmethod
    def _encode_image(image: np.ndarray, quality: int = 85) -> str:
        """RGB 이미지를 Base64 JPEG 문자열로 인코딩 (색상 변환 없음, LeRobot 호환)"""
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")

    def _warmup_sam3(self):
        """더미 이미지로 SAM3 첫 추론 실행."""
        logger.info("Warming up SAM3... This may take a minute.")
        dummy = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        dps = []
        for name in self.cameras.keys():
            prompts = self.text_prompts.get(name, ["object"])
            dp = _create_empty_datapoint()
            _set_image(dp, dummy)
            for prompt in prompts:
                _, self._id_counter = _add_text_prompt(dp, prompt, self._id_counter)
            dps.append(self._transform(dp))

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            batch = collate(dps, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(self.device))
            self._model(batch)
        logger.info("SAM3 warmup complete.")

    def _apply_sam3(self, raw_frames: dict) -> dict:
        """수집된 RGB 프레임 딕셔너리에 SAM3 마스킹을 일괄 적용"""
        dps = []
        ids = {}  # {cam_name: [id_obj0, id_obj1, ...]}

        # 1. Datapoint 구성 (카메라당 여러 프롬프트 지원)
        for name, rgb in raw_frames.items():
            pil_img = Image.fromarray(rgb)
            dp = _create_empty_datapoint()
            _set_image(dp, pil_img)
            prompts = self.text_prompts.get(name, ["object"])
            cam_ids = []
            for prompt in prompts:
                assigned_id, self._id_counter = _add_text_prompt(dp, prompt, self._id_counter)
                cam_ids.append(assigned_id)
            ids[name] = cam_ids
            dps.append(self._transform(dp))

        # 2. 배치 추론
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            batch = collate(dps, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(self.device))
            output = self._model(batch)
            results = self._postprocessor.process_results(output, batch.find_metadatas)

        # 3. 결과 마스킹 (프롬프트마다 다른 색상, 모든 인스턴스 적용)
        masked_frames = {}
        for name, rgb in raw_frames.items():
            overlay = rgb
            for obj_idx, obj_id in enumerate(ids[name]):
                res = results.get(obj_id)
                if res is None or len(res["masks"]) == 0:
                    continue
                color = _MASK_COLORS[obj_idx % len(_MASK_COLORS)]
                for mask_tensor in res["masks"]:
                    mask_np = mask_tensor.squeeze().numpy().astype(bool)
                    overlay = _apply_color_mask(overlay, mask_np, color=color)
            masked_frames[name] = overlay

        return masked_frames

    def start(self):
        logger.info("Connecting to all cameras...")
        for name, cam in self.cameras.items():
            try:
                cam.connect()
            except Exception as e:
                logger.error(f"Failed to connect to camera {name}: {e}")

        if self.use_sam3:
            self._warmup_sam3()

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._capture_thread.start()
        self._infer_thread.start()

        logger.info(f"VisionServer started on port {self.port}. Publishing data...")
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

    def _infer_loop(self):
        """큐에서 프레임을 꺼내 SAM3 추론 후 ZMQ로 전송하는 스레드."""
        while self._running:
            try:
                raw_frames = self._raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()

                if self.use_sam3:
                    processed_frames = self._apply_sam3(raw_frames)
                else:
                    processed_frames = raw_frames

                message = {"timestamps": {}, "images": {}}
                for name, frame in processed_frames.items():
                    message["timestamps"][name] = time.time()
                    message["images"][name] = self._encode_image(frame)

                if message["images"]:
                    with contextlib.suppress(zmq.Again):
                        self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                # SAM3 미사용 시에만 속도 제한 (SAM3는 추론 시간이 자연 제한)
                if not self.use_sam3:
                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0, (1.0 / 30.0) - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Inference/publish error: {e}")

    def stop(self):
        logger.info("Shutting down VisionServer...")
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=5.0)
        for cam in self.cameras.values():
            cam.disconnect()
        self.socket.close()
        self.context.term()
        logger.info("Disconnected safely.")