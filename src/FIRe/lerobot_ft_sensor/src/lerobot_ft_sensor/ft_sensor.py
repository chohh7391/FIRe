import threading
import time
import logging
import numpy as np
import bota_driver

from .configuration_ft_sensor import FTSensorConfig

logger = logging.getLogger(__name__)

class FTSensor:
    """
    LeRobot 호환 구조로 래핑된 Bota FT Sensor 클래스.
    """
    def __init__(self, config: FTSensorConfig):
        self.config = config
        
        self.driver = bota_driver.BotaDriver(self.config.config_path)
        
        self._is_connected = False
        self.thread = None
        self.stop_event = None
        
        # LeRobot 스타일 동기화 객체
        self.frame_lock = threading.Lock()
        self.new_data_event = threading.Event()
        
        # 최신 데이터
        self.latest_force = np.zeros(3, dtype=np.float32)
        self.latest_torque = np.zeros(3, dtype=np.float32)
        self.latest_timestamp = 0.0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        """센서를 초기화하고 백그라운드 스레드를 시작합니다."""
        if self.is_connected:
            logger.warning("[FTSensor] Already connected.")
            return

        logger.info(f"[FTSensor] Initializing Bota Driver (ver: {self.driver.get_driver_version_string()})")
        
        if not self.driver.configure():
            raise RuntimeError("FTSensor: Failed to configure driver")
        
        if not self.driver.tare():
            logger.warning("[FTSensor] Failed to tare sensor during start")
            
        if not self.driver.activate():
            raise RuntimeError("FTSensor: Failed to activate driver")

        self._start_read_thread()
        self._is_connected = True
        logger.info("[FTSensor] Connected and background thread started.")

        if warmup and self.config.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.config.warmup_s:
                self.async_read(timeout_ms=self.config.warmup_s * 1000)
                time.sleep(0.01)

    def _start_read_thread(self):
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._update_loop, daemon=True, name="FTSensor_Loop")
        self.thread.start()
        time.sleep(0.1)

    def _update_loop(self):
        """센서 주기에 맞춰 블로킹 읽기를 수행하는 백그라운드 루프"""
        while not self.stop_event.is_set():
            try:
                bota_frame = self.driver.read_frame_blocking()
                
                with self.frame_lock:
                    self.latest_force = np.array(bota_frame.force, dtype=np.float32)
                    self.latest_torque = np.array(bota_frame.torque, dtype=np.float32)
                    self.latest_timestamp = time.perf_counter() # 프레임워크와 시간 동기화 용이
                
                self.new_data_event.set()
                
            except Exception as e:
                # logger.warning(f"[FTSensor] Read error: {e}")
                pass

    def async_read(self, timeout_ms: float = None) -> dict:
        """새로운 센서 데이터가 수신될 때까지 대기한 후 반환 (동기화용)"""
        if not self.is_connected:
            raise ConnectionError("FTSensor is not connected.")

        timeout = timeout_ms if timeout_ms is not None else self.config.timeout_ms
        if not self.new_data_event.wait(timeout=timeout / 1000.0):
            raise TimeoutError(f"FTSensor async_read timeout after {timeout}ms")

        with self.frame_lock:
            data = {
                "force": self.latest_force.copy(),
                "torque": self.latest_torque.copy(),
                "timestamp": self.latest_timestamp
            }
            self.new_data_event.clear()

        return data

    def read_latest(self, max_age_ms: int = 50) -> dict:
        """대기 없이 버퍼의 최신 센서 데이터를 즉시 반환 (고속 제어 루프용)"""
        if not self.is_connected:
            raise ConnectionError("FTSensor is not connected.")

        with self.frame_lock:
            force = self.latest_force.copy()
            torque = self.latest_torque.copy()
            timestamp = self.latest_timestamp

        age_ms = (time.perf_counter() - timestamp) * 1000
        if age_ms > max_age_ms:
            logger.warning(f"[FTSensor] Data is stale: {age_ms:.1f}ms old")

        return {
            "force": force,
            "torque": torque,
            "timestamp": timestamp
        }

    def disconnect(self) -> None:
        """센서와 스레드를 안전하게 종료합니다."""
        if not self.is_connected:
            return

        if self.stop_event:
            self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        try:
            self.driver.deactivate()
            self.driver.shutdown()
        except Exception as e:
            logger.error(f"[FTSensor] Error during shutdown: {e}")

        self._is_connected = False
        logger.info("[FTSensor] Disconnected safely.")