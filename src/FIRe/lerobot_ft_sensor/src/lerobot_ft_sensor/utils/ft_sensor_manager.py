import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from typing import Dict
import numpy as np
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
from lerobot_ft_sensor.ft_sensor import FTSensor


class FTSensorManager:
    def __init__(self, node: Node, config: FTSensorConfig) -> None:
        self.node = node
        self.config = config
        self._ft_sensor = FTSensor(self.config)

        # data
        self._raw_data = {"force": np.zeros(3), "torque": np.zeros(3)}
        self._data = {"force": np.zeros(3), "torque": np.zeros(3)}
        self._data_lock = threading.Lock()

        self._scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self._is_connected = False
        self.is_initialized = False

        # publisher
        self._pub_raw = self.node.create_publisher(WrenchStamped, "/ft_sensor/wrench/raw", 10)
        self._pub = self.node.create_publisher(WrenchStamped, "/ft_sensor/wrench", 10)

        # timer (Update -> Process -> Publish를 한 번의 콜백에서 순차적으로 처리)
        self._update_timer = self.node.create_timer(
            1.0 / self.config.update_hz, self._update_timer_callback
        )

    def connect(self) -> None:
        if not self._is_connected:
            self._ft_sensor.connect()
            self._is_connected = True

    def disconnect(self) -> None:
        if self._is_connected:
            self._is_connected = False
            self._ft_sensor.disconnect()
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def _update_timer_callback(self) -> None:
        if not self._is_connected:
            return

        try:
            # 1. Read
            self._raw_data = self._ft_sensor.async_read()

            if not self.is_initialized:
                if self._raw_data is None:
                    return
                self.is_initialized = True

            # 2. Process
            self.process_data()

            # 3. Publish
            self._publish_data()
            
        except Exception as e:
            # 종료 과정에서 발생하는 센서 읽기 에러 무시
            pass

    def _publish_data(self) -> None:
        now_msg = self.node.get_clock().now().to_msg()

        # publish raw data
        msg_raw = WrenchStamped()
        msg_raw.header.stamp = now_msg
        msg_raw.header.frame_id = "ft_sensor" # 보통 base_link보다는 센서 링크를 씁니다.

        msg_raw.wrench.force.x = float(self._raw_data["force"][0])
        msg_raw.wrench.force.y = float(self._raw_data["force"][1])
        msg_raw.wrench.force.z = float(self._raw_data["force"][2])
        msg_raw.wrench.torque.x = float(self._raw_data["torque"][0])
        msg_raw.wrench.torque.y = float(self._raw_data["torque"][1])
        msg_raw.wrench.torque.z = float(self._raw_data["torque"][2])

        self._pub_raw.publish(msg_raw)

        # publish processed data
        msg_proc = WrenchStamped()
        msg_proc.header.stamp = now_msg
        msg_proc.header.frame_id = "ft_sensor"

        msg_proc.wrench.force.x = float(self._data["force"][0])
        msg_proc.wrench.force.y = float(self._data["force"][1])
        msg_proc.wrench.force.z = float(self._data["force"][2])
        msg_proc.wrench.torque.x = float(self._data["torque"][0])
        msg_proc.wrench.torque.y = float(self._data["torque"][1])
        msg_proc.wrench.torque.z = float(self._data["torque"][2])

        self._pub.publish(msg_proc)

    @property
    def raw_data(self) -> Dict[str, np.ndarray]:
        return self._raw_data

    @property    
    def raw_force(self) -> np.ndarray:
        return self._raw_data["force"]
    
    @property
    def raw_torque(self) -> np.ndarray:
        return self._raw_data["torque"]
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        with self._data_lock:
            return {"force": self._data["force"].copy(), "torque": self._data["torque"].copy()}

    @property
    def force(self) -> np.ndarray:
        with self._data_lock:
            return self._data["force"].copy()

    @property
    def torque(self) -> np.ndarray:
        with self._data_lock:
            return self._data["torque"].copy()

    def process_data(self) -> None:
        if self.is_initialized:
            scaled_force = self._raw_data["force"] * self._scale[0:3]
            scaled_torque = self._raw_data["torque"] * self._scale[3:6]
            with self._data_lock:
                self._data = {
                    "force": scaled_force,
                    "torque": scaled_torque
                }