import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from typing import Dict, Tuple, List
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras import CameraConfig
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
from lerobot_ft_sensor.ft_sensor import FTSensor


class CameraSensorManager:
    def __init__(self, node: Node, config: Dict[str, CameraConfig], fps: float = 30.0) -> None:
        self.node = node
        self.config = config
        self._cameras = make_cameras_from_configs(self.config)
        self.bridge = CvBridge()

        # data
        self._raw_data: Dict[str, np.ndarray] = {cam_name: None for cam_name in self._cameras.keys()}
        
        self._is_connected = False
        self.is_initialized = False

        self._pubs = {}
        for cam_name in self.names:
            topic_name = f"/camera/{cam_name}/image_raw"
            self._pubs[cam_name] = self.node.create_publisher(Image, topic_name, 10)

        self._update_timer = self.node.create_timer(
            1.0 / fps, self._update_timer_callback
        )
    
    def connect(self) -> None:
        if not self._is_connected:
            self._is_connected = True
            for cam in self._cameras.values():
                cam.connect()
    
    def disconnect(self) -> None:
        if self._is_connected:
            self._is_connected = False
            for cam in self._cameras.values():
                cam.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def names(self) -> List[str]:
        return list(self._cameras.keys())
    
    @property
    def shapes(self) -> Dict[str, Tuple[int, int, int]]:
        return {
            cam_name: (self._cameras[cam_name].height, self._cameras[cam_name].width, 3) for cam_name in self.names
        }
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        if self.is_initialized:
            return self._raw_data
        else:
            return {
                cam_name: np.zeros(self.shapes[cam_name], dtype=np.uint8) for cam_name in self.names
            }
    
    def _update_timer_callback(self) -> None:
        if not self._is_connected:
            return

        try:
            for cam_name, cam in self._cameras.items():
                frame = cam.async_read(timeout_ms=1000)
                if frame is not None:
                    self._raw_data[cam_name] = frame
            
            if not self.is_initialized:
                if all(frame is not None for frame in self._raw_data.values()):
                    self.is_initialized = True
                    self.node.get_logger().info("All cameras initialized successfully. Starting to publish.")

            if self.is_initialized:
                self._publish_data()
                
        except Exception as e:
            # 종료(Disconnect) 과정에서 카메라가 먼저 꺼져서 발생하는 읽기 에러를 무시하여 세그폴트 방지
            pass

    def _publish_data(self) -> None:
        now_msg = self.node.get_clock().now().to_msg()

        for cam_name, frame in self._raw_data.items():
            # 만약 이번 루프에서 프레임이 유실되어 이전 프레임(또는 None)이 있더라도 
            # is_initialized가 True인 상태에서는 처리 (None인 경우는 스킵)
            if frame is None:
                continue 

            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
                img_msg.header.stamp = now_msg
                img_msg.header.frame_id = f"{cam_name}_camera_link"

                self._pubs[cam_name].publish(img_msg)
            except Exception as e:
                self.node.get_logger().warn(f"Failed to publish {cam_name} image: {e}")



class FTSensorManager:
    def __init__(self, node: Node, config: FTSensorConfig) -> None:
        self.node = node
        self.config = config
        self._ft_sensor = FTSensor(self.config)

        # data
        self._raw_data = {"force": np.zeros(3), "torque": np.zeros(3)}
        self._data = {"force": np.zeros(3), "torque": np.zeros(3)}

        self._scale = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
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
        return self._data

    @property    
    def force(self) -> np.ndarray:
        return self._data["force"]
    
    @property
    def torque(self) -> np.ndarray:
        return self._data["torque"]
    
    def process_data(self) -> None:
        """딕셔너리 연산 에러 수정 (force와 torque를 분리해서 스케일링)"""
        if self.is_initialized:
            scaled_force = self._raw_data["force"] * self._scale[0:3]
            scaled_torque = self._raw_data["torque"] * self._scale[3:6]
            
            self._data = {
                "force": scaled_force,
                "torque": scaled_torque
            }