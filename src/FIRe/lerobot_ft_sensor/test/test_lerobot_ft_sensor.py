import rclpy
from rclpy.node import Node
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
from lerobot_ft_sensor.utils.ft_sensor_manager import FTSensorManager


def main():
    rclpy.init()

    node = Node("test_ft_sensor_node")
    
    config = FTSensorConfig()

    node.get_logger().info("Initializing FT Sensor Manager...")
    ft_sensor_manager = FTSensorManager(node=node, config=config)  # publish /ft_sensor/wrench
    ft_sensor_manager.connect()

    def print_data_callback():
        if ft_sensor_manager.is_initialized:
            force = ft_sensor_manager.force
            torque = ft_sensor_manager.torque
            
            node.get_logger().info(
                f"Force: [Fx: {force[0]:>7.3f}, Fy: {force[1]:>7.3f}, Fz: {force[2]:>7.3f}] N | "
                f"Torque: [Tx: {torque[0]:>7.3f}, Ty: {torque[1]:>7.3f}, Tz: {torque[2]:>7.3f}] Nm"
            )
        else:
            node.get_logger().warn("Waiting for sensor initialization...", throttle_duration_sec=1.0)

    print_timer = node.create_timer(0.5, print_data_callback)

    try:
        node.get_logger().info("Spinning node. Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        ft_sensor_manager.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()