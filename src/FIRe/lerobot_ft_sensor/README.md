# LeRobot FT Sensor

`lerobot_ft_sensor` is a standalone Python module for reading a Bota force/torque sensor through the Bota serial driver. It provides a small sensor wrapper and a ROS 2 manager that can publish wrench data.

This module is intended to be usable without depending on a specific robot project.

## Install

From this module directory:

```bash
cd path/to/lerobot_ft_sensor
pip install -e .
```

ROS 2 Python packages such as `rclpy` are expected to come from the system ROS installation, not from `pip`.

If using ROS 2 features, source the ROS workspace before running scripts that import `rclpy`:

```bash
source ~/ros2_ws/install/setup.bash
```

## Configuration

The default Bota driver config is:

```text
src/lerobot_ft_sensor/config/bota_binary.json
```

The default serial device path is:

```text
/dev/serial/by-id/usb-Bota_Systems_Bota_Systems_Force_Torque_Sensor_205330665131-if00
```

If your sensor appears under a different path, update this field in `bota_binary.json`:

```json
"communication_interface_params": {
  "com_port": "/dev/serial/by-id/YOUR_DEVICE_ID",
  "baudrate": 2000000
}
```

## Check the USB Device

Connect the sensor and check that Linux can see it:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/ttyACM*
```

Expected ownership is usually `root dialout` with read/write permissions for the group:

```text
crw-rw---- 1 root dialout ... /dev/ttyACM0
```

## Serial Permission Setup

If the Bota driver prints this error:

```text
Permission denied, add the current user to the dialout group
```

add the current user to `dialout`:

```bash
sudo usermod -aG dialout $USER
```

Then fully refresh the login session. Closing only the current terminal tab can be insufficient, especially when the terminal is owned by an older VS Code or desktop session.

Recommended options:

```bash
# Option 1: fully log out of Ubuntu and log back in

# Option 2: reboot
sudo reboot
```

For an immediate shell-only refresh, this can work:

```bash
newgrp dialout
groups
```

`groups` must include `dialout` before connecting to the FT sensor.

Also verify the group database contains the user:

```bash
getent group dialout
```

Expected output should include the username, for example:

```text
dialout:x:20:home
```

## Temporary Permission Test

For a short one-off test only:

```bash
sudo chmod a+rw /dev/ttyACM0
```

or, if using the by-id path:

```bash
sudo chmod a+rw /dev/serial/by-id/usb-Bota_Systems_Bota_Systems_Force_Torque_Sensor_205330665131-if00
```

This is not persistent. It can be reset by USB reconnect, reboot, or udev. The normal fix is `dialout` group membership.

## ROS 2 Sensor Test

The included test script creates an `FTSensorManager`, connects to the sensor, prints force/torque values, and publishes:

```text
/ft_sensor/wrench/raw
/ft_sensor/wrench
```

Run it from the module directory:

```bash
source ~/ros2_ws/install/setup.bash
python test/test_lerobot_ft_sensor.py
```

In another sourced terminal, inspect the published wrench data:

```bash
ros2 topic echo /ft_sensor/wrench
```

## Troubleshooting

If `sudo usermod -aG dialout $USER` was already run but `groups` still does not show `dialout`, the current shell is still using an old login session. Run `newgrp dialout`, or fully close VS Code and log out of the Ubuntu desktop session before trying again.

If the device exists but has a different path, update `src/lerobot_ft_sensor/config/bota_binary.json`.

If the device does not exist under `/dev/serial/by-id/` or `/dev/ttyACM*`, reconnect the USB cable and check `dmesg`:

```bash
dmesg | tail -n 50
```
