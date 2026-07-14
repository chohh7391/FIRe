# Inverse3 Teleoperation Manual

A minimal operation manual for collecting demonstrations with the Haption Inverse3 + VerseGrip in FIRe.

## 1. Initial Setup

```bash
cd /home/home/FIRe/src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/src/lerobot_teleoperator_inverse3/inverse3_bridge
make

cd /home/home/FIRe
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

Install the udev rule:

```bash
sudo cp src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/udev/99-haply-inverse3.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify the devices:

```bash
ls -l /dev/inverse3_left /dev/versegrip_left
ls -l /dev/ttyACM*
```

Expected output:

```text
/dev/inverse3_left -> ttyACM0
/dev/versegrip_left -> ttyACM1
```

## 2. Running

First, launch the Gazebo/controller bringup.

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

In another terminal, run teleop recording.

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/record.py --teleop inverse3
```

To save as a GR00T dataset, add the required dataset options.

```bash
python scripts/record.py --teleop inverse3 \
  --vla gr00t \
  --lerobot_repo_id chohh7391/gr00t_peg_insert \
  --lerobot_root /home/home/datasets \
  --lerobot_task "peg insert"
```

## 3. Buttons

| VerseGrip Button | Function |
|---|---|
| Button 0 | Hold to close the gripper; release to open the gripper |
| Button 1 | End the current episode and enter the save procedure |
| Button 2 | Calibrate the current Inverse3 pose and the current robot EEF pose to home |

Recommended operation sequence:

1. Place the Inverse3 in a comfortable neutral position.
2. **While holding the stylus in the reference direction (the direction you want to map to the robot's front)**, press Button 2 to calibrate.
3. Move the Inverse3 to operate the robot.
4. Press Button 0 only when grasping an object.
5. When the episode ends, press Button 1.

> **Rotation (roll/pitch) alignment:** The VerseGrip's absolute orientation has a
> heading (yaw about the gravity axis) that varies by computer and location depending on the
> magnetometer / power-on direction. Without correcting for this, the yaw is correct but
> roll/pitch get mixed in. `align_heading_on_calibration` (default True) takes the grip
> heading at the moment of calibration as the reference and removes this offset, so
> **the calibration pose becomes the forward reference**. That is why, in step 2, holding the
> stylus in a consistent reference direction when you press is important.

## 4. Current Defaults

| Item | Value |
|---|---|
| Inverse3 port | `/dev/inverse3_left` |
| VerseGrip port | `/dev/versegrip_left` |
| `position_scale` | `3.0` |
| `position_axes` | `("-y", "+x", "+z")` |
| `rotation_axes` | `("-y", "+x", "+z")` |
| `align_heading_on_calibration` | `True` (removes heading offset, aligns roll/pitch) |
| `absolute_teleop` | Enabled by default |
| `enable_button` | `-1`, always enabled |
| `require_calibration` | Enabled by default |
| gravity compensation | Disabled |
| haptic feedback | Disabled |

## 5. Quick Check

Check only the buttons and Inverse3 action:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_hardware_live.py --teleop
```

Check the button bits:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_hardware_live.py --map-buttons
```

mock test:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_teleop_mock.py
```

## 6. Troubleshooting

If it reports that the device is missing:

```bash
ls -l /dev/inverse3_left /dev/versegrip_left
ls -l /dev/ttyACM*
```

If `/dev/ttyACM*` is also missing, the USB connection has dropped. Connect directly to a powered hub or a USB port on the PC itself.

If you see an `over-current condition` in dmesg, it is a USB power problem.

```bash
sudo dmesg -w
```

If only the symlink is missing but `/dev/ttyACM*` exists, reapply the udev rule.

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add
```
