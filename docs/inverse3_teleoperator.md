# Inverse3 Teleoperation Manual

FIRe에서 Haption Inverse3 + VerseGrip으로 demonstration을 수집하는 최소 운용 매뉴얼이다.

## 1. 최초 설치

```bash
cd /home/home/FIRe/src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/src/lerobot_teleoperator_inverse3/inverse3_bridge
make

cd /home/home/FIRe
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

udev rule 설치:

```bash
sudo cp src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/udev/99-haply-inverse3.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

장치 확인:

```bash
ls -l /dev/inverse3_left /dev/versegrip_left
ls -l /dev/ttyACM*
```

정상 예:

```text
/dev/inverse3_left -> ttyACM0
/dev/versegrip_left -> ttyACM1
```

## 2. 실행

Gazebo/controller bringup을 먼저 실행한다.

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

다른 터미널에서 teleop recording을 실행한다.

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/record.py --teleop inverse3
```

GR00T dataset으로 저장하려면 필요한 dataset 옵션을 추가한다.

```bash
python scripts/record.py --teleop inverse3 \
  --vla gr00t \
  --lerobot_repo_id chohh7391/gr00t_peg_insert \
  --lerobot_root /home/home/datasets \
  --lerobot_task "peg insert"
```

## 3. 버튼

| VerseGrip 버튼 | 기능 |
|---|---|
| Button 0 | 누르고 있으면 gripper close, 떼면 gripper open |
| Button 1 | 현재 episode 종료 및 저장 절차 진입 |
| Button 2 | 현재 Inverse3 pose와 현재 robot EEF pose를 home으로 calibration |

권장 조작 순서:

1. Inverse3를 편한 중립 위치로 둔다.
2. **stylus를 기준 방향(로봇 정면에 대응시키고 싶은 방향)으로 잡은 채** Button 2를 눌러 calibration한다.
3. Inverse3를 움직여 로봇을 조작한다.
4. 물체를 잡을 때만 Button 0을 누른다.
5. episode가 끝나면 Button 1을 누른다.

> **회전(roll/pitch) 정렬:** VerseGrip의 절대 자세는 magnetometer/전원-on 방향에 따라
> heading(중력축 기준 yaw)이 컴퓨터·장소마다 달라진다. 이를 보정하지 않으면 yaw는 맞는데
> roll/pitch가 섞인다. `align_heading_on_calibration`(기본 True)이 calibration 시점의 grip
> heading을 기준으로 잡아 이 offset을 제거하므로, **calibration 자세가 곧 forward 기준**이
> 된다. 그래서 2번에서 stylus를 일관된 기준 방향으로 잡고 누르는 것이 중요하다.

## 4. 현재 기본값

| 항목 | 값 |
|---|---|
| Inverse3 port | `/dev/inverse3_left` |
| VerseGrip port | `/dev/versegrip_left` |
| `position_scale` | `3.0` |
| `position_axes` | `("-y", "+x", "+z")` |
| `rotation_axes` | `("-y", "+x", "+z")` |
| `align_heading_on_calibration` | `True` (heading offset 제거, roll/pitch 정렬) |
| `absolute_teleop` | 기본 사용 |
| `enable_button` | `-1`, 항상 enabled |
| `require_calibration` | 기본 사용 |
| gravity compensation | 비활성화 |
| haptic feedback | 비활성화 |

## 5. 빠른 점검

버튼과 Inverse3 action만 확인:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_hardware_live.py --teleop
```

버튼 bit 확인:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_hardware_live.py --map-buttons
```

mock test:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_teleop_mock.py
```

## 6. 문제 해결

장치가 없다고 나오면:

```bash
ls -l /dev/inverse3_left /dev/versegrip_left
ls -l /dev/ttyACM*
```

`/dev/ttyACM*`도 없으면 USB가 끊긴 상태다. powered hub 또는 PC 본체 USB 포트에 직접 연결한다.

`over-current condition`이 dmesg에 보이면 USB 전원 문제다.

```bash
sudo dmesg -w
```

symlink만 없고 `/dev/ttyACM*`는 있으면 udev rule을 다시 적용한다.

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add
```
