# Haption Inverse3 Teleoperator

## 개요

`lerobot_teleoperator_inverse3` 패키지는 Haption Inverse3 햅틱 기기와 VerseGrip 핸들을 LeRobot v0.4.4 `Teleoperator` 인터페이스로 감싼 모듈이다.
FIRe 프로젝트에 종속되지 않고 독립 패키지로 재사용할 수 있도록 설계됐다.

---

## 아키텍처

```
[Inverse3 Device] ──USB──► [inverse3_server (C++ exe)]
                                      │  stdin/stdout text protocol
                         [Inverse3Server (Python wrapper)]
                                      │
                         [Inverse3Teleop (LeRobot Teleoperator)]
                                      │
                         [Control loop / record.py]
                                      │
                         [FR3Robot.send_teleop_action()]
```

### 왜 subprocess IPC인가?

Haplay SDK(`libHaply.HardwareAPI.a`)는 `-fPIC` 없이 컴파일된 static library다.
x86_64 Linux에서 Python ctypes/pybind11 공유 라이브러리(`.so`)로 링크하면
`R_X86_64_PC32 relocation` 에러가 발생한다.
따라서 C++ 실행 파일(`inverse3_server`)을 별도로 빌드하고 stdin/stdout 텍스트 프로토콜로 통신한다.

---

## 패키지 구조

```
lerobot_teleoperator_inverse3/
├── pyproject.toml
└── src/lerobot_teleoperator_inverse3/
    ├── __init__.py
    ├── config_inverse3.py        # Inverse3TeleopConfig
    ├── inverse3.py               # Inverse3Teleop (LeRobot Teleoperator)
    └── inverse3_bridge/
        ├── __init__.py           # Inverse3Server (Python subprocess wrapper)
        ├── inverse3_bridge.cpp   # C++ server (SDK 래핑, stdio 프로토콜)
        ├── Makefile
        └── sdk/                  # 번들된 Haplay SDK (include/ + libHaply.HardwareAPI.a)
```

---

## 빌드 & 설치

```bash
# 1. C++ server 빌드 (최초 1회)
cd src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/\
src/lerobot_teleoperator_inverse3/inverse3_bridge
make

# 2. Python 패키지 설치
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

---

## stdio 프로토콜

| 명령 | 응답 | 설명 |
|---|---|---|
| `OPEN <inv3_port> <grip_port>` | `OK` 또는 `OK GRIP_UNAVAILABLE` | 두 장치 초기화. Inverse3 gravity compensation은 비활성화. 핸들이 꺼져 있으면 `GRIP_UNAVAILABLE` |
| `GET_STATE` | `STATE px py pz vx vy vz qw qx qy qz buttons battery` | 현재 상태 폴링 |
| `SEND_FORCE fx fy fz` | `STATE ...` | 힘 전송 후 상태 반환 |
| `CLOSE` | (exit) | 프로세스 종료 |

> `OPEN`은 VerseGrip 핸들이 응답하지 않아도 `OK GRIP_UNAVAILABLE`로 연결을 유지한다
> (Inverse3 translation은 동작). 핸들 전원을 켜면 다음 `GET_STATE`부터 자동으로 복구된다.

## 하드웨어 / 디바이스 설정 (udev)

raw `/dev/ttyACMx` 번호는 **안정적이지 않다** — Inverse3(Teensy)와 VerseGrip USB 트랜시버는
리셋/재연결 시 재열거되며 번호가 서로 바뀐다. 따라서 USB serial로 고정 symlink를 만든다.

```bash
cd src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/udev
sudo cp 99-haply-inverse3.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 확인
ls -l /dev/inverse3_left /dev/versegrip_left
```

현재 유닛 serial: Inverse3 = `16894910` (16c0:0483), VerseGrip 트랜시버 = `548F7FD33638E628` (2fe3:0100).
다른 유닛/오른팔을 추가하려면 rules 파일에 serial과 `inverse3_right`/`versegrip_right`를 추가한다.

## 구현 노트 — 펌웨어 프레임 워크어라운드

번들 SDK(`libHaply.HardwareAPI` v0.2.8)와 현재 Inverse3 펌웨어 사이에 프로토콜 불일치가 있다.
장치가 cursor-state 응답 사이에 **5바이트 heartbeat 프레임**(`BA 02 02 XX XX`)을 비동기로 스트리밍하는데,
SDK의 `EndEffectorForce()` 리더가 이 프레임에서 desync되어 `timeout waiting for header code`로 무한 대기한다
(쓰기는 정상 — torque는 들어감).

해결: bridge에서 force는 SDK(`SendEndEffectorForce`)로 보내되, 상태 응답은 **직접 파싱**한다.
- state 프레임 = `0x2B` + position[3] + velocity[3] (LE float32, 25바이트)
- heartbeat 프레임 = `0xBA 0x02 0x02 <2바이트>` → skip

또한 `DeviceWakeup`의 첫 시도가 자주 타임아웃되므로 wakeup을 재시도한다 (`wakeup_inverse3`).
이 워크어라운드로 양 장치 모두 ~1kHz로 유효 데이터를 폴링한다.

---

## 동작 방식

### Calibration

`calibrate()`는 현재 Inverse3 위치/자세를 **home**으로 저장한다.
Robot 상태는 필요 없다.

### Enable button (rising edge re-anchor)

`get_action()` 내부에서 enable button(기본: bit-0)을 감시한다.

- 버튼을 **처음 누른 순간(rising edge)**: Inverse3 home 위치를 현재 위치로 재설정한다.
- 이후 버튼을 누른 상태에서 이동하면 displacement가 0에서 시작 → 로봇이 점프하지 않는다.
- 버튼을 놓으면 `inv3.enabled = False`이고, 다시 누르면 새로운 anchor에서 시작한다.

### `get_action()` 출력

```python
{
    "inv3.pos"    : np.float32 (3,),   # home 기준 displacement (metres)
    "inv3.rot"    : np.float32 (4,),   # home 기준 delta rotation (WXYZ quat)
    "inv3.buttons": np.int32   (1,),   # VerseGrip button bitmask
    "inv3.enabled": bool       (1,),   # enable button 상태
    "inv3.gripper": np.float32 (1,),   # button-0 hold: close, released: open
    "inv3.end_episode": bool   (1,),   # button-1 episode 종료 요청
}
```

### Control loop에서 robot target 계산

```python
prev_enabled = False
robot_home_pos  = robot.ee_pos.copy()
robot_home_quat = robot.ee_quat.copy()

while True:
    action = teleop.get_action()
    enabled = action["inv3.enabled"].item()

    # Rising edge: robot home도 재캡처
    if enabled and not prev_enabled:
        robot_home_pos  = robot.ee_pos.copy()
        robot_home_quat = robot.ee_quat.copy()

    prev_enabled = enabled

    if enabled:
        target_pos  = robot_home_pos + action["inv3.pos"]
        target_quat = (delta_rot * home_rot).as_wxyz()   # scipy Rotation
        robot.send_teleop_action(TeleopAction(
            {"arm_actions": concat(target_pos, target_quat)},
            action_space="task_space",
            is_relative=False,   # Forge process_action() bypass
        ))
```

---

## Config

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `inverse3_port` | `/dev/inverse3_left` | Inverse3 serial port |
| `versegrip_port` | `/dev/versegrip_left` | VerseGrip serial port |
| `position_scale` | `3.0` | Inv3 displacement → robot displacement 배율 |
| `position_axes` | `("-y", "+x", "+z")` | robot XYZ에 대응되는 signed Inverse3 translation axes |
| `rotation_axes` | `("-y", "+x", "+z")` | robot XYZ에 대응되는 signed VerseGrip rotation frame axes (right-handed, det +1 필요) |
| `reanchor_on_enable` | `True` | enable button rising edge마다 현재 Inverse3/robot pose를 home으로 재캡처 |
| `require_calibration` | `False` | calibration button을 누르기 전에는 motion disabled |
| `enable_button` | `0` | 활성화 버튼 bit 번호. `-1`이면 enable button 없이 항상 enabled |
| `grasp_button` | `0` | 누르고 있으면 gripper close, 떼면 gripper open |
| `end_episode_button` | `1` | episode 종료 버튼 |
| `gripper_open_value` | `1.0` | open 명령 값 |
| `gripper_close_value` | `-1.0` | close 명령 값 |
| `haptic_feedback_enabled` | `False` | `send_feedback()` 외력 적용 허용. 기본은 zero force only |
| `calibration_button` | `2` | 현재 Inverse3 pose와 현재 robot EEF pose를 같은 home 기준으로 초기화 |

### Absolute teleop 초기화 흐름

기본 동작은 enable button을 누를 때마다 re-anchor해서 로봇 점프를 막는다. 같은 Inverse3 위치가 항상 같은 robot target으로 가는 absolute 방식이 필요하고 첫 번째 버튼을 grasp에 쓰려면 `scripts/record.py`에 `--absolute_teleop --enable_button -1 --require_calibration`을 준다.

권장 조작 순서:

1. enable button을 누르지 않은 상태에서 Inverse3를 편한 중립 위치로 움직인다.
2. calibration button을 누른다. 이 순간 `inv3_home`과 현재 robot EEF pose가 동시에 home으로 저장된다.
3. 첫 번째 버튼은 누르고 있는 동안 gripper close, 떼면 gripper open이다.
4. 두 번째 버튼을 누르면 현재 episode를 종료하고 저장 절차로 들어간다.
5. Inverse3를 움직이면 calibration 기준 pose로부터 absolute target이 계산된다.
6. 다시 기준을 잡고 싶으면 Inverse3를 새 중립 위치로 옮긴 뒤 calibration button을 다시 누른다.

프레임이 다르면 CLI에서 축을 바꿔 실험한다.

```bash
python scripts/record.py --teleop inverse3 --absolute_teleop \
  --enable_button -1 --require_calibration
```

`rotation_axes`는 right-handed mapping이어야 하므로 determinant가 `+1`인 조합만 허용된다.

---

## 남은 작업

### 필수 (하드웨어 연결 전 확인 필요)

- [x] **실제 기기로 통합 테스트** — 완료 (2026-06-22). bridge 재빌드 후 양 장치 ~1kHz 폴링 확인.
  Inverse3 position/velocity, VerseGrip quaternion(|q|≈1)/buttons/battery 모두 유효.
  펌웨어 heartbeat 프레임 desync 이슈를 직접 파싱으로 해결(위 "구현 노트" 참조).
  라이브 테스트: `test/test_hardware_live.py`. 고수준 `Inverse3Teleop` connect/calibrate/get_action도 검증.
  - [ ] **enable button + displacement 인터랙티브 검증** — VerseGrip button-0을 누른 채 기기를 움직여
    `inv3.pos`가 변하는지, 놓으면 0으로 돌아오는지 직접 확인 필요 (사용자 조작).
  - [ ] **VerseGrip 배터리** — 라이브 테스트에서 battery 값이 매우 낮게 표시됨(≈4). 핸들 충전 권장.

- [ ] **좌표계 정렬 (translation)** — Inverse3의 XYZ 축 방향이 로봇 world frame과 일치하는지 확인 필요.
  기기를 X 방향으로 움직였을 때 로봇도 X 방향으로 움직여야 한다.
  축이 다르면 `--position_axes` 또는 `Inverse3TeleopConfig.position_axes`로 signed axis mapping을 조정한다.

- [ ] **좌표계 정렬 (rotation)** — VerseGrip quaternion의 reference frame이 로봇 base frame과 다를 수 있음.
  기기를 roll/pitch/yaw로 각각 돌려보며 로봇 회전 방향과 대응 관계 확인.
  필요 시 `--rotation_axes` 또는 `Inverse3TeleopConfig.rotation_axes`로 right-handed frame mapping을 조정한다.

- [ ] **position_scale 튜닝** — Inverse3 workspace(~15cm) 대 로봇 유효 workspace 비율 보정.
  `--position_scale` 인자로 조정하며 실험적으로 결정.

### 기능 추가

- [ ] **초기화 절차 (Initialize)** — 현재는 connect() 시 즉시 현재 위치를 home으로 잡음.
  올바른 흐름:
  1. 로봇이 초기 pose로 이동 (또는 현재 위치 확인)
  2. 사용자가 Inverse3를 편안한 중립 위치에 잡음
  3. **특정 버튼** (예: VerseGrip button-1)을 눌러 "초기화 완료" 신호 전달
  4. 이 순간 `inv3_home` + `robot_home` 동시 캡처
  5. 이후 enable button(button-0)으로 teleop 시작/일시정지
  현재 enable button rising edge가 이 역할을 부분적으로 하지만,
  "초기화 전 로봇이 움직이지 않아야 한다"는 보장이 명시적이지 않음.

- [ ] **Teleop 종료 신호** — 현재는 `Ctrl+C`로만 종료 가능.
  다음 방식 중 결정 필요:
  - VerseGrip 버튼을 **특정 패턴** (예: 2초 이상 long press, 또는 더블 클릭)으로 누르면 종료
  - 별도의 종료 전용 버튼 지정 (button bitmask에 여유 있음)
  종료 시 로봇에게 마지막 명령 후 hold 상태로 전환해야 함 (갑작스러운 정지 방지).

- [x] **Gripper 제어** — VerseGrip 첫 번째 버튼(bit-0)을 hold-to-grasp로 매핑.
  버튼을 누르면 close, 떼면 open이다.

- [ ] **Haptic force feedback 연결** — 기본값은 안전하게 비활성화되어 zero force만 적용된다.
  FT 센서 데이터를 햅틱으로 실시간 전달하려면 `haptic_feedback_enabled=True` 또는 `--haptic_feedback`을 명시적으로 켠 뒤
  `send_feedback({"force": ft_sensor.force})`를 호출한다. force scale 튜닝 필요 (너무 강하면 기기 손상 위험).

### 데이터 품질

- [ ] **데이터셋 action format 정리** — 현재 teleop 녹화 시 absolute EEF pose(7-dim) 저장.
  GR00T 학습용 normalized format(6-dim, [-1,1])으로 변환하는 파이프라인 추가 검토.

- [ ] **Forge task constraint 선택적 적용** — teleop 시 roll/pitch 고정 여부 결정.
  현재 `is_relative=False`로 constraint bypass. 필요 시 task별 post-processing 추가.
