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
| `OPEN <inv3_port> <grip_port>` | `OK` | 두 장치 초기화, gravity compensation 활성화 |
| `GET_STATE` | `STATE px py pz vx vy vz qw qx qy qz buttons battery` | 현재 상태 폴링 |
| `SEND_FORCE fx fy fz` | `STATE ...` | 힘 전송 후 상태 반환 |
| `CLOSE` | (exit) | 프로세스 종료 |

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
| `position_scale` | `1.0` | Inv3 displacement → robot displacement 배율 |
| `rotation_scale` | `1.0` | 회전 배율 |
| `enable_button` | `0` | 활성화 버튼 bit 번호 (0 = bit-0) |

---

## 남은 작업

- [ ] **실제 기기로 통합 테스트** — `inverse3_server` 빌드 후 `/dev/inverse3_left`, `/dev/versegrip_left` 연결 확인
- [ ] **position_scale 튜닝** — Inverse3 workspace(~15cm) 대 로봇 workspace 비율 보정
- [ ] **Forge task constraint 선택적 적용** — teleop 시 roll/pitch 고정 여부 결정 (`is_relative=True` vs `False`)
- [ ] **데이터셋 action format 정리** — 현재 teleop 녹화 시 absolute EEF pose(7-dim) 저장; GR00T 학습용 normalized format으로 변환 파이프라인 추가 검토
- [ ] **Haptic feedback 연결** — `send_feedback({"force": ft_sensor.force})`로 FT 센서 데이터를 햅틱으로 전달
