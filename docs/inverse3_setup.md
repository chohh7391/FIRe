# Inverse3 Setup Guide

Haply Inverse3 + VerseGrip Stylus를 FIRe에서 teleop으로 쓰기 위한 최초 세팅 절차다.
한 번만 하면 되는 설치/설정 위주이며, 실제 녹화 운용은
[inverse3_teleoperator.md](inverse3_teleoperator.md)를 참고한다.

## 1. C++ bridge server 빌드

Inverse3/VerseGrip은 Haply hardware API(C++)로 제어한다. Python에서 subprocess로
띄우는 `inverse3_server` 바이너리를 최초 1회 빌드한다.

```bash
cd src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/src/lerobot_teleoperator_inverse3/inverse3_bridge
make
```

Python 패키지 설치:

```bash
conda activate fire
cd /home/home/FIRe
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

## 2. Inverse3 & VerseGrip Stylus 시리얼 포트 설정

`/dev/ttyACM*` 번호는 재연결/리셋마다 바뀌고 두 장치(Inverse3 = Teensy, VerseGrip =
USB transceiver)끼리 번호가 뒤바뀔 수 있다. udev rule로 USB serial 기준 고정 symlink를
박아두면 항상 같은 이름(`/dev/inverse3_left` 등)으로 접근할 수 있다.

rule 파일 작성:

```bash
sudo gedit /etc/udev/rules.d/99-inverse3-haptic.rules
```

아래 내용을 붙여넣는다.

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="0483", ATTRS{serial}=="16894910", MODE="0666", SYMLINK+="inverse3_left", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", ATTRS{idVendor}=="2fe3", ATTRS{idProduct}=="0100", ATTRS{serial}=="548F7FD33638E628", MODE="0666", SYMLINK+="versegrip_left", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="0483", ATTRS{serial}=="16895170", MODE="0666", SYMLINK+="inverse3_right", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", ATTRS{idVendor}=="2fe3", ATTRS{idProduct}=="0100", ATTRS{serial}=="7B42902DBE133AE4", MODE="0666", SYMLINK+="versegrip_right", ENV{ID_MM_DEVICE_IGNORE}="1"
```

> `ENV{ID_MM_DEVICE_IGNORE}="1"`는 ModemManager가 이 포트를 modem으로 오인하고
> probe(AT 명령)하는 것을 막는다. 이게 없으면 ModemManager가 plug 때마다 포트를
> 열어 serial handshake를 망가뜨려 `timeout waiting for header code`가 난다.
> (5번 문제 해결 참고)

저장하고 rule을 다시 적용한다.

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add
sudo systemctl restart ModemManager   # ENV{ID_MM_DEVICE_IGNORE}를 인식시킨다
```

> `ID_MM_DEVICE_IGNORE`는 *다음 plug*부터 적용되므로, 최초 설정 시에는 위 명령 후
> Inverse3/VerseGrip USB를 **한 번 뽑았다 다시 꽂는** 것이 가장 확실하다.

> 저장소에는 동일한 left-arm rule이 `src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/udev/99-haply-inverse3.rules`로 들어 있다.
> `sudo cp`로 복사해 써도 되지만, 위처럼 직접 작성하면 right-arm까지 한 번에 잡힌다.

각 항목의 `serial` 값은 본인 장치 값으로 바꿔야 한다. serial 확인:

```bash
udevadm info -q property -n /dev/ttyACM2 | grep -E 'ID_SERIAL=|ID_MODEL='
```

- Inverse3 = `idVendor 16c0` / `idProduct 0483` (Teensyduino "Haply inverse3")
- VerseGrip = `idVendor 2fe3` / `idProduct 0100` (ZEPHYR "Haply USB Transceiver")

## 3. 연결 확인

```bash
ls -l /dev/inverse3_left /dev/versegrip_left
ls -l /dev/ttyACM*
```

symlink이 잡혀 있으면 정상이다. 물리 장치가 어느 ttyACM에 매핑됐는지 확인:

```bash
for d in /dev/ttyACM*; do
  echo "=== $d ==="
  udevadm info -q property -n "$d" | grep -E 'ID_MODEL=|DEVLINKS='
done
```

`Haply_inverse3` → `inverse3_*`, `Haply_USB_Transceiver` → `versegrip_*`로 묶이면 OK.

## 4. 동작 점검

robot/ROS 스택 없이 **device만** 빠르게 점검하는 게 문제 격리에 가장 좋다.
bridge server를 직접 띄워 OPEN → STATE → CLOSE를 한 번에 보낸다.

```bash
cd /home/home/FIRe/src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/src/lerobot_teleoperator_inverse3/inverse3_bridge
printf 'OPEN /dev/inverse3_left /dev/versegrip_left\nGET_STATE\nCLOSE\n' | timeout 30 ./inverse3_server
```

정상 출력 예:

```text
[info] Inverse3 gravity compensation: disabled
OK                               # VerseGrip 켜져 있으면 "OK", 꺼져 있으면 "OK GRIP_UNAVAILABLE"
STATE 0 0 0 0 0 0 1 0 0 0 0 0
```

- `OK`(또는 `OK GRIP_UNAVAILABLE`)와 `STATE`가 나오면 **device는 정상**이다.
  이후 record.py가 실패하면 device가 아니라 Python/ROS 쪽 문제다.
- `[err] ... timeout waiting for header code`가 한 번 찍히는 것은 cold start에서
  Inverse3 첫 wakeup이 한 번 timeout 후 재시도하는 정상 동작이다. 결국 `OK`까지
  가면 문제없다.
- 처음 전원을 넣은 직후(cold)는 wakeup 재시도로 `OK`까지 수 초~십수 초 걸릴 수 있다.

기기 + teleop 변환까지 보려면:

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test/test_hardware_live.py --teleop
```

여기까지 되면 녹화 운용은 [inverse3_teleoperator.md](inverse3_teleoperator.md)로 넘어간다.

## 5. 문제 해결

### `timeout waiting for header code` / `No response from inverse3_server within 8.0s`

```text
[err] .../Device.cpp:87> timeout waiting for header code: [0] Success
TimeoutError: [Inv3Server] No response from inverse3_server within 8.0s
```

serial 포트는 열렸지만 Inverse3가 header frame을 제때 못 보내는 상태다. 실제로 이
에러는 원인이 두 가지가 겹쳐서 난다. 아래 순서대로 처리하면 된다.

먼저 device 자체가 정상인지 [4. 동작 점검](#4-동작-점검)의 bridge server 직접 실행으로
격리한다. 직접 실행에서 `OK`가 나오는데 record.py만 실패하면 (B) timeout 문제다.

#### (A) ModemManager가 포트를 probe해서 handshake를 망가뜨림

Ubuntu의 ModemManager가 ttyACM 장치를 modem 후보로 보고 plug 때마다 포트를 열어 AT
명령을 쏜다. 이 probe가 Inverse3/VerseGrip의 serial handshake를 깨서 header를 못 받게
한다. 확인:

```bash
systemctl is-active ModemManager
udevadm info -q property -n /dev/inverse3_left | grep -i ID_MM
```

`ID_MM_CANDIDATE=1`만 있고 `ID_MM_DEVICE_IGNORE=1`이 없으면 ModemManager가 이 포트를
건드리는 중이다. [2번](#2-inverse3--versegrip-stylus-시리얼-포트-설정)의 rule에
`ENV{ID_MM_DEVICE_IGNORE}="1"`이 있는지 확인하고 적용한다:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add
sudo systemctl restart ModemManager
udevadm info -q property -n /dev/inverse3_left | grep ID_MM_DEVICE_IGNORE   # ID_MM_DEVICE_IGNORE=1
```

> **중요:** `ID_MM_DEVICE_IGNORE`는 *다음에 꽂을 때*부터 적용된다. ModemManager가 이미
> 부팅 직후 포트를 잡아 stuck 시켜놓은 경우 위 명령만으로는 안 풀릴 수 있으니,
> **Inverse3/VerseGrip USB를 물리적으로 한 번 뽑았다 다시 꽂는다.** (급하면
> `sudo systemctl stop ModemManager`로 이번 부팅만 끄고 테스트해도 된다.)

#### (B) cold start에서 wakeup이 OPEN timeout보다 오래 걸림

전원을 막 넣은 Inverse3는 첫 `DeviceWakeup`이 한 번 timeout(`timeout waiting for
header code`) 후 재시도하며 깨어난다. 이 cold wakeup이 길어서, Python 쪽 OPEN
timeout이 짧으면 device는 멀쩡한데 Python이 먼저 포기한다 (`gravity compensation:
disabled`까지만 찍히고 timeout). 그래서 `Inverse3Server`의 `open_timeout_s` 기본값을
**25초**로 둔다 (`inverse3_bridge/__init__.py`).

직접 실행(4번)에서는 `OK`가 나오는데 record.py에서만 timeout이 나면 이 경우다.
warm 상태(직전에 한 번 연결)면 빠르게 붙으므로, 한 번 직접 실행으로 깨워둔 뒤
record.py를 돌리는 임시 회피도 가능하다.

#### (C) 그래도 안 되면 (장치 자체 stuck / 전원)

1. Inverse3 USB(및 별도 전원이 있으면 전원도)를 뽑고 ~5초 후 다시 연결한다.
2. symlink 재생성 확인: `ls -l /dev/inverse3_left /dev/versegrip_left`
   (udev가 serial 기준으로 다시 만들어주므로 ttyACM 번호가 바뀌어도 매핑은 유지된다.)
3. 다른 USB 포트/케이블로 바꾼다. Teensy는 전력 부족한 USB hub에 민감하므로 PC 본체
   포트나 powered hub에 직접 연결한다.

### VerseGrip 버튼/회전이 안 잡힐 때 (`OK GRIP_UNAVAILABLE` / `VerseGrip not responding`)

```text
[warn] VerseGrip handle not responding (only invalid frames). Power on / pair the handle ...
OK GRIP_UNAVAILABLE
```

VerseGrip은 무선 stylus다. **전원이 꺼져 있으면** Inverse3 연결(=위치 추종)은 되지만
버튼·회전이 안 잡혀 calibration/gripper 조작을 못 한다. stylus 전원을 켜고 동글과
페어링하면 자동 복구되고, `OK GRIP_UNAVAILABLE` 대신 `OK`로 올라온다.

### symlink만 없고 `/dev/ttyACM*`는 있을 때

udev rule을 다시 적용한다.

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add
```

### `/dev/ttyACM*`도 없을 때

USB 연결 자체가 끊긴 상태다. powered hub 또는 PC 본체 USB 포트에 직접 연결하고
`sudo dmesg -w`로 `over-current condition` 등 전원 경고가 있는지 확인한다.

### 포트는 잡혔는데 다른 프로세스가 점유 중일 때

```bash
lsof /dev/ttyACM*
pgrep -af inverse3_server
```

이전에 죽다 만 `inverse3_server`나 `record.py`가 포트를 잡고 있으면 종료한 뒤 재실행한다.
