"""
ZMQ 카메라 수신 시각화 테스트 (lerobot ZMQCamera 사용)
lerobot ZMQCamera로 3대 카메라를 수신하여 실시간 표시.

실행:
    python test_zmq_viewer.py --host 192.168.x.x --port 5555
"""

import argparse
import time

import cv2
import numpy as np

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig


CAM_NAMES = ["wrist", "left", "right"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="163.180.132.241", help="서버 IP (기본: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 포트 (기본: 5555)")
    parser.add_argument("--timeout", type=int, default=5000, help="수신 타임아웃 ms (기본: 5000)")
    args = parser.parse_args()

    # 카메라별 ZMQCamera 인스턴스 생성 (모두 같은 포트, camera_name으로 구분)
    cameras: dict[str, ZMQCamera] = {}
    for name in CAM_NAMES:
        config = ZMQCameraConfig(
            server_address=args.host,
            port=args.port,
            camera_name=name,
            timeout_ms=args.timeout,
        )
        cameras[name] = ZMQCamera(config)

    print(f"[ZMQ Viewer] 연결 중: tcp://{args.host}:{args.port}")
    for name, cam in cameras.items():
        cam.connect()
        print(f"  {name} 연결 완료 ({cam.width}x{cam.height})")

    print("[ZMQ Viewer] 시작. 종료: 'q' 키")

    fps_times = []

    try:
        while True:
            t0 = time.perf_counter()

            panels = []
            for name, cam in cameras.items():
                try:
                    # 버퍼에 있는 최신 프레임을 즉시 반환 (블로킹 없음)
                    frame = cam.read_latest(max_age_ms=1000)  # RGB numpy array
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = name
                except (TimeoutError, RuntimeError):
                    bgr = np.zeros((256, 256, 3), dtype=np.uint8)
                    label = f"{name} (no frame)"

                cv2.putText(bgr, label, (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                panels.append(bgr)

            # FPS 계산
            fps_times.append(time.perf_counter() - t0)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

            combined = np.hstack(panels)
            cv2.putText(combined, f"FPS: {fps:.1f}", (4, combined.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

            cv2.imshow("ZMQ Viewer (lerobot)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for cam in cameras.values():
            cam.disconnect()
        cv2.destroyAllWindows()
        print("[ZMQ Viewer] 종료.")


if __name__ == "__main__":
    main()
