import numpy as np
from scipy.spatial.transform import Rotation


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → (x, y, z, w)"""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) → (w, x, y, z)"""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """
    angle (rad) + unit axis (3,) → quaternion (w, x, y, z)

    angle ≈ 0 이면 identity [1, 0, 0, 0] 반환.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if np.abs(angle) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    rotvec = axis * angle
    r = Rotation.from_rotvec(rotvec)
    return _xyzw_to_wxyz(r.as_quat()).astype(np.float32)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    quaternion 곱셈: q1 * q2  (둘 다 wxyz)
    반환: wxyz
    """
    r1 = Rotation.from_quat(_wxyz_to_xyzw(q1))
    r2 = Rotation.from_quat(_wxyz_to_xyzw(q2))
    return _xyzw_to_wxyz((r1 * r2).as_quat()).astype(np.float32)

def get_euler_xyz(q: np.ndarray) -> np.ndarray:
    """
    quaternion (wxyz) → euler angles [roll, pitch, yaw] (rad), 'xyz' 순서
    반환: np.ndarray (3,) — mutable (caller에서 직접 수정하므로)
    """
    r = Rotation.from_quat(_wxyz_to_xyzw(q))
    # Isaac Lab / _process_action 에서 xyz intrinsic 순서 사용
    return r.as_euler("xyz", degrees=False).astype(np.float32).copy()

def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    euler (roll, pitch, yaw) rad, 'xyz' 순서 → quaternion (wxyz)
    """
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    return _xyzw_to_wxyz(r.as_quat()).astype(np.float32)