"""
Unit test cơ bản cho các hàm xử lý pose trong project.
Chạy: pytest tests/
"""

import numpy as np

from src.utils.inference_utils import pose_to_vector


class FakeLandmark:
    """Giả lập 1 landmark của MediaPipe (có x, y, z, visibility)."""

    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_pose_to_vector_shape():
    """Vector đặc trưng từ 33 landmarks phải có độ dài 33 * 4 = 132."""
    landmarks = [FakeLandmark(0.1, 0.2, 0.3, 0.9) for _ in range(33)]

    vec = pose_to_vector(landmarks)

    assert vec.shape == (132,)


def test_pose_to_vector_values():
    """Giá trị trong vector phải đúng thứ tự x, y, z, visibility."""
    landmarks = [FakeLandmark(1, 2, 3, 4)]

    vec = pose_to_vector(landmarks)

    assert np.array_equal(vec, np.array([1, 2, 3, 4]))
