"""Hàm tiện ích xác định đường dẫn gốc của project (BASE_DIR)."""

import os


def get_base_dir(current_file):
    """
    Trả về đường dẫn gốc của project (Project-I/) dựa trên vị trí
    của file đang gọi hàm này.

    Ví dụ: nếu current_file = src/inference/app.py
    thì BASE_DIR sẽ là Project-I/ (đi lên 2 cấp từ src/inference/).
    """
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
    )
