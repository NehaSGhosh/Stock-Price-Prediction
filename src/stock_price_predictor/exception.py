import sys
from typing import Any


class CustomException(Exception):
    def __init__(self, error_message: Any, error_detail: Any = None):
        super().__init__(str(error_message))
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: Any, error_detail: Any = None) -> str:
        if error_detail is None:
            return str(error_message)
        _, _, exc_tb = error_detail.exc_info() if hasattr(error_detail, "exc_info") else sys.exc_info()
        if exc_tb is None:
            return str(error_message)
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in script: [{file_name}] at line [{line_no}] message: {error_message}"

    def __str__(self) -> str:
        return self.error_message
