import os, sys
# import logging
from src.logger import logging

def get_error_message(error, sys_object):
    exc_type, exc_obj, exc_traceback = sys_object.exc_info()
    file_path = exc_traceback.tb_frame.f_code.co_filename
    line_no   = exc_traceback.tb_lineno
    message   = f"Exception occured in file: {file_path} at line no: {line_no} and Error message: {error}"
    return  message

class CustomException(Exception):
    # Constructor
    def __init__(self, error, sys_object):
        self.error = error
        self.error_message = get_error_message(self.error, sys_object)

    # __str__ to print the value
    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 2/0
#     except Exception as e:
#         logging.info(f"Exception occured: {e}")
#         raise CustomException(e, sys)