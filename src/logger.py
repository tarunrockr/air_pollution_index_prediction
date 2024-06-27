import logging, os
from datetime import date, datetime

today_date = date.today().strftime("%d_%m_%Y")
log_folder_path = os.path.join(os.getcwd(), "logs", today_date)
os.makedirs(log_folder_path, exist_ok=True)

time_str      = datetime.now().strftime("%d_%m_%Y_%H%M%S")
log_file_name = "log_"+time_str+".log"
file_path     = os.path.join(log_folder_path, log_file_name)

logging.basicConfig(filename=file_path, level=logging.INFO, format= "[%(asctime)s] %(lineno)d %(name)s  - %(levelname)s:%(message)s")

# if __name__ == "__main__":
#     logging.info("Logging test")