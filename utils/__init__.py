import time
from .save_results import SaveResults
from .camera_movements import get_cam_poses
from .convert_model import convert_model

def get_current_time():
    time_struct = time.gmtime()
    # construct a folder name from the current time
    folder_name = 'y_'
    folder_name += str(time_struct.tm_year) + '_d_'
    folder_name += str(time_struct.tm_yday).zfill(3) + '_h_'
    folder_name += str(time_struct.tm_hour).zfill(2) + '_m_' + str(time_struct.tm_min).zfill(2)
    return folder_name
