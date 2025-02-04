import tensorflow as tf
from tensorflow.python.client import device_lib

def get_gpu_info():
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if device.device_type == 'GPU':
            print(f"GPU Device: {device.physical_device_desc}")

get_gpu_info()