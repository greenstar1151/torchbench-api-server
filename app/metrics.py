# Reference: https://github.com/shuvozula/eth-runner/blob/8d611d458565e86e90ddcc03462c1240c82b5d38/metrics/start_metrics.py

from pynvml import (
  nvmlInit,
  nvmlShutdown,
  nvmlDeviceGetCount,
  nvmlDeviceGetHandleByIndex,
  nvmlDeviceGetPowerUsage,
  nvmlDeviceGetFanSpeed,
  nvmlDeviceGetTemperature,
  NVML_TEMPERATURE_GPU,
)

DEVICE_NAME_FORMAT = 'nvidia.gpu.%d'

class NvidiaMetrics():
  """
  Used for collecting NVIDIA GPU metrics, by tapping into the underlying NVML library
  available via pynvml. Power usage, Temperature and Fan-Speed are reported for each
  GPU to InfluxDB
  """

  def __init__(self):
    """
    Initialize NVML
    """
    print('Initializing NVML sensors....')
    nvmlInit()

  def __str__(self):
    return 'NVIDIA GPU metrics'

  def __del__(self):
    """
    Cleans up the NVML internal state for all GPUs
    """
    print('Shutting down NVIDA metrics collection....')
    nvmlShutdown()

  def collect_metrics(self):
    """
    Collect NVIDIA GPU metrics (eg: Temperature, Power-Consumption, fan-speed, etc.)
    """
    data_list = []
    for gpu_num in range(nvmlDeviceGetCount()):
      handle = nvmlDeviceGetHandleByIndex(gpu_num)
      device_name = DEVICE_NAME_FORMAT % gpu_num
      power_usage = float(nvmlDeviceGetPowerUsage(handle)) / 1000.0
      fan_speed = nvmlDeviceGetFanSpeed(handle)
      temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
      data_list.append({
        'measurement': device_name,
        'tags': {
          'host': 'minar',
          'gpu': device_name
        },
        'fields': {
          'power_usage': power_usage,
          'fan_speed': fan_speed,
          'temperature': temperature
        }
      })

    return data_list