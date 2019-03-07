# Script to check whether any GPUs are being detected.
from keras import backend

backend.tensorflow_backend._get_available_gpus()
