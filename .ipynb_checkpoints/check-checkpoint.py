import os
import tensorflow as tf

# Set environment variables
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Print system information
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("cuDNN available:", tf.test.is_built_with_cudnn())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Try GPU operation
try:
    with tf.device('/GPU:0'):
        print("Attempting GPU computation...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("GPU computation successful!")
        print(c)
except RuntimeError as e:
    print("GPU computation failed:", e)