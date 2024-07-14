import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print(f"Yes, there are {len(tf.config.list_physical_devices('GPU'))} GPUs available.")
else:
    print('No, GPUs are not available.')
