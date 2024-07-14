#!/usr/bin/env python

# wrapper to run 
import os.path,sys
sys.path.insert(0, "mdapi")
from tf_detector import TFDetector

def checkgpu():
    import tensorflow as tf
    print(tf.version.GIT_VERSION, tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def detecttest():
    tf_detector = TFDetector(model_path='models/md_v4.1.0.pb', output_path='/mnt/home/billspat/docs/wilton/output')
    results = tf_detector.run_detection(input_path='/mnt/home/billspat/docs/wilton/input')
    print(results)



if __name__ == "__main__":
    print("tensorflow GPU status...")
    checkgpu()
    print("===========")
  
    input_path=sys.argv[1]  # absolute path, no following slash
    output_path=sys.argv[2]
#   previously, made the output path here, now send as a CLI param
#   output_path=os.path.join("output", os.path.split(os.path.normpath(p))[-1])

    if not os.path.exists(input_path):
        print(f"{input_path}: path not found, exiting")
        sys.exit(1)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print(f"writing to {output_path}")

    model_path = 'models/md_v4.1.0.pb'
    tf_detector = TFDetector(model_path=model_path, output_path=output_path)
    results = tf_detector.run_detection(input_path=input_path)
    print("complete")
	
