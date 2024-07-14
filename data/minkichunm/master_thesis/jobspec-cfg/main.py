from parser import __create_options
from compression import compress_NN_param
from decompression import decompress_NN_param
from utils import load_dataset, format_time
import time

if __name__ == "__main__":
    options = __create_options()
    start_time = time.time()
    
    # Load dataset
    x_train, y_train, x_test, y_test, train_generator, steps_per_epoch = load_dataset(dataset = options["dataset"], batch_size = options["batch_size"])    
        
    # Compression
    compress_NN_param(options, x_train, y_train, x_test, y_test, train_generator, steps_per_epoch)
    
    # Decompression
    decompress_NN_param(options, x_train, y_train, x_test, y_test, train_generator, steps_per_epoch)

    # Test done   
    end_time = time.time() - start_time
    print(f"Execution time: {format_time(end_time)} | Test completed successfully")
