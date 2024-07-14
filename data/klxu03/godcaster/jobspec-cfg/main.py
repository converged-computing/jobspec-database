from captions import WhisperXRunner
import argparse

def main(input_dir, output_dir, model_name, compute_type, batch_size):
    runner = WhisperXRunner(input_dir, output_dir, model_name=model_name, compute_type=compute_type, batch_size=batch_size)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe videos in a directory")
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-m', '--model_name', default="large-v3")
    parser.add_argument('-c', '--compute_type', default="float16")
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_name, args.compute_type, args.batch_size)