import subprocess
command = "python src/hotdog_classifier/main.py --config-name='HotDogClassifier.yaml' wandb.tag='FINAL_TEST'"

if __name__ == "__main__":
    subprocess.call(f"{command} models='BASIC_CNN'",shell=True)
    subprocess.call(f"{command} models='efficientnet'",shell=True)
    subprocess.call(f"{command} models='googlenet'",shell=True)

