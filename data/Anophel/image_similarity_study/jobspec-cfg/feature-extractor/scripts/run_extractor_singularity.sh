pip install --no-cache-dir opencv-python-headless==4.5.5.62
pip install --no-cache-dir torch==1.9.0
pip install --no-cache-dir matplotlib==3.3.4
pip install --no-cache-dir alive_progress==2.2.0
pip install --no-cache-dir transformers==4.16.2
pip install --no-cache-dir scikit_image==0.19.1
pip install --no-cache-dir mxnet-cu112==1.8.0.post0
pip install --no-cache-dir tensorflow==2.6.0
pip install --no-cache-dir keras==2.6.0
pip install --no-cache-dir Pillow==9.0.0
pip install --no-cache-dir scikit_learn==1.0.2

cd /scratch

echo "GPU information:"
nvidia-smi
echo "***"

python feature-extractor/extract_images.py #EXTRACT# -i ./imagelist.txt -o ./output -v
