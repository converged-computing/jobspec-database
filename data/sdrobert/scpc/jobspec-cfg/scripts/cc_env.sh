module load python/3.10 cmake boost gcc
if [ ! -f "env/bin/activate" ]; then
    echo "No environment!"
    exit 1
fi
source env/bin/activate