#!/bin/bash
#SBATCH --job-name=my_job       # Nombre del trabajo
#SBATCH --output=output.out  # Nombre del archivo de salida
#SBATCH --error=error.err    # Nombre del archivo de error
#SBATCH --ntasks=1              # Número de tareas (procesos) que se ejecutarán
#SBATCH --cpus-per-task=8       # Número de CPUs por tarea
#SBATCH --mem=16G               # Memoria total asignada
#SBATCH --time=01:30:00         # Tiempo máximo de ejecución (horas:minutos:segundos)
#SBATCH -N 1                    # Número de nodos
#SBATCH --gres=gpu:2            # Número de GPUs por tarea

ml scikit-learn/0.23.2-foss-2020b
ml NLTK/3.7-foss-2020b
ml PyTorch-Geometric/2.0.2-foss-2020b-PyTorch-1.10.0-CUDA-11.4.3
ml Transformers/4.24.0-foss-2020b
ml datasets/2.10.1-foss-2020b-Python-3.8.6

# Ejecutamos el script Python
python transformer.py
