#!/bin/bash
#SBATCH --reservation=akya
#SBATCH --account=egitim
#SBATCH -p akya-cuda        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A egitim32         # Kullanici adi
#SBATCH -J print_gpu        # Gonderilen isin ismi
#SBATCH -o print_gpu.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=1:00:00      # Sure siniri koyun.

# Modüller
# Çalıştırılacak komutlar
