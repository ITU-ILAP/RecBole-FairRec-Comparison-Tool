#!/bin/bash

#SBATCH -p longq        # İşin çalıştırılması istenen kuyruk seçilir
#SBATCH -o %j.out      # Çalıştırılan kodun ekran çıktılarını içerir
#SBATCH -e %j.err      # Karşılaşılan hata mesajlarını içerir 
#SBATCH -n 2           # Talep edilen işlemci  çekirdek sayısı
#SBATCH --gres=gpu:2   # Talep edilen GPU sayısı

source activate recbole_venv

python /okyanus/users/atosun/ilap/fairrec/RecBole-FairRec-Comparison-Tool/new_run_compare.py
