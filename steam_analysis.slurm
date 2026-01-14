#!/bin/bash
#SBATCH --job-name=steam_analysis
#SBATCH --partition=debug
#SBATCH --nodelist=altocpu1,altocpu2,altogpu1,altogpu2
#SBATCH --ntasks-per-node=2
#SBATCH --output=steam_analysis.out
#SBATCH --error=steam_analysis.err

# Запускаем анализ игр Steam
# Убедитесь, что файл games.csv находится в директории /mnt/share/
mpirun /mnt/share/steam_analysis /mnt/share/games.csv
