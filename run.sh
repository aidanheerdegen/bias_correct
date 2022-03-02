#!/usr/bin/env bash
#PBS -P w97
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -q normalbw
#PBS -l ncpus=28
#PBS -l wd
#PBS -l storage=gdata/w97+gdata/hh5
#PBS -M l.teckentrup@student.unsw.edu.au
#PBS -m abe

module use /g/data/hh5/public/modules
module load conda/analysis3 parallel

declare -a experiments=(
    'EC-Earth3-Veg'
    'KIOST-ESM'
    'NorESM2-MM'
    'INM-CM4-8'
    'MPI-ESM1-2-HR'
    )

declare -a methods=(
    'QM'
    'QDM'
    'MRec'
    'dOTC'
    'CDFt'
    )

parallel -v python correct.py --experiment {1} --method {2} --month_num {3} \
                              ::: "${experiments[@]}" \
                              ::: "${methods[@]}" \
                              ::: {1..12}
