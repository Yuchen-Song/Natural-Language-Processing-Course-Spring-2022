#!/bin/bash
#PBS -N lab10
#PBS -e ./error_log.txt
#PBS -o ./outptu_log.txt

echo Start of calculation
# python lab10_skeleton.py
python lab5_ans.py
echo End of calculation