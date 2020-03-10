#!/usr/bin/env bash
for VARIABLE in -20 -15 -10 -5 0 5 10 15 20 -1000 1000
do
  echo ${VARIABLE}
  python long_preprocess.py train mp robot_pouring ${VARIABLE}
  python long_preprocess.py test mp robot_pouring ${VARIABLE}
done
