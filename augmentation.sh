#!/bin/bash

arr=(224 232)

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_b2_$i
  python augmentation_sh_pipeline.py ${i}
done