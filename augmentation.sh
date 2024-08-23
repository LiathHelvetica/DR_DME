#!/bin/bash

arr=(320 384 480 456 528 600 518)

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_b4_$i
  python augmentation_sh_pipeline.py ${i}
done
