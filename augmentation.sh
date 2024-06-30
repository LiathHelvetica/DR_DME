#!/bin/bash

arr=(528 600 480 518 238 246 272 260) # done till 456 (including)

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_$i
  python augmentation_sh_pipeline.py ${i}
done