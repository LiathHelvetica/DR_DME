#!/bin/bash

arr=(260 272 246 238 288 256 342 230 236 224 232)

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_b2_$i
  python augmentation_sh_pipeline.py ${i}
done