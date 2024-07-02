#!/bin/bash

arr=(260 272 246 238 518 480 600 528 456 320 288 256 342 230 236 384 224 232)

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_plain_$i
  python augmentation_sh_pipeline.py ${i}
done