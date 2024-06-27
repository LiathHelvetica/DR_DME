#!/bin/bash

arr=(236 230 342 256 288 320 456 528 600 480 518 238 246 272 260) # done: 384 224 232

for i in "${arr[@]}"
do
  mkdir /home/liath/DR_DME_DATA/aug_out_$i
  python augmentation_sh_pipeline.py ${i}
done