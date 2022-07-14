# bboxRegNet4BldHeight 

This repository contains the codes for [Large-scale building height retrieval from single SAR imagery based on bounding box regression networks](https://www.sciencedirect.com/science/article/pii/S0924271621003221). 



## Code 

The code structure is largely borrowed from https://github.com/jwyang/faster-rcnn.pytorch. Please follow README.md in the above link for the environment and dependencies. To use the code, the SAR & GIS data are to be concatenated and stored in the VOC format. 

Train / Test: 
```
python train.py --dataset pascal_voc --net res101 --bs $BATCH_SIZE --nw $WORKER_NUMBE --cuda --s $SESSION

python test.py --dataset pascal_voc --net res101  --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT --cuda
```
## Citation

If you find the repo useful, please cite the following paper:

```
@article{SUN2022boxhsar,
author = {Yao Sun and Lichao Mou and Yuanyuan Wang and Sina Montazeri and Xiao Xiang Zhu},
title = {Large-scale building height retrieval from single SAR imagery based on bounding box regression networks},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {184},
pages = {79-95},
year = {2022},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2021.11.024},
}
```

