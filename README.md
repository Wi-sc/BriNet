# BriNet: Towards Bridging the Intra-class andInter-class Gaps in One-Shot Segmentation
By Xianghui Yang, Bairun Wang, Kaige Chen, Xinchi Zhou, Shuai Yi, Wanli Ouyang, Luping Zhou

## Paper

You can find our paper at https://arxiv.org/abs/205.03410


## Citation

If you find BriNet useful in your research, please consider to cite:

```
@inproceedings{shaban2017one,
 title={One-Shot Learning for Semantic Segmentation},
 author={Shaban, Amirreza and Bansal, Shray and Liu, Zhen and Essa, Irfan and Boots, Byron},
 journal={British Machine Vision Conference ({BMVC})},
 year={2017}
}
 ```

## Instructions for Testing (tested on Ubuntu 16.04)
We assume you have downloaded the repository into ${OSLSM_HOME} path.

Install Caffe prerequisites and build the Caffe code (with PyCaffe). See http://caffe.berkeleyvision.org/installation.html for more details

```shell 
cd ${OSLSM_HOME}
python train.py
```

If you prefer Make, set BLAS to your desired one in Makefile.config. Then run:

```shell
cd ${OSLSM_HOME}
python test.py
```

Update the `$PYTHONPATH`: 

```shell
export PYTHONPATH=${OSLSM_HOME}/OSLSM/code:${OSLSM_HOME}/python:$PYTHONPATH
```

Download PASCAL VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Download trained models from: https://gtvault-my.sharepoint.com/:u:/g/personal/ashaban6_gatech_edu/EXS5Cj8nrL9CnIJjv5YkhEgBQt9WAcIabDQv22AERZEeUQ

Set `CAFFE_PATH=${OSLSM_HOME}` and `PASCAL_PATH` in `${OSLSM_HOME}/OSLSM/code/db_path.py` file

Run the following to test the models in one-shot setting:

```shell
cd ${OSLSM_HOME}/OSLSM/os_semantic_segmentation
python test.py deploy_1shot.prototxt ${TRAINED_MODEL} ${RESULTS_PATH} 1000 fold${FOLD_ID}_1shot_test
```

Where ${FOLD_ID} can be 0,1,2, or 3 and ${TRAIN_MODEL} is the path to the trained caffe model. Please note that we have included different caffe models for each ${FOLD_ID}.

Simillarly, run the following to test the models in 5-shot setting:

```shell
cd ${OSLSM_HOME}/OSLSM/os_semantic_segmentation
python test.py deploy_5shot.prototxt ${TRAINED_MODEL} ${RESULTS_PATH} 1000 fold${FOLD_ID}_5shot_test
```

For training your own models, we have included all prototxts in `${OSLSM_HOME}/OSLSM/os_semantic_segmentation/training` directory and the vgg pre-trained model can be found in `snapshots/os_pretrained.caffemodel`.

You will also need to

1) Download/Prepare SBD dataset (http://home.bharathh.info/pubs/codes/SBD/download.html).

2) Set `SBD_PATH` in `${OSLSM_HOME}/OSLSM/code/db_path.py`

3) Set the profile to `fold${FOLD_ID}\_train` for our data layer (check the prototxt files and `${OSLSM_HOME}/OSLSM/code/ss_datalayer.py`) to work.

## License

The code and models here are available under the same license as Caffe (BSD-2) and the Caffe-bundled models (that is, unrestricted use; see the BVLC model license).


## Contact

For further questions, you can leave them as issues in the repository, or contact the authors directly:
xianghui.yang@sydney.edu.au