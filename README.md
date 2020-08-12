# BriNet: Towards Bridging the Intra-class andInter-class Gaps in One-Shot Segmentation
By Xianghui Yang, Bairun Wang, Kaige Chen, Xinchi Zhou, Shuai Yi, Wanli Ouyang, Luping Zhou

## Paper

You can find our paper at https://arxiv.org/abs/205.03410


## Citation

If you find BriNet useful in your research, please consider to cite:

```
 ```

## Training
For training your own models, we have included all prototxts in `${OSLSM_HOME}/OSLSM/os_semantic_segmentation/training` directory and the vgg pre-trained model can be found in `snapshots/os_pretrained.caffemodel`.

You will also need to

1) Download/Prepare SBD dataset (http://home.bharathh.info/pubs/codes/SBD/download.html).

2) Set `SBD_PATH` in `${OSLSM_HOME}/OSLSM/code/db_path.py`

3) Set the profile to `fold${FOLD_ID}\_train` for our data layer (check the prototxt files and `${OSLSM_HOME}/OSLSM/code/ss_datalayer.py`) to work.

## Testing
Download PASCAL VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Download trained models from: https://google/drive

We assume you have downloaded the repository into ./checkpoint path.

```shell 
cd ${OSLSM_HOME}
python train.py
```


Update the `$PYTHONPATH`: 

```shell
export PYTHONPATH=${OSLSM_HOME}/OSLSM/code:${OSLSM_HOME}/python:$PYTHONPATH
```

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


## License

The code and models here are available under the same license as Caffe (BSD-2) and the Caffe-bundled models (that is, unrestricted use; see the BVLC model license).


## Contact

For further questions, you can leave them as issues in the repository, or contact the authors directly:
xianghui.yang@sydney.edu.au