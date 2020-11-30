# Regression Before Classification for Temporal Action Detection
This repository contains the code for the ICASSP2020 paper "[Regression Before Classification for Temporal Action Detection](https://ieeexplore.ieee.org/document/9053319)".

## Highly Recommended: 
Download preprocessed data by visiting [Onedrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/Eh9uvfgl5CxCmgvaxoWHKfYBuBmd2QVBhDw6SYgzqcA3ZA?e=beDJuO) provided by [Decouple-SSAD](https://github.com/HYPJUDY/Decouple-SSAD).


## Environment
* TensorFlow
* One or more GPU with 12G memory
* pandas, numpy

Have been tested on Ubuntu 16.04
* CUDA 9.0, Python3.5/3.6, tensorflow 1.12.0

## Run code

1. (Optinal) [Prepare THUMOS14 data](data/README.md).
2. Specify your path in [`config.py`](config.py) (e.g. `feature_path`, `get_models_dir()`, `get_predict_result_path()`).
3. Modify `run.sh` to specify gpu(s) device and other parameters according to your need.

Training logs are saved at `logs` folder.
Tensorboard logs are saved at `logs` folder too.
Models are save at `models` folder.
Results are saved at `results` folder.

## Citation
If you like this paper or code, please cite us:
```
@INPROCEEDINGS{9053319,
  author={C. {Jin} and T. {Zhang} and W. {Kong} and T. {Li} and G. {Li}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Regression Before Classification for Temporal Action Detection}, 
  year={2020},
  pages={1-5},
  doi={10.1109/ICASSP40776.2020.9053319}}
```

## Reference

This implementation largely borrows from [Decouple-SSAD](https://github.com/HYPJUDY/Decouple-SSAD) by Yupan Huang.
