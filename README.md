# RoDE
Robust Duality Learning for Unsupervised Visible-Infrared Person Re-Identification (IEEE Transactions on Information Forensics and Security, PyTorch Code)

Authors: Yongxiang Li, Yuan Sun, Yang Qin, Dezhong Peng, Xi Peng and Peng Hu

## Abstract
Unsupervised visible-infrared person re-identification (UVI-ReID) aims to retrieve pedestrian images across different modalities without costly annotations, but faces challenges due to the modality gap and lack of supervision. Existing methods often adopt self-training with clustering-generated pseudo-labels but implicitly assume these labels are always correct. In practice, however, this assumption fails due to inevitable pseudo-label noise, which hinders model learning. To address this, we introduce a new learning paradigm that explicitly considers Pseudo-Label Noise (PLN), characterized by three key challenges: noise overfitting, error accumulation, and noisy cluster correspondence. To this end, we propose a novel Robust Duality Learning framework (RoDE) for UVI-ReID to mitigate the effects of noisy pseudo-labels. First, to combat noise overfitting, a Robust Adaptive Learning mechanism (RAL) is proposed to dynamically emphasize clean samples while down-weighting noisy ones. Second, to alleviate error accumulation-where the model reinforces its own mistakes-RoDE employs dual distinct models that are alternately trained using pseudo-labels from each other, encouraging diversity and preventing collapse. However, this dual-model strategy introduces misalignment between clusters across models and modalities, creating noisy cluster correspondence. To resolve this, we introduce Cluster Consistency Matching (CCM), which aligns clusters across models and modalities by measuring cross-cluster similarity. Extensive experiments on three benchmarks demonstrate the effectiveness of RoDE.

## Dataset Preprocessing
Convert the dataset format (like Market1501).
```shell
python prepare_regdb.py  # for RegDB
```
You need to change the file path in the `prepare_regdb.py`.

## Training
```shell
./train_regdb.sh  # for RegDB
```
Two training stages are included and you need to specify the training stage by commenting another stage's `main_worker` like this:
```python
main_worker_stage1(args,log_s1_name) # Stage 1
main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2
```

## Test
```shell
./test_regdb.sh   # for RegDB
```

## Citation
If you find this work useful in your research, please consider citing:

```
@article{li2025robust,
  title={Robust duality learning for unsupervised visible-infrared person re-identification},
  author={Li, Yongxiang and Sun, Yuan and Qin, Yang and Peng, Dezhong and Peng, Xi and Hu, Peng},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={20},
  pages={1937--1948},
  year={2025},
  publisher={IEEE}
}
```
