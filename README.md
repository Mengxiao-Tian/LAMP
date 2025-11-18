This is implementation for the paper "LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching" (LAMP, ICCV 2025)
It is built on top of [HPT](https://github.com/ThomasWangY/2024-AAAI-HPT).


## Requirements
Follow the installation and dependency instructions in [INSTALL.md](https://github.com/ThomasWangY/2024-AAAI-HPT/blob/main/docs/INSTALL.md).

## Download Image Data
Download the raw image datasets from the official sources: [MSCOCO](https://cocodataset.org/#download) and [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset).

## Download Knowledge Data
Action knowledge can be obtained from [Baidu Netdisk](https://pan.baidu.com/s/14NGtgq-zwvPN86K5EMH3RQ) with extraction code **snt6**.

## Training and Evaluation
Run the script below to train and evaluate on both MSCOCO and Flickr30K:

```bash
bash ./scripts/hpt/xd_train.sh
```

## Reference
```
@inproceedings{tian2025llm,
  title     = {LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching},
  author    = {Tian, Mengxiao and Wu, Xinxiao and Yang, Shuo},
  booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
