This is implementation for the paper "LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching" (LAMP, ICCV 2025)
It is built on top of [HPT](https://github.com/ThomasWangY/2024-AAAI-HPT).


## Requirements 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](https://github.com/ThomasWangY/2024-AAAI-HPT/blob/main/docs/INSTALL.md).

## Download image data
The raw images can be downloaded from their original sources [here](https://cocodataset.org/#download), and [here](https://www.kaggle.com/hsankesara/flickr-image-dataset).

## Download knowledge data
The action knowledge can be downloaded from https://pan.baidu.com/s/14NGtgq-zwvPN86K5EMH3RQ. 提取码: snt6

## Training and Evaluation
For MSCOCO and Flickr30K:
bash ./scripts/hpt/xd_train.sh


## Reference
```
@inproceedings{tian2025llm,
  title={LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching},
  author={Tian, Mengxiao and Wu, Xinxiao and Yang, Shuo},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
