# A Relation-aware Network for Defocus Blur Detection

## Abstract

Defocus blur detection (DBD) is an important task in computer vision that aims to segment clear regions from images. Recently, deep learning based methods have made great progress in the defocus blur detection task based on their powerful learning capabilities. However, most of existing methods often directly predict clear regions without considering the complementary relationship between clear and blur context information, which produce clutter and low confident predictions in boundary areas. To address this challenge, we propose a relation-aware network for defocus blur detection. Specifically, we disentangle the complementary relationship both from region level and pixel level. For region level, we introduce the separated attention mechanism to highlight the contrast between clear and blur areas of an image, where the normal attention is benefit to distinguish the clear region, and the reverse attention helps to focus on blur region. These two-stream separated attention module would generate the segmentation mask with high confidence. Furthermore, we try to uncover the pixel-to-pixel relationship via the connectivity contour in eight directions
which can enhance the accuracy of contour detection. To evaluate the superiority of the proposed method, we implement extensive experiments on two two public benchmark datasets, CUHK and DUT. The experimental results demonstrate that our method achieves state-of-the-art performance.

## Framework
![Image text](https://github.com/WAbur/RAN/blob/main/framework.png)

## Data structure
use "contour_generator.py" to generate the contour masks

<pre><code>
dataset
├── Train
│   ├── DUT
│   │   ├── images
│   │   ├── masks
│   │   ├── contour
│   ├── CUHK
│   │   ├── images
│   │   ├── masks
│   │   ├── contour
├── DUT
│   ├── Test
│   │   ├── images
│   │   ├── masks
├── CUHK
│   ├── Test
│   │   ├── images
│   │   ├── masks
</code></pre>

Download and unzip datasets from https://github.com/shangcai1/SG [1] to "./dataset"

## Requirements

* Python >= 3.7.x
* Pytorch >= 1.8.0
* albumentations >= 0.5.1
* tqdm >=4.54.0
* scikit-learn >= 0.23.2

## Train
```
  python main.py train --data_pth /data_path/ --model_path /model_path/ --batch_size 2
```

## Test  
  ```
  python main.py test --data_path /data_path/ --model_path /model_path/ --bact_size 2 --save_map True
  ```
'save_map' means method will save the predicted results. If you needn't the results. please set 'save_map' as 'False'.

## citation
<pre><code>
@inproceedings{wang2023relation,
  title={A Relation-Aware Network for Defocus Blur Detection},
  author={Wang, Yi and Huang, Peiliang and Han, Longfei and Xu, Chenchu},
  booktitle={2023 7th Asian Conference on Artificial Intelligence Technology (ACAIT)},
  pages={66--74},
  year={2023},
  organization={IEEE}
}
</code></pre>
