# A Relation-aware Network for Defocus Blur Detection

## Abstract

Defocus blur detection (DBD) is an important task in computer vision that aims to segment clear regions from images. Recently, deep learning based methods have made great progress in the defocus blur detection task based on their powerful learning capabilities. However, most of existing methods often directly predict clear regions without considering the complementary relationship between clear and blur context information, which produce clutter and low confident predictions in boundary areas. To address this challenge, we propose a relation-aware network for defocus blur detection. Specifically, we disentangle the complementary relationship both from region level and pixel level. For region level, we introduce the separated attention mechanism to highlight the contrast between clear and blur areas of an image, where the normal attention is benefit to distinguish the clear region, and the reverse attention helps to focus on blur region. These two-stream separated attention module would generate the segmentation mask with high confidence. Furthermore, we try to uncover the pixel-to-pixel relationship via the connectivity contour in eight directions
which can enhance the accuracy of contour detection. To evaluate the superiority of the proposed method, we implement extensive experiments on two two public benchmark datasets, CUHK and DUT. The experimental results demonstrate that our method achieves state-of-the-art performance.

## Framework

![](C:\Users\xduwa\AppData\Roaming\marktext\images\2023-06-01-14-59-23-image.png)

## Output

![](C:\Users\xduwa\AppData\Roaming\marktext\images\2023-06-01-15-00-09-image.png)

## Data structure

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
│   │   ├── Test
├── DUT
│   ├── Test
│   │   ├── images
│   │   ├── masks
├── CUHK
│   ├── Test
│   │   ├── images
│   │   ├── masks
</code></pre>

## Requirements

* Python >= 3.7.x
* Pytorch >= 1.8.0
* albumentations >= 0.5.1
* tqdm >=4.54.0
* scikit-learn >= 0.23.2

## Train

* Run **main.py** scripts.
  
  <pre><code>
  python main.py train --data_pth /data_path/ --model_path /model_path/ --batch_size 2

Our method only support training which set the size of batch as 2.

## Test

- Run **main.py** scripts.\
  
  ```
  python main.py test --data_path /data_path/ --model_path /model_path/ --bact_size 2 --save_map True
  ```

'save_map' means method will save the predicted results. If you needn't the results. please set 'save_map' as 'False'.
