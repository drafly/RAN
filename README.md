# A Relation-aware Network for Defocus Blur Detection

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

## Run

* Run **main.py** scripts.
  
  <pre><code>
  python main.py train

## Test

- Run **main.py** scripts.
  
  ```
  python main.py test
  ```
