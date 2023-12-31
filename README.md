
# Hero Name Recognition (LOL) - Sample Solution

This repository contains a sample solution for Hero Name Recognition in the context of League of Legends (LOL).


![Pipe Line](./Data/pipeline.png)

Because I have provided the checkpoint.pth file and Data on GitHub for convenience, it is a bit large in size and will take some time during the git clone process.

```bash
git clone https://github.com/TrNgTinh/Hero-Name-Recognition.git
```

## Environment Setup

Install required packages:

    ```bash
    # Install Python 3.10
    pip install -r requirements.txt
    ```



## Inference
The argument is `--folder_path`. The results will be saved to the file test.txt. Due to the integrated Faiss, it will take some time to build the Faiss index.

```bash
python src/main.py --folder_path './test_images'
```
    
## Training
### Dataset

Download and save data file for LOL champions

```bash
python src/crawl_data.py
```
Data is stored in Data/Champions.

Add more data from test_images given.

```bash
python src/box_detect.py
```
### Argument Data 

```bash
python src/argument_data.py
```
    
### Train    

```bash
cd src/arcface_torch
pip install -r requirements.txt
python train_v2.py --config configs/ms1mv2_r50.py
 ```

## Todo

## Acknowledgments


