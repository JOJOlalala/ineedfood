
# Food11 Model

## Setup

1. **Download and extract the Food11 dataset under folder dataset**
    ```bash
    # Download the dataset to the root folder
    mkdir datset
    wget https://www.kaggle.com/api/v1/datasets/download/vermaavi/food11
    unzip food11.zip # or use the data preprocessing pipeline to unzip
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

The train.ipynb will:
- Load the Food11 dataset from the root folder
- Preprocess and prepare the data
- Train the model
- Evaluate performance in train and val dataset

util.ipynb can retrieve history of previous performance.

eval.ipynb for testing on test set.


## Dataset Structure

```
ineedfood/
├──dataset/
│   ├── food11/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
```

## Requirements

- Python 3.8+
- TensorFlow/PyTorch
- Jupyter