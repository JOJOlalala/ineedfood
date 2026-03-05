
# Food11 Model

## Setup

1. **Download and extract the Food11 dataset under folder dataset**
    ```bash
    # Download the dataset to the root folder
    unzip food11.zip
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

The notebook will:
- Load the Food11 dataset from the root folder
- Preprocess and prepare the data
- Train the model
- Evaluate performance

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
