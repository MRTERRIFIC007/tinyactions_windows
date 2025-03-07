# TinyActions Video Recognition

This repository contains code for training and evaluating a video recognition model on the TinyVIRAT dataset.

## Setup

### Environment Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch torchvision tqdm scikit-learn matplotlib
   ```

### Data Setup

The code expects the TinyVIRAT_V2 dataset with the following structure:

```
TinyVIRAT_V2/
├── videos/
│   ├── train/
│   ├── val/
│   └── test/
├── tiny_train_v2.json
├── tiny_val_v2.json
├── tiny_test_v2_public.json
└── class_map.json
```

### Configuration

The code uses a configuration file (`config.py`) to specify paths and parameters. You can configure the data location in two ways:

1. **Environment Variable**: Set the `TINYACTIONS_DATA_DIR` environment variable to point to your TinyVIRAT_V2 directory:

   ```
   export TINYACTIONS_DATA_DIR=/path/to/your/TinyVIRAT_V2
   ```

2. **Edit config.py**: If you can't set environment variables (e.g., in some Jupyter environments), you can directly edit the `BASE_DIR` in `config.py`.

## Running the Code

### Training

To train the model:

```
python3 train.py
```

### In Jupyter Notebook

To run in a Jupyter notebook on another computer:

1. Make sure the TinyVIRAT_V2 dataset is available on that computer
2. Set the `TINYACTIONS_DATA_DIR` environment variable in your notebook:
   ```python
   import os
   os.environ['TINYACTIONS_DATA_DIR'] = '/path/to/your/TinyVIRAT_V2'
   ```
3. Or modify the `BASE_DIR` in `config.py` to point to your dataset location

## Troubleshooting

If you encounter path-related errors:

1. Run `python3 config.py` to check if the paths are correctly configured
2. Ensure all the dataset files exist at the specified locations
