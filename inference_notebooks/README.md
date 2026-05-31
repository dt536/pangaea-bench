## Cross-Disaster Inference Notebooks

This folder contains the Google Colab notebooks used for cross-disaster inference. Trained model checkpoints have not yet been uploaded.

### Myanmar and Haiti

The Myanmar and Haiti notebooks are used for inference where no ground-truth labels are available. As a result, no evaluation metrics are calculated; the notebooks only export prediction GeoTIFFs.

For these notebooks, use:

```python
dataset = "earthquake"
```

### Turkey

The Turkey inference notebook remaps the original five damage classes into four classes before calculating evaluation metrics.

For metric calculation, the original xView2 dataset setting can be used with class remapping enabled:

```python
dataset = "xview2"
task.evaluator.remap_classes = True
```

### Beirut

Two Beirut inference notebooks are provided:

1. Inference and metric calculation without class remapping.
2. Inference and metric calculation with class remapping.

The Beirut notebooks use:

```python
dataset = "beirut_new"
```

This is because the Beirut dataset consists of PNG images rather than GeoTIFFs.

### Test Data Structure

The `test.zip` file should contain two required folders and one optional folder:

```text
test.zip
├── pre/
├── post/
└── target/   # optional; required only for metric calculation, should be png files
```

The files in each folder should share a common key, with suffixes indicating the image type:

```text
pre/
└── <common_key>_pre_disaster.*

post/
└── <common_key>_post_disaster.*

target/
└── <common_key>_target.*
```

For example:

```text
pre/
└── tile_001_pre_disaster.tif

post/
└── tile_001_post_disaster.tif

target/
└── tile_001_target.png
```

The `pre` and `post` folders are required for inference. The `target` folder is optional and is only needed when calculating evaluation metrics.
