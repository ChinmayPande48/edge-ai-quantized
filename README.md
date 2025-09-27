## Crop Disease Classification for Edge Devices

This project demonstrates how to build and optimize a deep learning model for classifying crop diseases on resource-constrained edge devices. It utilizes a lightweight MobileNetV2 architecture and explores different quantization techniques to reduce model size and improve inference speed while maintaining high accuracy.

### 📌 Key Features

  - **Lightweight Architecture:** Uses the MobileNetV2 model pre-trained on ImageNet for efficient feature extraction.
  - **Model Compression:** Implements two post-training quantization techniques:
    1.  **Post-Training Quantization (PTQ)**: Calibrates the model using a representative dataset to convert weights and activations to 8-bit integers.
    2.  **Dynamic Range Quantization (DRQ)**: A simpler technique that quantizes weights to 8-bit integers while dynamically quantizing activations at inference time.
  - **Comprehensive Evaluation:** Compares the full-precision model against the quantized models based on accuracy, F1-score, and file size.
  - **Data-Specific:** Trained and evaluated on the **Rice Leaf Disease Dataset**.

### 📦 Prerequisites

Before running the code, ensure you have the following libraries installed:

  - `tensorflow`
  - `tensorflow-model-optimization`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

### 🚀 Installation

You can install all the required packages using `pip`:

```bash
pip install tensorflow==2.15.0 tensorflow-model-optimization numpy scikit-learn matplotlib
```

### 📁 File Structure

The project assumes the following directory structure. Place the dataset in the root folder as shown.

```
.
├── rice_leaf_disease_dataset/
│   ├── train/
│   │   ├── Bacterial Leaf Blight/
│   │   ├── Brown Spot/
│   │   └── ...
│   └── validation/
│       ├── Bacterial Leaf Blight/
│       └── ...
├── leaf_disease_classification.ipynb              # The main script to train and quantize models
└── README.md
```

### 💻 Usage

1.  **Download the Dataset:** Obtain the **Rice Leaf Disease Dataset** and place it in the project's root directory.
2.  **Update Path:** In the `main.py` script, update the `data_dir` variable to the correct path of your dataset.
3.  **Run the Script:** Execute the main script from your terminal.

<!-- end list -->

```bash
python main.py
```

The script will perform the following steps:

1.  Train the full-precision MobileNetV2 model.
2.  Save the full-precision model as `rice_leaf_fp32.h5`.
3.  Perform PTQ and save the quantized model as `rice_leaf_quantized.tflite`.
4.  Perform DRQ and save the quantized model as `rice_leaf_drq.tflite`.
5.  Print evaluation metrics and file sizes for all three models.

### 📊 Results

The following table summarizes the key metrics, showcasing the trade-off between model accuracy and size.

| Model | Accuracy | Size (MB) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- |
| **Full-Precision** | \~94% | \~9 MB | 0.9374 |
| **PTQ (Quantized)** | 83% | \~2.5 MB | 0.8245 |
| **DRQ (Quantized)** | 92.23% | 2.59 MB | 0.9212 |

***Analysis:** The DRQ model provides a superior balance of size reduction and accuracy retention, making it the most suitable for edge deployment in this scenario.*

### 🙏 Acknowledgements

  - **Dataset:** [Rice Leaf Disease Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/sadiqshahbaz/rice-leafs-diseases-dataset) hosted on Kaggle.
