
# Improving Quality Control in Corrugated Board Production

## Overview

This project aims to enhance the efficiency and accuracy of quality control in the production of corrugated sheets used for home-delivery boxes. By integrating cutting-edge computer vision techniques and large language models, we address key challenges in detecting and resolving malfunctions during the manufacturing process.

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Accomplishments & Next Steps](#accomplishments--next-steps)
- [Contributors](#contributors)
- [License](#license)

## Background

Corrugated Supplies, LLC faces significant challenges in ensuring the quality of corrugated sheets. Traditional quality control processes are time-consuming and prone to human error. This project leverages automation, machine learning, and cloud computing to streamline these processes and improve the accuracy of defect detection.

### Problem Statement

To improve the detection and resolution of manufacturing defects, our research focuses on:

1. Developing a sensor to measure damaged raw materials.
2. Implementing a real-time computer vision model to detect malfunctions in manufacturing.

## Project Structure

```
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── utils.py
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
├── README.md
└── requirements.txt
```

- **data/**: Contains raw and processed data used for model training.
- **notebooks/**: Jupyter notebooks for data preprocessing and model training.
- **src/**: Source code for data processing, model definition, and utility functions.
- **tests/**: Unit tests for the data processing and model scripts.

## Installation

To run this project locally, you'll need Python 3.8+ and the following dependencies:

```bash
pip install -r requirements.txt
```

### Prerequisites

- Python 3.8+
- TensorFlow
- OpenCV
- Azure SDK for Python
- PyTesseract / EasyOCR

## Usage

### 1. Data Preprocessing

Run the data preprocessing script to clean and prepare the data for model training:

```bash
python src/data_processing.py --input data/raw --output data/processed
```

### 2. Model Training

Use the following command to train the model:

```bash
python src/model.py --config config/model_config.json
```

### 3. Running Tests

Ensure everything is working by running the tests:

```bash
pytest tests/
```

## Model Details

### Dimension Measurement

We utilized OpenCV and TensorFlow to create a bounding box on objects and obtain dimensional measurements. The model can accurately measure the length, height, and breadth of corrugated boxes in real-time.

### Anomaly Detection

Early work on anomaly detection shows promising results in identifying defects like tears, waves, and delamination in the material using both custom-trained YOLOv8 models and Google's Vertex AI Vision.

### Label Extraction

OCR tools such as PyTesseract and EasyOCR were employed to extract and analyze labels from paper rolls, with efforts to improve accuracy through image preprocessing techniques.

## Accomplishments & Next Steps

### Accomplishments

- Developed a custom model for blister detection using video feeds.
- Integrated Azure ML for model deployment and API management.
- Implemented OCR for label extraction with a confidence interval of 80-90%.

### Next Steps

- Enhance the accuracy of the damage classification model by collecting more training data.
- Refine the video variance estimation process to improve defect detection.
- Continue to integrate and test cloud solutions from GCP and Azure for better performance.

## Contributors

- Colleen Jung
- Ayan Raheel
- Thomas Shi
- Nia Gangar

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
