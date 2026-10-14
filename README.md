
# Indian Cricket Team Member Face Identification

## Project Description

Indian Cricket Team Member Face Identification is a machine
learning and computer vision project that classifies images
into selected Indian cricket player categories.

The application uses transfer learning with MobileNetV2
and provides a Streamlit interface for image upload and
prediction.

## Objectives

- Prepare a labeled image dataset of Indian cricket players.
- Clean invalid images and remove exact duplicate files.
- Split the dataset into training, validation, and testing sets.
- Train and fine-tune a pretrained image classification model.
- Evaluate performance on a held-out test set.
- Predict player categories from new images.
- Provide a simple interactive web interface.

## Technologies

- Python
- TensorFlow and Keras
- MobileNetV2
- Pillow
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Kaggle API

## Dataset

Source:
https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset

Review the dataset license, class labels, image quality,
and available player categories before using the images.

## Installation

Create a virtual environment and activate it.

Install dependencies:

    python -m pip install -r requirements.txt

Configure Kaggle API credentials and download the dataset
into dataset/raw/.

Prepare the dataset:

    python src/prepare_dataset.py

Train the model:

    python src/train_model.py

Evaluate the model:

    python src/evaluate_model.py

Launch the application:

    streamlit run app.py

## Generated Files

- models/cricket_player_model.keras
- models/class_names.json
- outputs/training_history.png
- outputs/confusion_matrix.png
- outputs/classification_report.txt

## Limitations

The model can classify only the player categories in its
training data. The score displayed by the model is not a
verified identity or a calibrated probability.

Performance depends on the number, quality, diversity,
and labeling accuracy of the images. Similar photographs
across data splits can inflate reported test accuracy.
The application is an educational classification system,
not a secure biometric identity-verification service.
