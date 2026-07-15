# Bangalore House Price Prediction

This repository contains a machine learning project designed to predict residential real estate prices in Bangalore, India. It leverages a Random Forest Regressor to provide accurate pricing estimates based on various property attributes.

## Repository Contents

* **`House_Price_Predictor.py`**: The core Python script used to run the house price prediction logic.
* **`randomforest_regressor.pkl`**: The trained machine learning model saved as a serialized pickle file.
* **`house_label_encoders.pkl`**: Serialized label encoders used to preprocess categorical features (like location) into numerical formats.
* **`requirements.txt`**: The file listing all necessary Python packages and dependencies required to execute the script.

## Getting Started

Follow these steps to set up and run the predictor on your local machine.

### Prerequisites

Make sure you have Python installed (version 3.8 or higher is recommended).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Bangalore_House_Price_Prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

To use the model for predicting property values, you can run the primary script:

```bash
python House_Price_Predictor.py
```

*The script loads the pre-trained `randomforest_regressor.pkl` model along with the `house_label_encoders.pkl` to process inputs, evaluate feature values, and output estimated prices.*

## Technologies Used

* **Python**: Core programming language.
* **Scikit-Learn**: For the Random Forest Regressor algorithm and label encoding tools.
* **Pickle**: For saving and loading the serialized model and data preprocessing components.
