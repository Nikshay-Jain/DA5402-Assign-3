# Handwritten Digit Classifier

This project implements a handwritten digit classification system using a neural network model. The system consists of a FastAPI-based server for inference and a Tkinter-based GUI application for drawing and predicting digits.

## Features
- **FastAPI Server**: Serves the trained model for digit classification.
- **Tkinter GUI**: Provides a canvas for drawing digits and making predictions.
- **Configurable API URL**: The client (app.py) and the server (server.py) allows specifying the API endpoint via command line arguments.
- **Model Loading**: Loads a pre-trained model (`model.pkl`) for inference.

## Installation
### Prerequisites
Ensure you have Python 3.9 or later installed on your system.

### Step 1: Clone the Repository
```sh
git clone https://github.com/DA5402-MLOps-JanMay2025/assignment-03-Nikshay-Jain.git
cd assignment-03-Nikshay-Jain
cd HandwrittenDigitClassifier
```

### Step 2: Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application

### Step 1: Start the Server
Run the server, specifying the server URL:
```sh
python server.py --url http://127.0.0.1:7070/predict
```

### Step 2: Run the GUI Application
Run the Tkinter-based client, specifying the server URL:
```sh
python app.py --url http://127.0.0.1:7070/predict
```
Make sure you put the link to the same port in both the files to avoid any error.

## Usage
1. Draw a digit on the canvas.
2. Click **Predict The Digit** to send the drawing to the server for classification.
3. Click **Erase** to clear the canvas.
4. The predicted digit will be displayed in a popup.

## File Structure
```
HandwrittenDigitClassifier/
|── server.py              # FastAPI server for model inference
│── app.py                 # Tkinter GUI for drawing and prediction
│── model.pkl              # Pre-trained neural network model
│── dense_neural_class.py   # Neural network implementation
│── utils.py               # Helper functions
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Troubleshooting
- **Server not starting?** Ensure `model.pkl` exists in the project directory.
- **Connection error in `app.py`?** Ensure the correct API URL is provided using `--url`.
- **Permission issues?** Try running with administrator privileges.