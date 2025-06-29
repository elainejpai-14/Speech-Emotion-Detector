# Speech Emotion Detector

This project implements a Speech Emotion Recognition (SER) system using a Convolutional Neural Network (CNN) trained on the RAVDESS dataset. The model takes raw audio input and predicts the speaker's emotional state such as happy, sad, angry, etc., based on speech characteristics.

## Dataset
RAVDESS stands for Ryerson Audio-Visual Database of Emotional Speech and Song. It is a professionally recorded dataset that includes 24 professional actors vocalizing two lexically-matched statements in a neutral North American accent across eight emotions. It contains 7356 audio and video files depicting 8 different emotions like Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised with speech and song modalities.
Download RAVDESS Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

The trained model is available here: https://drive.google.com/file/d/10CJ2bdkjs-yKT1-hMoZRDGMOlnccx1Wg/view?usp=sharing

## Setup Instructions
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```
2. Create virtual environment (optional but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Download and unzip the RAVDESS dataset
5. Run the "Emotion Recognition from Speech.ipynb" jupyter notebook
6. Run the Gradio UI
   ```bash
   python app.py
   ```
## Model Architecture
The model uses a simple CNN with the following components:

- 2 convolutional layers

- Max pooling and dropout

- Flatten + Fully Connected layer for classification

Input: Mel spectrogram (128 Mel bins, 400 time frames)

## Results 

| Metric              | Value                      |
| ------------------- | -------------------------- |
| Validation Accuracy | **\~76.4%**                |
| Test Accuracy       | **77.6%**                  |
| Final Test Loss     | **0.80**                   |
| Early Stopping      | ✅ Triggered after epoch 20 |
