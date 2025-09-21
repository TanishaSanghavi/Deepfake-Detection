# Deepfake Image Detector

A powerful application that uses deep learning and frequency domain analysis to detect AI-generated and manipulated images. This tool is designed to help users verify the authenticity of visual content in a world increasingly filled with deepfakes and misinformation.

You can try out the live demo of this application here:
[https://deep-guard.streamlit.app/](https://deep-guard.streamlit.app/)

## Key Features

- **Frequency Domain Analysis:** The core of the detection method involves converting images to the frequency domain using a Fast Fourier Transform (FFT). This technique is effective because deepfake models often leave detectable artifacts or unusual patterns in the high-frequency components of an image, which are not easily seen by the human eye.

- **Deep Learning Model:** A custom Convolutional Neural Network (CNN) is trained on a dataset of real and fake images (Celeb-DF, FaceForensics++). The model learns to identify the subtle patterns and inconsistencies in the frequency data that are characteristic of manipulated content.

- **Streamlit Web Interface:** The application is packaged in a user-friendly web interface built with Streamlit, allowing for easy image uploads and real-time analysis.

- **Confidence Scoring:** Each prediction is accompanied by a confidence score, providing more transparency and context to the user about the model's certainty.

## How It Works

1.  **Image Upload:** The user uploads an image file (JPG, JPEG, or PNG).
2.  **Preprocessing:** The image is converted to grayscale and resized to 64x64 pixels to standardize the input for the model.
3.  **FFT Processing:** The `optimized_fft_processing` function applies a low-pass filter in the frequency domain. This process isolates the low-frequency components (the general structure of the image) and removes high-frequency noise, which helps the model focus on key detection features.
4.  **Model Inference:** The processed image, now containing the frequency-domain artifacts, is fed into the pre-trained deep learning model.
5.  **Prediction:** The model outputs a single value between 0 and 1. A value closer to 0 indicates a real image, while a value closer to 1 suggests a deepfake.
6.  **Results Display:** The application interprets this value and displays a clear result (Real, Deepfake, or Uncertain) along with a confidence meter.

## Setup and Installation

### Prerequisites

* Python 3.x
* `pip` package manager

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file should contain the following libraries:*
    ```
    streamlit
    tensorflow
    numpy
    opencv-python
    Pillow
    scikit-image
    ```
3.  **Download the model:**
    Place the `optimized_deepfake_model.h5` file in the same directory as `app.py`. (This file is not included in the repository and needs to be downloaded separately from the project owner).

## Running the Application

To start the Streamlit application, run the following command from your terminal:

```bash
streamlit run app.py