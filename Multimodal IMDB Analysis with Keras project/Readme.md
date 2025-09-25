# Multimodal IMDB Genre Classification with Keras

This repository contains a project that performs multimodal genre classification on the IMDB dataset. It involves implementing and training two separate deep learning models in Keras to predict movie genres from two different data types: movie posters (images) and movie overviews (text).

The primary goal is to evaluate and compare the performance of these two models on their respective classification tasks.

---

##  Project Structure

The entire project is contained within a single Jupyter Notebook: `Multimodal IMDB Analysis with Keras project.ipynb`.

The workflow is divided into four main sections:
1.  **Data Processing**: Efficiently loading and preparing the image (posters) and text (overviews) data using `tf.data` pipelines. This includes image resizing, normalization, and text vectorization.
2.  **Model Definition**: Building two distinct models using the Keras API.
3.  **Model Training**: Compiling and training both models using the Adam optimizer, binary cross-entropy loss, and callbacks for checkpointing (`ModelCheckpoint`) and dynamic learning rate adjustment (`LearningRateScheduler`).
4.  **Model Evaluation**: Visualizing the training history (loss, precision, recall) and comparing the top genre predictions from both models against the ground truth for sample films.

---

##  Models & Architectures

Two models are developed for this task:

### 1. Convolutional Neural Network (CNN) for Poster Classification
* **Purpose**: To classify movie posters into one or more genres.
* **Architecture**: A deep CNN built with the Keras Functional API. It consists of multiple convolutional blocks (`Conv2D`, `MaxPooling2D`, `Dropout`) followed by fully connected `Dense` layers for the final multi-label classification.

### 2. Long Short-Term Memory (LSTM) for Overview Classification
* **Purpose**: To classify movie plot overviews into one or more genres.
* **Architecture**: A Sequential model that processes text. Its key layers include:
    * A `TextVectorization` layer to create a vocabulary from the text corpus.
    * An `Embedding` layer to generate dense vector representations of words.
    * Two stacked, bidirectional `LSTM` layers to capture contextual information from the text sequences.
    * `Dense` layers with `Dropout` for the final classification.

---

##  Technologies Used

* **Framework**: TensorFlow & Keras
* **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib
* **Environment**: The notebook is designed to be run in Google Colab, leveraging Google Drive for dataset storage.

---

##  How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    ```
2.  **Environment Setup:**
    * This project is intended for a **Google Colab** environment.
    * Upload the Jupyter Notebook (`.ipynb` file) to your Colab instance.
3.  **Dataset:**
    * Ensure the `Multimodal_IMDB_dataset` (containing the `Images` folder and `IMDB_overview_genres.csv` file) is located in your Google Drive.
    * The notebook contains commands to mount Google Drive and copy the data to the Colab runtime for faster access.
4.  **Execute:**
    * Open the notebook in Colab and run the cells sequentially. The notebook will handle package installations, data processing, model training, and evaluation.