# Next Character Prediction

This project is part of the ES335 Machine Learning course and focuses on building a model for predicting the next character in a sequence of text using PyTorch. The model is trained on text data and utilizes embeddings and neural networks to make predictions.

## Project Structure

The project consists of the following key components:

1. **Data Preprocessing**
   - Functions to preprocess text data by converting characters to ASCII values and creating training sequences.

2. **Model Definition**
   - `NextChar1` class defining a neural network with embedding and linear layers for character prediction.
   
3. **Model Training**
   - Functions to build and train the model using the preprocessed data.

4. **Embedding Visualization**
   - Visualization of embeddings using t-SNE to reduce dimensions and plot the character embeddings.

5. **Character Prediction**
   - Functions to predict the next characters based on the trained model and user input.

## Streamlit Application

This project includes a Streamlit application that allows users to interact with the trained model through a web interface. Users can enter some text and the app will predict the next characters based on the trained model.

### How to Run the Application

1. Ensure you have the required dependencies installed:
   
   ```bash
   pip install torch scikit-learn matplotlib streamlit
2. Run the Streamlit application:

    ```bash
    streamlit run app.py

3. Open your web browser and navigate to the URL provided by Streamlit to interact with the application.

### Application Features
  - Text Input: Users can enter a sequence of text and a value k.
  - Prediction: The app predicts the next k characters based on the input text.
  - Visualization: Users can visualize the character embeddings using t-SNE.
