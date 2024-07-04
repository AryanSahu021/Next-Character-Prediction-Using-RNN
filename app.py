import streamlit as st
import joblib
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_text(text):
    # Convert characters to ASCII values
    ascii_values = [ord(char) for char in text]
    return ascii_values

# Function to create training sequences
def create_sequences(text, seq_length):
    sequences = []
    for i in range(len(text) - seq_length-1):
        seq = text[i:i+seq_length+1]
        sequences.append(seq)
    return sequences

# Function to predict next k characters
def predict_next_chars(text, k, model, seq_length):
    input_seq = np.array(preprocess_text(text))
    if len(input_seq) < seq_length:
        # Pad zeros to the beginning if input text is shorter than seq_length
        input_seq = np.pad(input_seq, (seq_length - len(input_seq), 0), mode='constant')
    else:
        input_seq = input_seq[-seq_length:]  # Get only the last seq_length characters
    input_seq = np.reshape(input_seq, (1, -1))
    input_seq = torch.tensor(input_seq).to(device)  # Get only the last seq_length characters
    predicted_chars = ''
    for i in range(k):
      with torch.no_grad():
        y_pred = model(input_seq)
        prediction = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        predicted_chars += chr(prediction)
        input_seq = torch.cat((input_seq[:, 1:], torch.tensor(prediction).to(device).view(1,-1)),axis=1)
        input_seq = input_seq.reshape((1, -1))
    return predicted_chars

def plot_emb_2d(prev_emb, loaded_emb, itos):
    # Compute t-SNE transformation for prev_emb
    tsne = TSNE(n_components=2)
    prev_emb_2d = tsne.fit_transform(prev_emb.cpu().detach().numpy())

    # Compute t-SNE transformation for loaded_emb
    loaded_emb_2d = tsne.fit_transform(loaded_emb.cpu().detach().numpy())

    # Plot prev_emb
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_single_emb_2d(prev_emb_2d, itos, axs[0], title='Previous Weights')

    # Plot loaded_emb
    plot_single_emb_2d(loaded_emb_2d, itos, axs[1], title='Loaded Weights')

    # Show plot
    st.pyplot(fig)

def plot_single_emb_2d(emb_2d, itos, ax, title):
    for i in range(len(itos)):
        x, y = emb_2d[i]
        if itos[i] in 'aeiou':
            color = 'r'
        elif itos[i] in 'AEIOU':
            color = 'b'
        elif itos[i] in 'bcdfghjklmnpqrstvwxyz':
            color = 'g'
        elif itos[i] in 'bcdfghjklmnpqrstvwxyz'.upper():
            color = 'orange'
        else:
            continue
        ax.scatter(x, y, color=color)
        ax.text(x + 0.05, y + 0.05, itos[i])
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

# Main function for Streamlit app
def main():
    st.title("Next Character Predictor")
    seed = st.number_input("Enter Seed:",step = 1)
    torch.manual_seed(seed)
    # Load model
    class NextChar(nn.Module):
      def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size*8)
        self.lin2 = nn.Linear(hidden_size*8, hidden_size*4)
        self.lin3 = nn.Linear(hidden_size*4, hidden_size*2)
        self.lin4 = nn.Linear(hidden_size*2, hidden_size)
        self.lin5 = nn.Linear(hidden_size, vocab_size)

      def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.sin(self.lin1(x))
        x = torch.sin(self.lin2(x))
        x = torch.sin(self.lin3(x))
        x = torch.sin(self.lin4(x))
        x = self.lin5(x)
        return x
    switch_state = st.checkbox("Vary parameters")
    if (switch_state):
      block_size = st.select_slider("Select Block Size:",options = [3,4,5,6])
      vocab_size = 256
      if block_size == 5:
        h_options = [20,40,60,80]
        hidden_size = st.select_slider("Select hidden layer size:", options = h_options)
      else:
        hidden_size = 20
        st.text("Hidden size = 20")
      if hidden_size == 20:
        if block_size == 5:
          e_options = [3, 5, 8, 12]
          emb_dim = st.select_slider("Select Embedding dimensions:", options = e_options)
        else:
          emb_dim = 3
          st.text("Embedding Dimension = 3")
      else:
        emb_dim = 5
        st.text("Embedding Dimension = 5")
    else:
      block_size = 5
      vocab_size = 256
      hidden_size = 80
      emb_dim = 20
    your_model = NextChar(block_size, vocab_size, emb_dim, hidden_size).to(device)  # Make sure to initialize your model class
    # Load the state dictionary
    weights_prev = your_model.emb.weight
    state_dict = torch.load(f'model_{block_size}_{emb_dim}_{hidden_size}.pth',map_location = device)

    # Adjust keys if necessary
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        adjusted_key = key.replace('_orig_mod.', '')  # Adjust the keys as needed
        adjusted_state_dict[adjusted_key] = value

    # Load adjusted state dictionary
    your_model.load_state_dict(adjusted_state_dict)
    loaded_weight = your_model.emb.weight
    # Input box for user to enter text
    input_string = st.text_input("Enter your text:", "")

    # Input box for user to enter k
    k = st.number_input("Enter k:", value=1, step=1)

    # Button to trigger prediction
    if st.button("Predict"):
        if input_string:
            predicted_chars = predict_next_chars(input_string, k, your_model, block_size)
            st.write("Predicted next {} characters:\n".format(k))
            st.write(input_string+predicted_chars)

    if st.button("Visualize Embeddings"):
        itos = {i:chr(i) for i in range(256)}
        plot_emb_2d(weights_prev, loaded_weight, itos)


# Run the app
if __name__ == "__main__":
    main()
