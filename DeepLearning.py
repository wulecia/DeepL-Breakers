import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)

class DeepLearner:
    def __init__(self, data, labels, vocab_length=0, model_type='LSTM'):
        self.tr_data, self.val_data, tr_labels, val_labels = train_test_split(
            np.array(data), labels, test_size=0.35, stratify=labels
        )

        # One-hot encode labels
        self.tr_labels = self.one_hot(tr_labels)
        self.val_labels = self.one_hot(val_labels)

        # Define vocab length & max sequence length
        self.vocab_length = vocab_length
        self.max_len = max(len(max(self.tr_data, key=len)), len(max(self.val_data, key=len)))

        # Encode corpus
        self.tr_data = self.encode_corpus(self.tr_data)
        self.val_data = self.encode_corpus(self.val_data)

        # Choose model
        if model_type == 'CNN':
            self.model = self.CNN()
        elif model_type == 'CNN_2D':
            self.model = self.CNN_2D()
        elif model_type == 'LSTM':
            self.model = self.LSTM()
        else:
            raise Exception('No such model.')

        # Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.008)

    def one_hot(self, labels):
        encoder = OneHotEncoder(sparse=False)  # ✅ Works in older and newer versions of scikit-learn
        return torch.tensor(encoder.fit_transform(np.array(labels).reshape(-1, 1)), dtype=torch.float32)

    
    def CNN(self):
        class Permute(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1)  # Swap (batch_size, max_len, embedding_dim) -> (batch_size, embedding_dim, max_len)

        model = nn.Sequential(
            nn.Embedding(self.vocab_length, 30, padding_idx=0),  # (batch_size, max_len, 30)
            Permute(),  # Fix shape: (batch_size, 30, max_len)
            nn.Conv1d(30, 64, kernel_size=5, stride=1, padding=2),  # Now correct shape
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten(),
        )
        
        # Test the model output shape before defining Linear layer
        sample_input = torch.randint(0, self.vocab_length, (1, self.max_len))  # Dummy input
        sample_output = model(sample_input)
        flattened_dim = sample_output.shape[1]  # Get the correct number of input features
        
        model.add_module("fc", nn.Linear(flattened_dim, self.tr_labels.shape[1]))  # Adjust input size
        model.add_module("softmax", nn.Softmax(dim=1))
        
        return model

    def LSTM(self):
        class LSTMModel(nn.Module):
            def __init__(self, vocab_length, output_dim, max_len):
                super(LSTMModel, self).__init__()
                self.embedding = nn.Embedding(vocab_length, 30, padding_idx=0)
                self.lstm = nn.LSTM(30, 200, batch_first=True)
                self.fc1 = nn.Linear(200, max_len)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(max_len, output_dim)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.embedding(x)
                lstm_out, _ = self.lstm(x)  # Extract LSTM output
                x = lstm_out[:, -1, :]  # Take the last hidden state
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return self.softmax(x)

        return LSTMModel(self.vocab_length, self.tr_labels.shape[1], self.max_len)



    def encode_corpus(self, data):
        """Convert text data to integer sequences and ensure proper padding/truncation."""
        vectorized_data = [[hash(word) % self.vocab_length for word in d] for d in data]

        # Ensure all sequences are exactly self.max_len
        padded_sequences = [
            seq[:self.max_len] + [0] * max(0, self.max_len - len(seq)) for seq in vectorized_data
        ]

        return torch.tensor(padded_sequences, dtype=torch.long)

    def train(self, epochs=10, batch_size=64):
        dataset = TensorDataset(self.tr_data, self.tr_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_data, batch_labels in loader:
                batch_data, batch_labels = batch_data.to(torch.long), batch_labels.to(torch.float32)  # Ensure correct dtype
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def test(self, tst_data, tst_labels):
        if not isinstance(tst_data, torch.Tensor):
            tst_data = self.encode_corpus(tst_data)
        if not isinstance(tst_labels, torch.Tensor):
            tst_labels = self.one_hot(tst_labels)

        with torch.no_grad():
            outputs = self.model(tst_data)
            loss = self.criterion(outputs, tst_labels)
            accuracy = (outputs.argmax(dim=1) == tst_labels.argmax(dim=1)).float().mean().item()

        print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def test_and_plot(self, tst_data, tst_labels, class_num=2):
        if not isinstance(tst_data, torch.Tensor):
            tst_data = self.encode_corpus(tst_data)
        if not isinstance(tst_labels, torch.Tensor):
            tst_labels = self.one_hot(tst_labels)

        predicted_tst_labels = self.model(tst_data).detach().numpy()

        conf = np.zeros([class_num, class_num])
        confnorm = np.zeros([class_num, class_num])
        for i in range(tst_data.shape[0]):
            j = np.argmax(tst_labels[i, :])
            k = np.argmax(predicted_tst_labels[i])
            conf[j, k] += 1
        for i in range(class_num):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

        self._confusion_matrix(confnorm, labels=[i for i in range(class_num)])
        return self.test(tst_data, tst_labels)

    def _confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.arange(len(labels)), labels, rotation=45)
        plt.yticks(np.arange(len(labels)), labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
