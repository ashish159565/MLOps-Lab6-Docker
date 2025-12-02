import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model_store/model.pth"


class WineQualityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train():
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Training on device:", device)

    X_train, X_test, y_train, y_test = load_data()
    input_dim = X_train.shape[1]

    model = WineQualityNet(input_dim).to(device)

    epochs = 50
    lr = 0.0008
    batch_size = 32

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_tensor).squeeze()
        loss = criterion(preds, y_train_tensor)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # Save model to shared volume
    os.makedirs("model_store", exist_ok=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)

    print(f">>> Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
