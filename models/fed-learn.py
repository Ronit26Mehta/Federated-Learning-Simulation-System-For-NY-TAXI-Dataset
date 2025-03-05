import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
def load_data():
    # Load data from a CSV (replace 'nyc_taxi.csv' with your actual file)
    df = pd.read_csv(r"C:\taxi-fed-learn\original_cleaned_nyc_taxi_data_2018.csv")

    # Select features and target
    features = [
        "trip_distance", "rate_code", "fare_amount", "extra", "mta_tax",
        "tip_amount", "tolls_amount", "imp_surcharge", "pickup_location", "dropoff_location",
        "year", "month", "day", "day_of_week", "hour_of_day", "trip_duration"
    ]
    target = "calculated_total_amount"

    # Prepare data
    X = df[features].values
    y = df[target].values

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split into train and validation sets
    X_train, X_val = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_val = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    # Simulate client splits
    X_train_1, X_train_2 = np.array_split(X_train, 2)
    y_train_1, y_train_2 = np.array_split(y_train, 2)

    return (X_train_1, y_train_1), (X_train_2, y_train_2), (X_val, y_val)

# Define the PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 64)  # 16 features
        self.fc2 = nn.Linear(64, 1)   # Predict `calculated_total_amount`

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Federated Client class
class FederatedClient:
    def __init__(self, model, data, target, device="cpu"):
        self.model = model.to(device)
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.target = torch.tensor(target, dtype=torch.float32).to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(self.data)
        loss = self.loss_fn(outputs, self.target)
        loss.backward()
        self.optimizer.step()
        return self.model.state_dict()

    def evaluate(self, x_val, y_val):
        self.model.eval()
        x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(x_val)
            loss = self.loss_fn(predictions, y_val)
        return loss.item()

# Federated Server to aggregate model weights (FedAvg)
class FederatedServer:
    def __init__(self, model):
        self.model = model

    def aggregate(self, models_weights):
        # FedAvg: Average the weights of the models
        total_clients = len(models_weights)
        new_weights = {}
        for key in models_weights[0].keys():
            new_weights[key] = torch.mean(torch.stack([weights[key] for weights in models_weights]), dim=0)
        self.model.load_state_dict(new_weights)

# Main function to run federated learning
if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    (x_train_1, y_train_1), (x_train_2, y_train_2), (x_val, y_val) = load_data()

    # Initialize models
    model = SimpleNN().to(device)
    client1 = FederatedClient(model, x_train_1, y_train_1, device)
    client2 = FederatedClient(model, x_train_2, y_train_2, device)
    server = FederatedServer(model)

    # Training loop
    if sys.argv[1] == "server":
        # Aggregate models from clients
        client1_weights = client1.train()
        client2_weights = client2.train()
        server.aggregate([client1_weights, client2_weights])
        print("Server: Aggregated weights and updated global model.")
    
    elif sys.argv[1] == "client1":
        # Train on client1 data
        client1.train()
        print("Client1: Training complete.")
        # Send model weights to server (this would be a network operation in practice)
        client1_weights = client1.train()
        server.aggregate([client1_weights])

    elif sys.argv[1] == "client2":
        # Train on client2 data
        client2.train()
        print("Client2: Training complete.")
        # Send model weights to server (this would be a network operation in practice)
        client2_weights = client2.train()
        server.aggregate([client2_weights])

    # Evaluate on validation set
    val_loss = client1.evaluate(x_val, y_val)
    print(f"Validation loss: {val_loss}")
