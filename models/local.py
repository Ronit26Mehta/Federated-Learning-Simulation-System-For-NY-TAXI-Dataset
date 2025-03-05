import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
def load_data():
    # Load data from a CSV (replace 'nyc_taxi.csv' with your actual file)
    df = pd.read_csv(r"C:\taxi-fed-learn\original_cleaned_nyc_taxi_data_2018.csv")
    # Select features and target
    features = [
        "trip_distance", "rate_code", "fare_amount", "extra", "mta_tax",
        "tip_amount", "tolls_amount", "imp_surcharge", "pickup_location_id", "dropoff_location_id",
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

    return (X_train_1, y_train_1), (X_train_2, y_train_2), (X_val, y_val), df

# PyTorch model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 64)  # 16 input features
        self.fc2 = nn.Linear(64, 1)   # Output predicted value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Federated Client
class FederatedClient:
    def __init__(self, model, train_data, target_data, device="cpu"):
        self.model = model.to(device)
        self.train_data = train_data
        self.target_data = target_data
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def train(self):
        self.model.train()
        inputs = torch.tensor(self.train_data, dtype=torch.float32).to(self.device)
        labels = torch.tensor(self.target_data, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return self.model.state_dict(), loss.item()

# Federated Server
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

    def evaluate(self, val_data, val_targets):
        self.model.eval()
        inputs = torch.tensor(val_data, dtype=torch.float32)
        labels = torch.tensor(val_targets, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(inputs)
            loss = nn.MSELoss()(predictions, labels)
        return loss.item()

if __name__ == "__main__":
    # Load dataset
    (x_train_1, y_train_1), (x_train_2, y_train_2), (x_val, y_val), df = load_data()

    # Initialize model and clients
    global_model = SimpleNN()
    client1 = FederatedClient(SimpleNN(), x_train_1, y_train_1)
    client2 = FederatedClient(SimpleNN(), x_train_2, y_train_2)
    server = FederatedServer(global_model)

    # Simulate federated learning rounds
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}")

        # Train each client and get their model weights and losses
        client1_weights, client1_loss = client1.train()
        client2_weights, client2_loss = client2.train()

        print(f"  Client 1 Loss: {client1_loss:.4f}")
        print(f"  Client 2 Loss: {client2_loss:.4f}")

        # Aggregate the weights on the server
        server.aggregate([client1_weights, client2_weights])

        # Update the clients with the global model weights after aggregation
        client1.model.load_state_dict(server.model.state_dict())
        client2.model.load_state_dict(server.model.state_dict())

        # Evaluate aggregated model on validation data
        val_loss = server.evaluate(x_val, y_val)
        print(f"  Validation Loss (Server): {val_loss:.4f}")

    print("\nFederated learning completed.")
