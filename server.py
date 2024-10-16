import os
import flwr as fl

# Define strategy for Federated Averaging
strategy = fl.server.strategy.FedAvg()

if __name__ == "__main__":
    # Get the port from the environment variable, defaulting to 8080 if not set
    port = os.getenv("PORT", 0)
    server_address = f"35.164.95.156:{port}"
    print(f"Starting global server at {server_address}...")

    # Start the Flower server on the specified port
    fl.server.start_server(server_address=server_address, strategy=strategy)
