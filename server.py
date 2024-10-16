import os
import flwr as fl

# Define strategy for Federated Averaging
strategy = fl.server.strategy.FedAvg()

if __name__ == "__main__":
    # Get host and port from Render's environment variables (if available)
    host = "0.0.0.0"
    port = os.getenv("PORT", "8080")  # Render sets PORT dynamically, default to 8080

    server_address = f"{host}:{port}"
    print(f"Starting global server at {server_address}...")

    # Start the Flower server
    fl.server.start_server(server_address=server_address, strategy=strategy)    
