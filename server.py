import flwr as fl

# Define strategy for Federated Averaging
strategy = fl.server.strategy.FedAvg()

# Start the server with the specified strategy
if __name__ == "__main__":
    print("Starting global server...")
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy)
