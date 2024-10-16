import socket
import flwr as fl

# Define strategy for Federated Averaging
strategy = fl.server.strategy.FedAvg()

if __name__ == "__main__":
    # Dynamically find an available port by binding to port 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))  # Bind to port 0 to let the OS choose an available port
        address, port = s.getsockname()  # Get the assigned port from the socket
        s.close()  # Close the socket, we'll use this port in Flower

    server_address = f"0.0.0.0:{port}"
    print(f"Starting global server at {server_address}...")

    # Start the Flower server on the dynamically assigned port
    fl.server.start_server(server_address=server_address, strategy=strategy)
