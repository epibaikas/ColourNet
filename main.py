import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pickle

import socket
import json

# Convert the value of a weight to an RGB value
def weight_to_colour(weight):
    green = int(weight * (-255 / 2) + (255/2))
    blue = 255 - green
    colour = str(green) + ',0,' + str(blue)
    return colour

# Pack the model parameters into json format
def get_parameters_json(model):
    param_dict = model.state_dict()

    fc1_weight = param_dict['fc1.weight'].reshape((9))
    fc2_weight = param_dict['fc2.weight'].reshape((6))

    fc1_weight_list = []
    fc2_weight_list = []

    for i, weight in enumerate(fc1_weight):
        fc1_weight_list.append(("fc1_weight_" + str(i), weight_to_colour(weight)))
    for i, weight in enumerate(fc2_weight):
        fc2_weight_list.append(("fc2_weight_" + str(i), weight_to_colour(weight)))

    param_list = fc1_weight_list + fc2_weight_list
    param_json = json.dumps(dict(param_list), separators=(',', ':'))
    return param_json

def create_server_request(parameters_json):
    # Create an HTTP POST request to the server to update the colours of the LED strip
    request_str = "POST /json HTTP/1.1\r\n Host:172.20.10.2\r\n Accept:application/json\r\n Content-Type:application/json\r\n " + parameters_json + " \r\n\r\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
    except Exception as e:
        print("Cannot connect to the server:", e)

    # Convert POST request string to bytes
    request_str_bytes = request_str.encode('utf-8')
    sock.send(request_str_bytes)
    server_response = sock.recv(1024)

    # Close socket
    sock.close()

class ColourNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ColourNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    print('ColourNet')

    # Set the address and the port for the web server running on the Raspberry Pi Pico
    HOST = '172.20.10.2'
    PORT = 80

    # Set the random seeds
    torch.manual_seed(42)

    # Load the data
    with open('datasets/iris_data_reduced.pkl', 'rb') as f:
        df = pickle.load(f)

    X = torch.tensor(df[[0, 1, 2]].values, dtype=torch.float32)
    y = torch.tensor(df['Labels'].values, dtype=torch.long)

    # Visualise the data in the 3D
    class_0_idx = torch.nonzero(y == 0)
    class_1_idx = torch.nonzero(y == 1)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(X[class_0_idx, 0], X[class_0_idx, 1], X[class_0_idx, 2], marker='o')
    ax.scatter(X[class_1_idx, 0], X[class_1_idx, 1], X[class_1_idx, 2], marker='^')
    ax.set_xlabel('$x_{0}$', fontsize=12)
    ax.set_ylabel('$x_{1}$', fontsize=12)
    ax.set_zlabel('$x_{2}$', fontsize=12)
    ax.set_title('3D Scatter Plot', fontsize=14)

    # Create a Tensor dataset and split it to train and test sets
    dataset = TensorDataset(X, y)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

    # Create train and test data loaders
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=True)

    # Define the training hyperparameters
    model = ColourNet(3, 3, 2)
    lr = 7*1e-3
    loss_fun = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # Update the LED strip to show the initial weights
    parameters_json = get_parameters_json(model)
    create_server_request(parameters_json)

    # Show the 3D scatter plot to visualize the data
    plt.show()

    # Training loop
    num_of_epochs = 100
    model.train()
    for epoch in range(0, num_of_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-1, 1)

            running_loss += loss.item()

        print("Epoch %d, loss %4.2f" % (epoch, running_loss))
        parameters_json = get_parameters_json(model)
        create_server_request(parameters_json)
    print('**** Finished Training ****')