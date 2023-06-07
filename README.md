# Automatic Optimization of Deep Neural Network Hyperparameters

This repository contains code for automatically optimizing hyperparameters of a deep neural network using a predefined objective function. The optimization process aims to achieve a desired accuracy and loss value.

## Repository Specifications

Repository URL: [Automatic-optimization-of-deep-neural-network-hyperparameters](https://github.com/armansouri9/Automatic-optimization-of-deep-neural-network-hyperparameters)

## Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import configparser
import sys

# Function to check if desired performance is achieved
def is_desired_performance(acc, loss, desired_acc, desired_loss):
    return acc >= desired_acc and loss <= desired_loss

# Read settings from config file
config = configparser.ConfigParser()
config.read('/content/config.ini')

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_neurons):
        super(NeuralNet, self).__init__()
        layers = [nn.Linear(input_size, num_neurons)]
        layers.extend([nn.Linear(num_neurons, num_neurons) for _ in range(num_layers - 2)])
        layers.append(nn.Linear(num_neurons, output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

# Define the training and evaluation function
def train_and_evaluate_model():
    # Define initial settings
    global objective_function 
    objective_function = config['Model']['objective_function']
    
    global num_layers
    num_layers = int(config['Model']['num_layers'])
    
    global num_neurons
    num_neurons = int(config['Model']['num_neurons'])
    
    global num_classes
    num_classes = int(config['Model']['num_classes'])
    
    global input_size 
    input_size= int(config['Model']['input_size'])
    
    global epochs 
    epochs= int(config['Training']['epochs'])
    
    global optimizer_type
    optimizer_type = config['Training']['optimizer_type']
    
    global learning_rate
    learning_rate = float(config['Training']['learning_rate'])
    
    global desired_acc 
    desired_acc= float(config['Training']['desired_accuracy'])
    
    global desired_loss
    desired_loss = float(config['Training']['desired_loss'])

    # Define the model
    model = NeuralNet(input_size, num_classes, num_layers, num_neurons)

    # Define the objective function
    if objective_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Define the optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer type not supported.")

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        # Check if desired performance is achieved
        if is_desired_performance(accuracy, loss.item(), desired_acc, desired_loss):
            break

    return model, accuracy, loss.item()

# Training data and labels
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([0, 1, 1,

 0], dtype=torch.long)

# Initial model training and evaluation
model, accuracy, loss = train_and_evaluate_model()

# Check if desired performance is achieved
if not is_desired_performance(accuracy, loss, desired_acc, desired_loss):
    # Adjust settings to improve results
    config.set('Model', 'num_layers', str(num_layers + 1))
    config.set('Model', 'num_neurons', str(num_neurons + 10))

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    # Restart the code execution
    exec(open(__file__).read())

# Results and changes
print("Accuracy:", accuracy)
print("Loss:", loss)
sys.exit()
```

The code provided demonstrates the automatic optimization of hyperparameters for a deep neural network. It reads the settings from a configuration file, defines the neural network model, trains the model using the specified hyperparameters, and evaluates its performance based on accuracy and loss values.

If the desired performance is not achieved, the code adjusts the hyperparameters to improve the results and restarts the execution.

## Results

Accuracy: 0.5
Loss: 0.6931471824645996

Feel free to explore and modify the code to suit your needs.
## License

This project is licensed under a Free License.

---

Note: The code provided assumes the presence of a `config.ini` file containing the necessary settings. Please make sure to update the file with appropriate values before running the code.




