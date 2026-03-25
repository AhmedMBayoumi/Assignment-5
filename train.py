import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import os

class SimpleMLP(nn.Module): 
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("MNIST Training")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "3-layer MLP")
        mlflow.log_param("dataset", "MNIST")
        
        epochs = 10
        mlflow.log_param("epochs", epochs)
        
        model.train()
        for epoch in range(epochs):
            correct = 0
            total = 0
            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

            accuracy = correct / total
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("loss", running_loss / 100, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.4f}, Loss: {running_loss/100:.4f}")
        
        
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Training complete. Accuracy: {accuracy}")

if __name__ == "__main__":
    train()
