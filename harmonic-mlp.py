import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

class HarmonicLayer(nn.Module):
    def __init__(self, input_size, output_size, base_freq=1.0, max_freq=10.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
        # Create harmonic series frequencies for neurons, but cap the maximum frequency
        # to prevent numerical instability
        self.frequencies = torch.tensor([
            min(base_freq * (i + 1), max_freq) for i in range(output_size)
        ], dtype=torch.float32)
        
        # Initialize weights using a frequency-aware scheme
        # Higher frequencies get smaller initial weights
        with torch.no_grad():
            # Compute weight scaling factors
            freq_scale = 1.0 / torch.sqrt(self.frequencies)
            # Normalize the scaling factors
            freq_scale = freq_scale / freq_scale.mean()
            # Xavier/Glorot-like initialization with frequency scaling
            bound = 1 / np.sqrt(input_size)
            self.linear.weight.data.uniform_(-bound, bound)
            self.linear.weight.data *= freq_scale.unsqueeze(1)
            
            # Initialize biases considering frequencies
            self.linear.bias.data.uniform_(-bound, bound)
            self.linear.bias.data *= freq_scale
    
    def forward(self, x):
        x = self.linear(x)
        frequencies = self.frequencies.to(x.device)
        
        # Modified sawtooth activation with smooth transition and frequency scaling
        outputs = []
        for i in range(x.shape[1]):
            # Scale the input based on frequency
            scaled_x = x[:, i] / frequencies[i]
            # Smooth sawtooth-like activation
            activated = torch.sin(2 * np.pi * scaled_x)
            # Add residual connection for better gradient flow
            outputs.append(activated + 0.1 * x[:, i])
        
        return torch.stack(outputs, dim=1)

class HarmonicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_size = 784
        self.hidden_sizes = [512, 256]
        self.num_classes = 10
        
        layers = []
        
        # Input layer with lower max frequency
        layers.append(HarmonicLayer(self.input_size, self.hidden_sizes[0], 
                                  base_freq=1.0, max_freq=5.0))
        
        # Add batch normalization after first layer
        layers.append(nn.BatchNorm1d(self.hidden_sizes[0]))
        layers.append(nn.Dropout(0.2))
        
        # Hidden layer
        layers.append(HarmonicLayer(self.hidden_sizes[0], self.hidden_sizes[1], 
                                  base_freq=1.0, max_freq=3.0))
        layers.append(nn.BatchNorm1d(self.hidden_sizes[1]))
        layers.append(nn.Dropout(0.2))
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Initialize final classifier with appropriate scaling
        self.classifier = nn.Linear(self.hidden_sizes[-1], self.num_classes)
        bound = 1 / np.sqrt(self.hidden_sizes[-1])
        self.classifier.weight.data.uniform_(-bound, bound)
        self.classifier.bias.data.uniform_(-bound, bound)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_layers(x)
        return self.classifier(x)

class FrequencyAwareOptimizer:
    def __init__(self, optimizer, model, freq_scale_factor=0.1):
        self.optimizer = optimizer
        self.model = model
        self.freq_scale_factor = freq_scale_factor
    
    def step(self):
        # Scale gradients based on frequencies before optimizer step
        with torch.no_grad():
            for layer in self.model.modules():
                if isinstance(layer, HarmonicLayer):
                    freq_scale = 1.0 / (1.0 + self.freq_scale_factor * layer.frequencies)
                    freq_scale = freq_scale.to(layer.linear.weight.device)
                    layer.linear.weight.grad *= freq_scale.unsqueeze(1)
                    if layer.linear.bias is not None:
                        layer.linear.bias.grad *= freq_scale
        
        self.optimizer.step()

def train_epoch(model, train_loader, criterion, freq_optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        freq_optimizer.optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Use frequency-aware optimizer
        freq_optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    
    model = HarmonicMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    freq_optimizer = FrequencyAwareOptimizer(base_optimizer, model, freq_scale_factor=0.1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, 'min', patience=2)
    
    num_epochs = 15
    best_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, freq_optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        
        scheduler.step(train_loss)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'Best Acc: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
