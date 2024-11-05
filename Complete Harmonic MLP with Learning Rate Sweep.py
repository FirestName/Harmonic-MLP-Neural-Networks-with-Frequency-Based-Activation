import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class HarmonicLayer(nn.Module):
    def __init__(self, input_size, output_size, base_freq=1.0, max_freq=10.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
        # Create harmonic series frequencies for neurons, but cap the maximum frequency
        self.frequencies = torch.tensor([
            min(base_freq * (i + 1), max_freq) for i in range(output_size)
        ], dtype=torch.float32)
        
        # Initialize weights using a frequency-aware scheme
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

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save the initial state
        self.best_weights = deepcopy(model.state_dict())
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, smooth_f=0.05):
        lrs = []
        losses = []
        best_loss = float('inf')
        
        # Set initial learning rate
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] = start_lr
            
        # Calculate multiplication factor
        lr_factor = (end_lr / start_lr) ** (1 / num_iter)
        
        running_loss = None
        iter_num = 0
        
        for inputs, labels in train_loader:
            if iter_num > num_iter:
                break
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Smooth out the loss
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * (1 - smooth_f) + loss.item() * smooth_f
            
            # Store values
            lrs.append(start_lr * (lr_factor ** iter_num))
            losses.append(running_loss)
            
            # Update best loss and save model if loss is getting too high
            if running_loss < best_loss:
                best_loss = running_loss
            
            if running_loss > 4 * best_loss:
                break
            
            # Update learning rate
            for param_group in self.optimizer.optimizer.param_groups:
                param_group['lr'] *= lr_factor
                
            iter_num += 1
        
        # Restore the best weights
        self.model.load_state_dict(self.best_weights)
        return lrs, losses

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

def plot_lr_finder(lrs, losses):
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.show()

def train_with_lr(model, train_loader, test_loader, criterion, lr, device, epochs=5):
    print(f"\nTraining with learning rate: {lr:.2e}")
    
    # Initialize optimizer with the current learning rate
    base_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    freq_optimizer = FrequencyAwareOptimizer(base_optimizer, model, freq_scale_factor=0.1)
    
    best_acc = 0
    results = []
    
    # Save initial weights to restore later
    initial_weights = deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, freq_optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        results.append({
            'lr': lr,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # Restore initial weights
    model.load_state_dict(initial_weights)
    return best_acc, results

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
    
    # Initialize model and criterion
    model = HarmonicMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # First, run LR finder
    print("Running learning rate finder...")
    base_optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-5)
    freq_optimizer = FrequencyAwareOptimizer(base_optimizer, model, freq_scale_factor=0.1)
    lr_finder = LRFinder(model, freq_optimizer, criterion, device)
    
    lrs, losses = lr_finder.range_test(train_loader)
    plot_lr_finder(lrs, losses)
    
    # Based on LR finder results, test a range of learning rates
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    results = []
    best_lr = None
    best_overall_acc = 0
    
    for lr in learning_rates:
        acc, epoch_results = train_with_lr(model, train_loader, test_loader, 
                                         criterion, lr, device, epochs=5)
        results.extend(epoch_results)
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_lr = lr
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for lr in learning_rates:
        lr_results = [r for r in results if r['lr'] == lr]
        plt.plot([r['epoch'] for r in lr_results], 
                [r['test_acc'] for r in lr_results], 
                label=f'LR: {lr:.2e}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epoch for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nBest learning rate found: {best_lr:.2e} with accuracy: {best_overall_acc:.2f}%")

if __name__ == '__main__':
    main()
