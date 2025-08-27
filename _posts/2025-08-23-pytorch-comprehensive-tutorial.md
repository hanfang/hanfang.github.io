---
title: 'PyTorch Basics & Tutorial'
date: 2025-08-23
permalink: /posts/2025/08/pytorch-comprehensive-tutorial/
tags:
  - machine learning
  - pytorch
  - deep learning
  - tutorial
  - education
citation: 'Fang, H. (2025). "PyTorch Basics & Tutorial." <i>Han Fang</i>. Available at: https://hanfang.info/posts/2025/08/pytorch-comprehensive-tutorial/'
---

I've created a comprehensive PyTorch tutorial that takes you from basic tensor operations to advanced topics like attention mechanisms and mixed precision training. This hands-on guide includes real code examples and practical implementations that demonstrate core concepts in modern deep learning.

## Tutorial Overview

The tutorial consists of 5 progressive modules, each building upon the previous concepts with practical code examples you can run and experiment with.

---

## Part 1: Tensor Fundamentals

Understanding tensors is crucial for any PyTorch work. Here's how we start:

```python
import torch
import numpy as np

# Creating tensors - the foundation of PyTorch
tensor_from_data = torch.tensor([1, 2, 3, 4])
tensor_zeros = torch.zeros(2, 3)
tensor_ones = torch.ones(2, 3)
tensor_random = torch.randn(2, 3)  # Normal distribution

# Essential tensor properties
x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")
print(f"Number of elements: {x.numel()}")
```

### Key Broadcasting Concepts

One of the most powerful features in PyTorch:

```python
a = torch.tensor([[1], [2], [3]])  # 3x1
b = torch.tensor([10, 20, 30])     # 1x3
result = a + b  # Broadcasting to 3x3

# Result:
# [[11, 21, 31],
#  [12, 22, 32], 
#  [13, 23, 33]]
```

---

## Part 2: Automatic Differentiation

PyTorch's autograd system is what makes deep learning possible. Here's how gradients flow:

```python
# Basic gradient computation
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

z = x * y + x ** 2
z.backward()

print(f"dz/dx: {x.grad}")  # Should be y + 2*x = 2 + 2*3 = 8
print(f"dz/dy: {y.grad}")  # Should be x = 3
```

### Higher-Order Gradients

For advanced optimization techniques:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx: {dy_dx}")  # 3*2² = 12

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx²: {d2y_dx2}")  # 6*2 = 12
```

---

## Part 3: Neural Network Architecture

Building neural networks with proper PyTorch patterns:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Complete XOR problem solution
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

### Training Loop Pattern

The standard PyTorch training pattern:

```python
model = XORNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
```

---

## Part 4: Practical Implementation Problems

This section tackles real-world implementation challenges you might encounter:

### 1. Numerically Stable Softmax

A common requirement that shows deep understanding:

```python
def softmax_from_scratch(x):
    """Numerically stable softmax implementation"""
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

# Test against PyTorch implementation
x = torch.randn(3, 5)
our_softmax = softmax_from_scratch(x)
pytorch_softmax = F.softmax(x, dim=-1)
print(f"Difference: {torch.max(torch.abs(our_softmax - pytorch_softmax))}")
```

### 2. Custom Dataset Implementation

Essential for real-world applications:

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Usage example
data = torch.randn(100, 10)  # 100 samples, 10 features
targets = torch.randint(0, 2, (100,))  # Binary classification
dataset = CustomDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
    # Your training code here
    pass
```

### 3. Batch Normalization from Scratch

Understanding the internals:

```python
class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_normalized + self.bias
```

---

## Part 5: Advanced Architectures

### Multi-Head Attention Implementation

The backbone of modern transformer architectures:

```python
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """Create padding mask to ignore padded tokens"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Handle both padding masks (True/False) and causal masks (1/0)
            if mask.dtype == torch.bool:
                scores.masked_fill_(~mask, float('-inf'))
            else:
                scores.masked_fill_(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, padding_mask=None, causal_mask=None):
        batch_size, seq_length, d_model = query.size()
        
        # Linear transformations and reshape for multi-head
        Q = self.W_q(query).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
        # Combine masks if both are provided
        combined_mask = None
        if padding_mask is not None:
            combined_mask = padding_mask
        if causal_mask is not None:
            if combined_mask is not None:
                combined_mask = combined_mask & causal_mask
            else:
                combined_mask = causal_mask
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, combined_mask)
        
        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, d_model)
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Usage examples
d_model, n_heads, seq_len, batch_size = 512, 8, 10, 2
attention = MultiHeadAttention(d_model, n_heads)

# Example 1: Self-attention without masks
x = torch.randn(batch_size, seq_len, d_model)
output, weights = attention(x, x, x)

# Example 2: With padding mask (for variable-length sequences)
seq_tokens = torch.randint(1, 1000, (batch_size, seq_len))  # Token IDs
padding_mask = attention.create_padding_mask(seq_tokens, pad_idx=0)
output, weights = attention(x, x, x, padding_mask=padding_mask)

# Example 3: With causal mask (for autoregressive generation)
causal_mask = attention.create_causal_mask(seq_len)
output, weights = attention(x, x, x, causal_mask=causal_mask)

# Example 4: With both masks (common in decoder self-attention)
output, weights = attention(x, x, x, padding_mask=padding_mask, causal_mask=causal_mask)
```

### Custom Optimizer Implementation

Understanding optimization at a fundamental level:

```python
class SGDWithMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        for param, velocity in zip(self.parameters, self.velocities):
            if param.grad is not None:
                velocity.mul_(self.momentum).add_(param.grad, alpha=1)
                param.data.add_(velocity, alpha=-self.lr)
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
```

### Mixed Precision Training

For efficient training on modern GPUs:

```python
# Requires CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    
    # Example training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):  # Example: 10 epochs
        for batch_data, batch_targets in dataloader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(batch_data)
                loss = criterion(output, batch_targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

---

## Getting Started

The complete tutorial is available on GitHub with an interactive runner:

```bash
git clone https://github.com/hanfang/pytorch-practice.git
cd pytorch-practice
pip install torch torchvision numpy matplotlib
python run_tutorial.py
```

The interactive runner lets you:
- Choose specific topics or run the entire curriculum
- Experiment with code examples in real-time
- Track your progress through the learning modules

## Interactive Learning Features

- **Hands-on Examples**: Every concept includes runnable code
- **Progressive Complexity**: From basic tensors to transformer attention
- **Best Practices**: Production-ready patterns and debugging techniques
- **Performance Tips**: Memory optimization and efficient training strategies

---

## Learning Outcomes

After completing this tutorial, you'll master:

1. **Tensor Operations**: Efficient manipulation and broadcasting rules
2. **Automatic Differentiation**: Gradient computation and custom functions  
3. **Neural Architecture**: Building complex models with proper PyTorch patterns
4. **Advanced Techniques**: Attention mechanisms, custom optimizers, mixed precision
5. **Production Skills**: Debugging, profiling, and optimization strategies

Whether you're building research prototypes or production ML systems, this tutorial provides the deep PyTorch knowledge needed for modern deep learning applications.

The combination of theoretical understanding and hands-on implementation has been crucial in my journey from academic research to building large-scale AI systems. I hope this resource accelerates your own PyTorch mastery!

---

## Citation

```
Fang, Han. "PyTorch Basics & Tutorial". Han Fang (Aug 2025). https://hanfang.info/posts/2025/08/pytorch-comprehensive-tutorial/

@article{fang2025pytorch,
    title = {PyTorch Basics \& Tutorial},
    author = {Fang, Han},
    journal = {hanfang.info},
    year = {2025},
    month = {Aug},
    url = "https://hanfang.info/posts/2025/08/pytorch-comprehensive-tutorial/"
}
```