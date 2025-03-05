# Models Overview

This section provides detailed documentation for all supported models in PyGDA. Our framework offers a comprehensive collection of graph domain adaptation models, built on a flexible and extensible architecture.

### Core Architecture

#### [BaseGDA](BaseGDA.md)
The foundation of PyGDA's model architecture, providing:

- Base class for all graph domain adaptation models
- Core training and inference functionalities
- Standardized interfaces for model customization
- Common utility methods and configurations

### Customization Guide

PyGDA is designed for easy customization and extension. To create your own model:

```python
from pygda.models import BaseGDA

class CustomGDA(BaseGDA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model components
        
    def fit(self, data):
        # Implement your training logic
        pass
        
    def predict(self, data):
        # Implement your inference logic
        return predictions
```

### Key Features

1. **Flexible Base Architecture**

    - Inherit from `BaseGDA` for consistent interface
    - Access to core functionalities and utilities
    - Standardized training and evaluation methods

2. **Easy Training Process**
    
    - Use `fit()` method for model training
    - Support for custom hyperparameters
    - Flexible dataset input handling
    - Built-in optimization utilities

3. **Streamlined Evaluation**

    - Simple `predict()` interface
    - Standardized performance metrics
    - Easy integration with evaluation pipelines

4. **Extensibility**

    - Create custom model architectures
    - Add new training strategies
    - Implement domain-specific features
    - Integrate with existing PyGDA components

### Usage Example

```python
from pygda.models import A2GNN

# Initialize model
model = A2GNN(in_dim=100, hidden_dim=64, num_classes=7)

# Train model
model.fit(train_data)

# Make predictions
predictions = model.predict(test_data)
```

For detailed information about each model, please visit their respective documentation pages linked above.