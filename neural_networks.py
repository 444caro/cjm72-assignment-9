import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # define layers and initialize weights
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    
        # Storage for activations and gradients
        self.z1 = None  # Pre-activation of hidden layer
        self.a1 = None  # Post-activation (hidden features)
        self.z2 = None  # Pre-activation of output layer
        self.a2 = None  # Output layer activations (predictions)
        self.grads = {}  # To store gradients for visualization

    def _activation(self, z):
        if self.activation_fn == 'tanh':
            return np.tanh(z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_fn == 'relu':
            return (z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError("Unsupported activation function")


    
    def forward(self, X):
        #  forward pass, apply layers to input X
        #  store activations for visualization
        # Forward pass: Input -> Hidden Layer -> Output Layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activation(self.z1)  # Apply activation function
        
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid for output layer
        
        return self.a2
   

    def backward(self, X, y):
        # Compute gradients for backpropagation
        m = X.shape[0]  # Number of samples
        
        # Gradient for output layer
        dz2 = self.a2 - y  # Derivative of cross-entropy loss w.r.t. z2
        self.grads['W2'] = (self.a1.T @ dz2) / m
        self.grads['b2'] = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradient for hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._activation_derivative(self.z1)
        self.grads['W1'] = (X.T @ dz1) / m
        self.grads['b1'] = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= self.lr * self.grads['W1']
        self.b1 -= self.lr * self.grads['b1']
        self.W2 -= self.lr * self.grads['W2']
        self.b2 -= self.lr * self.grads['b2']

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Space Features")
    ax_hidden.set_xlabel("Neuron 1")
    ax_hidden.set_ylabel("Neuron 2")
    ax_hidden.set_zlabel("Neuron 3")

    # Decision boundary in input space
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    predictions = mlp.forward(grid)
    zz = predictions.reshape(xx1.shape)
    ax_input.contourf(xx1, xx2, zz, levels=[0, 0.5, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Decision Boundary in Input Space")
    ax_input.set_xlabel("X1")
    ax_input.set_ylabel("X2")
    
    # Visualize gradients
    gradient_magnitudes = np.linalg.norm(mlp.grads['W1'], axis=1)
    for i in range(mlp.W1.shape[0]):
        circle = Circle((X[i, 0], X[i, 1]), radius=0.1, edgecolor='black', facecolor='none', lw=gradient_magnitudes[i])
        ax_gradient.add_patch(circle)
    ax_gradient.set_xlim([-2, 2])
    ax_gradient.set_ylim([-2, 2])
    ax_gradient.set_title("Gradients Visualization")
    ax_gradient.set_xlabel("X1")
    ax_gradient.set_ylabel("X2")
    # The edge thickness visually represents the magnitude of the gradient


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)