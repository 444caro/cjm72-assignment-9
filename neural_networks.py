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
        self.activation = self._activation()
        self.activation_derivative = self._activation_derivative()
        # define layers and initialize weights
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros((1, output_dim))
    


        self.hidden = None 
        self.grads = None
        # Storage for activations and gradients
        self.activations = {}  # **Added storage for activations for debugging**
        self.gradients = {}  # **Added storage for gradients for visualization**


    def _activation(self):
        if self.activation_fn == 'tanh':
            return lambda x: np.tanh(x)
        elif self.activation_fn == 'relu':
            return lambda x: np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return lambda x: 1/(1+np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self):
        if self.activation_fn == 'tanh':
            return lambda x: 1-np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return lambda x: (x>0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = self._activation
            return lambda x: sigmoid(x)*(1-sigmoid(x))
        else:
            raise ValueError("Unsupported activation function")


    
    def forward(self, X):
        #  forward pass, apply layers to input X, store activations for visualization
        # First layer: Input -> Hidden
        self.z1 = np.dot(X,self.W1)+self.b1
        self.a1 = self.activation(self.z1)
        self.activations['Z1'] = self.z1  
        self.activations['A1'] = self.a1

        # Second layer: Hidden -> Output
        self.z2 = np.dot(self.a1, self.W2)+self.b2
        self.a2 = self.z2
        self.activations['Z2'] = self.z2  
        self.activations['A2'] = self.a2
        return self.a2
   

    def backward(self, X, y):
        m = y.shape[0]  

        # Gradient for output layer
        delta2 = self.a2 - y  # Derivative of cross-entropy loss w.r.t. z2
        dW2 = np.dot(self.a1.T, delta2)/m
        db2 = np.sum(delta2,axis=0,keepdims=True)/m

        # Backpropagate to hidden layer 
        delta1 = np.dot(delta2,self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1)/m
        db1 = np.sum(delta1, axis=0,keepdims=True)/m
        
        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients = {  
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1,
            'delta2': delta2,
            'delta1': delta1,
        }

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)*2-1  # Circular boundary
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
    hidden_features = mlp.activations['A1']
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title("Hidden Layer Features")
    ax_hidden.set_xlim(-2, 2)
    ax_hidden.set_ylim(-2, 2)

    # Decision boundary in input space
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, zz, levels=20, cmap='bwr', alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space Decision Boundary")


    # Visualize the gradients 
    input_layer_size = mlp.W1.shape[0]
    hidden_layer_size = mlp.W1.shape[1]
    output_layer_size = mlp.W2.shape[1]
    nodes_input = np.array([[x, 0.8] for x in np.linspace(0.2, 0.8, input_layer_size)])
    nodes_hidden = np.array([[x, 0.5] for x in np.linspace(0.2, 0.8, hidden_layer_size)])
    nodes_output = np.array([[0.5, 0.2]])  # For single output neuron
    for i, node in enumerate(nodes_input):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='blue'))
        ax_gradient.text(
            node[0], node[1] + 0.05,
            f"Input {i+1}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    for i, node in enumerate(nodes_hidden):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='green'))
        ax_gradient.text(
            node[0], node[1] + 0.05,
            f"Hidden {i+1}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    for i, node in enumerate(nodes_output):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='red'))
        ax_gradient.text(
            node[0], node[1] - 0.05,
            f"Output {i+1}",
            ha='center',
            va='top',
            fontsize=10
        )
    for i in range(input_layer_size):
        for j in range(hidden_layer_size):
            gradient_magnitude = np.abs(mlp.gradients['dW1'][i, j])
            ax_gradient.plot(
                [nodes_input[i, 0], nodes_hidden[j, 0]],
                [nodes_input[i, 1], nodes_hidden[j, 1]],
                'k-',
                linewidth=gradient_magnitude * 100
            )
    for i in range(hidden_layer_size):
        for j in range(output_layer_size):
            gradient_magnitude = np.abs(mlp.gradients['dW2'][i, j])
            ax_gradient.plot(
                [nodes_hidden[i, 0], nodes_output[j, 0]],
                [nodes_hidden[i, 1], nodes_output[j, 1]],
                'k-',
                linewidth=gradient_magnitude * 100
            )
    ax_gradient.axis('off')
    ax_gradient.set_title("Gradient Visualization")


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