import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]]) # Target


np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(1, 2)

weights_hidden_output = np.random.rand(2, 1)
bias_output = np.random.rand(1, 1) 


epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    loss = np.mean((y - final_output) ** 2) 
    error = y - final_output

    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += x.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")



print(f'\n\nFinal Output: \n{final_output}')
print(f'\nRounded Output: \n{final_output.round()}')