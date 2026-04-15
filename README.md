# Micro-Gradients

Micro-Gradients is a tiny automatic differentiation engine for scalar values, built from scratch in Python.
It demonstrates how reverse-mode autodiff works under the hood by constructing a dynamic computation graph and backpropagating gradients through it.

This project is inspired by Andrej Karpathy's micrograd and is designed for learning, experimentation, and interview-ready understanding of backprop basics.

## Features

- Scalar-based reverse-mode automatic differentiation
- Dynamic computation graph tracking parent nodes and operations
- Core operations: addition, multiplication, power
- Non-linear functions: ReLU, tanh, exp
- Topological-order backward pass for correct gradient propagation
- Minimal implementation in a single readable file

## Project Structure

```
.
|- micro_gradients.py
|- README.md
```

## Requirements

- Python 3.8+
- numpy

## Quick Start

### 1) Clone and enter the repository

```bash
git clone https://github.com/Boules123/Micro-Gradients.git
cd Micro-Gradients
```

### 2) Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install numpy
```

### 4) Run the included example

```bash
python micro_gradients.py
```

Expected output:

```text
a: 21.0, b: 16.0, c: 5.0, d: 6.0, e: 1.0
```

## Usage

### Basic graph and backward pass

```python
from micro_gradients import Value

a = Value(2.0, label="a")
b = Value(3.0, label="b")

c = a * b      # 6
d = a + b      # 5
e = c * d      # 30

e.backward()

print(a.grad)  # 21.0
print(b.grad)  # 16.0
```

### Using activations

```python
from micro_gradients import Value

x = Value(1.5, label="x")
y = (x * x + 2).tanh()

y.backward()

print(y.data)  # tanh(4.25)
print(x.grad)  # dy/dx
```

## How It Works

1. Every `Value` stores:
	- `data`: scalar numeric value
	- `grad`: gradient accumulator
	- `_prev`: parent nodes in the graph
	- `_op`: operation label for debugging
	- `_backward`: local gradient function
2. Mathematical operations create new `Value` nodes and capture local derivative rules in closures.
3. Calling `backward()` on the final output:
	- builds a topological ordering of nodes
	- seeds output gradient with `1.0`
	- executes local backward functions in reverse topological order

## API At a Glance

| Method / Operator | Description |
|---|---|
| `Value(data, label="")` | Create a scalar value node |
| `a + b` | Addition with gradient support |
| `a * b` | Multiplication with gradient support |
| `a ** p` | Power operation (`p` must be int/float) |
| `a.relu()` | ReLU activation |
| `a.tanh()` | tanh activation |
| `a.exp()` | Exponential function |
| `loss.backward()` | Backpropagate gradients from output node |

## Current Limitations

- Scalar-only engine (no tensors)
- No subtraction/division operator overloads yet
- No right-hand operator overloads (`2 + Value(...)`, `2 * Value(...)`)
- No built-in gradient reset helper across graphs
- No automated tests included yet

## Learning Goals

This project is ideal if you want to:

- Understand backprop mechanics at a low level
- Connect calculus derivatives to actual code execution
- Build intuition before using full frameworks like PyTorch or JAX

## Roadmap

- Add subtraction, division, and negation support
- Add right-hand operator overloads (`__radd__`, `__rmul__`)
- Add numerical gradient checking utilities
- Add simple graph visualization support
- Add unit tests and CI

## References

- Micrograd by Andrej Karpathy:
  https://github.com/karpathy/micrograd

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.