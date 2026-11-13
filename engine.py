import math
import json

class Value:
    """Stores a scalar value and its gradient for backpropagation."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        return self.__relu()

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        return other * (self ** -1.0)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Module:
    """Base class for all neural network modules."""
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def train(self, mode: bool = True):
        self.training = mode
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self) -> list:
        return []

    def named_parameters(self, prefix=''):
        for name, value in self.__dict__.items():
            if isinstance(value, Value):
                yield prefix + name, value
            elif hasattr(value, 'data') and isinstance(value.data, list):
                for i, row in enumerate(value.data):
                    if isinstance(row, list):
                        for j, cell in enumerate(row):
                            if isinstance(cell, Value):
                                yield f"{prefix}{name}.{i}.{j}", cell
                    elif isinstance(row, Value):
                        yield f"{prefix}{name}.{i}", row
            elif isinstance(value, Module):
                yield from value.named_parameters(prefix + name + '.')
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")
                    elif hasattr(item, 'data') and isinstance(item.data, list):
                        for r_idx, row in enumerate(item.data):
                            if isinstance(row, list):
                                for c_idx, cell in enumerate(row):
                                    if isinstance(cell, Value):
                                        yield f"{prefix}{name}.{i}.{r_idx}.{c_idx}", cell

    def state_dict(self) -> dict:
        return {name: p.data for name, p in self.named_parameters()}

    def load_state_dict(self, state_dict: dict):
        param_map = dict(self.named_parameters())
        for name, data_val in state_dict.items():
            if name in param_map:
                param_map[name].data = data_val
                param_map[name].grad = 0.0
            else:
                raise KeyError(f"Unexpected parameter name '{name}' found in state_dict.")

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.state_dict(), f, indent=4)
        print(f"Model successfully saved to {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        self.load_state_dict(state_dict)
        print(f"Model successfully loaded from {filepath}")

    def summary(self, input_shape: tuple):
        print("=" * 70)
        print(f"{'Layer (Type)':<25} {'Output Shape':<25} {'Param #':<15}")
        print("=" * 70)

        total_params = 0
        current_shape = input_shape

        if hasattr(self, 'layers'):
            for i, layer in enumerate(self.layers):
                layer_type = layer.__class__.__name__
                if hasattr(layer, 'W') and hasattr(layer, 'b'):
                    out_features = len(layer.b.data)
                    current_shape = (current_shape[0], out_features)
                    num_params = (len(layer.W.data) * len(layer.W.data[0])) + len(layer.b.data)
                else:
                    num_params = len(layer.parameters())

                total_params += num_params
                print(f"{f'{layer_type}_{i+1}':<25} {str(current_shape):<25} {num_params:<15}")

                if hasattr(self, 'activations') and i < len(self.activations):
                    act_name = self.activations[i]
                    if act_name != 'linear':
                        print(f"{f'Activation ({act_name})':<25} {str(current_shape):<25} {0:<15}")

        print("=" * 70)
        print(f"Total Trainable Parameters: {total_params:,}")
        print("=" * 70)
