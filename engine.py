import random
import math
from typing import List, Union, Callable

class Value:
    
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = ''):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set = set(_children)
        self._op: str = _op
        self.label: str = label

    def __add__(self, other: Union['Value', float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other: Union['Value', float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other: Union[int, float]) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def exp(self) -> 'Value':
        x = self.data
        out = Value(2.718281828459045 ** x, (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> 'Value':
        x = self.data
        t = (2.718281828459045**(2*x) - 1) / (2.718281828459045**(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self) -> 'Value':
        t = 1.0 / (1.0 + (2.718281828459045 ** (-self.data)))
        out = Value(t, (self,), 'sigmoid')

        def _backward():
            self.grad += t * (1.0 - t) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        
        return out

    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: Union[int, float]) -> 'Value':
        return Value(other) * (self ** -1)

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> 'Value':
        return Value(other) + (-self)

    def __neg__(self) -> 'Value':
        return self * -1

    def __radd__(self, other: Union[int, float]) -> 'Value': return self + other
    def __rmul__(self, other: Union[int, float]) -> 'Value': return self * other

    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self) -> None:
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
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Module:
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
            
    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    
    def __init__(self, nin: int, activation: str = 'tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
        
    def __call__(self, x: List[Value]) -> Value:
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return activation.tanh()
        elif self.activation == 'sigmoid':
            return activation.sigmoid()
        elif self.activation == 'relu':
            return activation.relu()
        return activation # linear
        
    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer(Module):
    
    def __init__(self, nin: int, nout: int, activation: str = 'tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
        
    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    
    def __init__(self, nin: int, nouts: List[int], activations: List[str] = None):
        sz = [nin] + nouts
        if activations is None:
            activations = ['tanh'] * (len(nouts) - 1) + ['linear']
        
        self.layers = [
            Layer(sz[i], sz[i+1], activations[i]) 
            for i in range(len(nouts))
        ]
        
    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:

    def __init__(self, parameters: List[Value], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        
    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.lr * p.grad
