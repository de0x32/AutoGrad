import random
from engine import Value, Module

class Tensor:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = [[(cell if isinstance(cell, Value) else Value(cell)) for cell in row] for row in data]
        else:
            raise TypeError("Tensor data must be a 2D list.")
        
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0

    @property
    def shape(self):
        return (self.rows, self.cols)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Tensor(self.data[idx])
        elif isinstance(idx, list):
            return Tensor([self.data[i] for i in idx])
        return self.data[idx]

    def __matmul__(self, other):
        assert self.cols == other.rows, f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}"
        res = [[Value(0.0) for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                s = Value(0.0)
                for k in range(self.cols):
                    s = s + (self.data[i][k] * other.data[k][j])
                res[i][j] = s
        return Tensor(res)

    def __add__(self, other):
        if isinstance(other, Tensor):
            assert self.shape == other.shape, f"Shape mismatch for addition: {self.shape} vs {other.shape}"
            res = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        else:
            res = [[self.data[i][j] + other for j in range(self.cols)] for i in range(self.rows)]
        return Tensor(res)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Tensor):
            assert self.shape == other.shape
            res = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        else:
            res = [[self.data[i][j] - other for j in range(self.cols)] for i in range(self.rows)]
        return Tensor(res)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return other - self
        else:
            res = [[other - self.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
            return Tensor(res)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            assert self.shape == other.shape
            res = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        else:
            res = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
        return Tensor(res)

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        return Tensor([[cell.tanh() for cell in row] for row in self.data])

    def relu(self):
        return Tensor([[cell.relu() for cell in row] for row in self.data])

    def sigmoid(self):
        return Tensor([[cell.sigmoid() for cell in row] for row in self.data])

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={[[c.data for c in row] for row in self.data]})"


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        # Xavier / Glorot initialization
        scale = (2.0 / nin) ** 0.5
        w_data = [[random.uniform(-scale, scale) for _ in range(nout)] for _ in range(nin)]
        b_data = [[0.0 for _ in range(nout)]]
        
        self.W = Tensor(w_data)
        self.b = Tensor(b_data)

    def __call__(self, x: Tensor) -> Tensor:
        # (X @ W) + b
        return (x @ self.W) + self.b

    def parameters(self) -> list:
        params = []
        for row in self.W.data:
            params.extend(row)
        for row in self.b.data:
            params.extend(row)
        return params


class Dropout(Module):
    def __init__(self, p: float = 0.2):
        self.p = p
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        res_data = []
        for row in x.data:
            new_row = []
            for val in row:
                if random.random() < keep_prob:
                    new_row.append(val * (1.0 / keep_prob))
                else:
                    new_row.append(Value(0.0))
            res_data.append(new_row)
        return Tensor(res_data)

    def parameters(self) -> list:
        return []


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = Tensor([[1.0 for _ in range(dim)]])
        self.beta = Tensor([[0.0 for _ in range(dim)]])
        self.running_mean = [0.0] * dim
        self.running_var = [1.0] * dim

    def __call__(self, x: Tensor) -> Tensor:
        batch_size = x.rows
        num_features = x.cols

        if self.training:
            col_means = []
            for j in range(num_features):
                col_sum = sum([x.data[i][j] for i in range(batch_size)], Value(0.0))
                col_means.append(col_sum * (1.0 / batch_size))

            col_vars = []
            for j in range(num_features):
                var_sum = sum([(x.data[i][j] - col_means[j]) * (x.data[i][j] - col_means[j]) for i in range(batch_size)], Value(0.0))
                col_vars.append(var_sum * (1.0 / batch_size))

            norm_data = []
            for i in range(batch_size):
                row = []
                for j in range(num_features):
                    inv_std = (col_vars[j] + self.eps) ** -0.5
                    row.append((x.data[i][j] - col_means[j]) * inv_std)
                norm_data.append(row)

            for j in range(num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * col_means[j].data
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * col_vars[j].data
        else:
            norm_data = []
            for i in range(batch_size):
                row = []
                for j in range(num_features):
                    inv_std = (self.running_var[j] + self.eps) ** -0.5
                    row.append((x.data[i][j] - self.running_mean[j]) * inv_std)
                norm_data.append(row)

        res_data = []
        for i in range(batch_size):
            row = []
            for j in range(num_features):
                row.append((self.gamma.data[0][j] * norm_data[i][j]) + self.beta.data[0][j])
            res_data.append(row)
        return Tensor(res_data)

    def parameters(self) -> list:
        params = []
        for row in self.gamma.data:
            params.extend(row)
        for row in self.beta.data:
            params.extend(row)
        return params


class MLP(Module):
    def __init__(self, nin: int, nouts: list, activations: list = None, dropout_p: float = 0.1, use_bn: bool = True):
        sz = [nin] + nouts
        if activations is None:
            activations = ['tanh'] * (len(nouts) - 1) + ['linear']
            
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.use_bn = use_bn
        if self.use_bn:
            self.bns = [BatchNorm1d(nouts[i]) for i in range(len(nouts) - 1)]
        self.dropout = Dropout(p=dropout_p)
        self.activations = activations

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_bn and i < len(self.layers) - 1:
                x = self.bns[i](x)
            
            act = self.activations[i]
            if act == 'tanh':
                x = x.tanh()
            elif act == 'relu':
                x = x.relu()
            elif act == 'sigmoid':
                x = x.sigmoid()
                
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        if self.use_bn:
            for bn in self.bns:
                params.extend(bn.parameters())
        return params


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {id(p): 0.0 for p in parameters}
        self.v = {id(p): 0.0 for p in parameters}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + (self.weight_decay * p.data)

            m_buf = self.beta1 * self.m[id(p)] + (1 - self.beta1) * grad
            v_buf = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (grad ** 2)
            self.m[id(p)] = m_buf
            self.v[id(p)] = v_buf

            m_corrected = m_buf / (1 - self.beta1 ** self.t)
            v_corrected = v_buf / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_corrected / (v_corrected ** 0.5 + self.eps)
