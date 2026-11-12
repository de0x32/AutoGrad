from engine import MLP, SGD, Value

def main():
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, -1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(nin=3, nouts=[4, 4, 1], activations=['tanh', 'tanh', 'linear'])
    optimizer = SGD(model.parameters(), lr=0.05)

    epochs = 35

    for epoch in range(epochs):
        preds = [model(x) for x in xs]
        loss = sum(((pred - y) ** 2 for y, pred in zip(ys, preds)), Value(0.0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.data:.6f}")

    for x, y in zip(xs, ys):
        pred = model(x)
        print(f"Input: {x} | Target: {y:4.1f} | Prediction: {pred.data:6.4f}")

if __name__ == "__main__":
    main()
