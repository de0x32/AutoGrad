import random
import math
from engine import Value
from tensor_nn import Tensor, MLP, Adam

def generate_dataset(num_samples=150):
    X_data, Y_data = [], []
    for _ in range(num_samples):
        x1 = random.uniform(-2.0, 2.0)
        x2 = random.uniform(-2.0, 2.0)
        y = math.sin(x1) + math.cos(x2)
        X_data.append([x1, x2])
        Y_data.append([y])
    return Tensor(X_data), Tensor(Y_data)

def calculate_metrics(preds, targets):
    p_vals = [p.data if hasattr(p, 'data') else p for p in preds]
    t_vals = [t.data if hasattr(t, 'data') else t for t in targets]
    
    n = len(p_vals)
    mae = sum(abs(t - p) for t, p in zip(t_vals, p_vals)) / n
    
    mean_target = sum(t_vals) / n
    rss = sum((t - p) ** 2 for t, p in zip(t_vals, p_vals))
    tss = sum((t - mean_target) ** 2 for t in t_vals)
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    
    return mae, r2

def main():
    X_full, Y_full = generate_dataset(150)
    num_samples = X_full.shape[0]

    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(0.8 * num_samples)

    X_train, Y_train = X_full[indices[:split_idx]], Y_full[indices[:split_idx]]
    X_val, Y_val = X_full[indices[split_idx:]], Y_full[indices[split_idx:]]

    model = MLP(nin=2, nouts=[16, 8, 1], activations=['tanh', 'tanh', 'linear'], dropout_p=0.0, use_bn=True)
    model.summary(input_shape=(32, 2))

    initial_lr = 0.03
    optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    epochs = 200
    batch_size = 16
    decay_rate = 0.98
    patience = 20
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        current_lr = initial_lr * (decay_rate ** epoch)
        optimizer.lr = current_lr

        model.train()
        model.zero_grad()
        train_idxs = list(range(X_train.shape[0]))
        random.shuffle(train_idxs)
        
        num_batches = math.ceil(len(train_idxs) / batch_size)
        train_loss_total = 0.0

        for b in range(num_batches):
            b_idx = train_idxs[b * batch_size : (b + 1) * batch_size]
            xb, yb = X_train[b_idx], Y_train[b_idx]

            preds = model(xb)
            diff = preds - yb
            batch_error = sum([(diff[i][0] * diff[i][0]) for i in range(diff.rows)], Value(0.0))
            loss = batch_error * (1.0 / diff.rows)

            loss.backward()
            optimizer.step()
            train_loss_total += loss.data

        avg_train_loss = train_loss_total / num_batches

        model.eval()
        val_preds = model(X_val)
        val_diff = val_preds - Y_val
        val_error = sum([(val_diff[i][0] * val_diff[i][0]) for i in range(val_diff.rows)], Value(0.0))
        val_loss = (val_error * (1.0 / val_diff.rows)).data

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.5f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nRestored best model with Validation Loss: {best_val_loss:.6f}")

    model.eval()
    final_preds = model(X_val)
    preds_list = [final_preds[i][0] for i in range(final_preds.rows)]
    targets_list = [Y_val[i][0] for i in range(Y_val.rows)]

    mae, r2 = calculate_metrics(preds_list, targets_list)
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R²:  {r2:.4f}")

    model.save("best_mlp_model.json")

if __name__ == "__main__":
    main()
