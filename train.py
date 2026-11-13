import random
import math
import matplotlib.pyplot as plt
from engine import Value
from tensor_nn import Tensor, MLP, Adam, StandardScaler, load_dataset_from_csv

def plot_predicted_vs_actual(preds, targets):
    p_vals = [p.data if hasattr(p, 'data') else p for p in preds]
    t_vals = [t.data if hasattr(t, 'data') else t for t in targets]
    
    plt.figure(figsize=(7, 7))
    
    plt.scatter(t_vals, p_vals, color='royalblue', edgecolor='k', alpha=0.7, s=70, label='Predictions')
    
    min_val = min(min(t_vals), min(p_vals)) - 0.5
    max_val = max(max(t_vals), max(p_vals)) + 0.5
    
    plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', linewidth=2, label='Ideal ($y = x$)')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel('Actual Values ($y$)', fontsize=12)
    plt.ylabel('Predicted Values ($\hat{y}$)', fontsize=12)
    plt.title('Predicted vs. Actual Values (Regression Performance)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

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
    try:
        X_full, Y_full = load_dataset_from_csv("dataset.csv", has_header=True)
        print(f"Successfully loaded dataset from CSV. Total samples: {X_full.shape[0]}")
    except FileNotFoundError:
        print("dataset.csv not found. Generating synthetic dataset for demonstration...")
        X_data, Y_data = [], []
        for _ in range(150):
            x1 = random.uniform(-2.0, 2.0)
            x2 = random.uniform(-2.0, 2.0)
            y = math.sin(x1) + math.cos(x2)
            X_data.append([x1, x2])
            Y_data.append([y])
        X_full, Y_full = Tensor(X_data), Tensor(Y_data)

    num_samples = X_full.shape[0]

    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(0.8 * num_samples)

    X_train = X_full[indices[:split_idx]]
    Y_train = Y_full[indices[:split_idx]]
    X_val = X_full[indices[split_idx:]]
    Y_val = Y_full[indices[split_idx:]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    in_features = X_train.shape[1]
    model = MLP(nin=in_features, nouts=[16, 8, 1], activations=['tanh', 'tanh', 'linear'], dropout_p=0.0, use_bn=True)
    model.summary(input_shape=(32, in_features))

    initial_lr = 0.03
    optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    epochs = 200
    batch_size = 16
    decay_rate = 0.98
    patience = 20
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    print("\n--- Starting Training Pipeline ---")
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

    plot_predicted_vs_actual(preds_list, targets_list)

    model.save("best_mlp_model.json")

if __name__ == "__main__":
    main()
