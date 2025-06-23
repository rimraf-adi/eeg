import itertools
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import random

# --------------- Step 1: Preprocess EEG Data -------------------

eeg_idx = [1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
           36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
           76, 77, 78, 79]

train_idx = random.sample(eeg_idx, int(len(eeg_idx) * 0.75))
test_idx = [i for i in eeg_idx if i not in train_idx]

train_frames = []
test_frames = []

for i in train_idx:
    print(f'[TRAIN] Processing file {i}')
    sfilename = f'./features/eeg{i}_seizure_features.csv'
    nsfilename = f'./features/eeg{i}_non_seizure_features.csv'
    sdf = pd.read_csv(sfilename)
    nsdf = pd.read_csv(nsfilename)
    sdf['label'] = 1
    nsdf['label'] = 0
    train_frames.append(sdf)
    train_frames.append(nsdf)

train_df = pd.concat(train_frames, ignore_index=True)

for i in test_idx:
    print(f'[TEST] Processing file {i}')
    sfilename = f'./features/eeg{i}_seizure_features.csv'
    nsfilename =f'./features/eeg{i}_non_seizure_features.csv'
    sdf = pd.read_csv(sfilename)
    nsdf = pd.read_csv(nsfilename)
    sdf['label'] = 1
    nsdf['label'] = 0
    test_frames.append(sdf)
    test_frames.append(nsdf)

test_df = pd.concat(test_frames, ignore_index=True)

# --------------- Step 2: Define Model -------------------

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- Step 3: Train + Store Confusion Matrices -------------------

bands = ['alpha', 'beta', 'gamma', 'delta', 'theta']

conf_matrices = []
labels_for_plots = []

for r in range(1, len(bands)+1):
    for subset in itertools.combinations(bands, r):
        # Only use slope features
        features = [f"midband_{band}" for band in subset]

        # Prepare data subset
        train_subset = train_df[features + ['label']].copy()
        test_subset = test_df[features + ['label']].copy()

        # Replace inf and impute median
        train_subset[features] = train_subset[features].replace([np.inf, -np.inf], np.nan)
        test_subset[features] = test_subset[features].replace([np.inf, -np.inf], np.nan)

        imputer = SimpleImputer(strategy='median')
        train_subset[features] = imputer.fit_transform(train_subset[features])
        test_subset[features] = imputer.transform(test_subset[features])

        X_train = train_subset[features].astype('float32').values
        X_test = test_subset[features].astype('float32').values

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_subset['label'])
        y_test = label_encoder.transform(test_subset['label'])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nTraining subset: {subset}")
        print(f"Features count: {len(features)}")
        print(f"Train size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")
        print(f"Train class distribution: {Counter(y_train)}")
        print(f"Test class distribution: {Counter(y_test)}")

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"Class weights: {class_weights_tensor.cpu().numpy()}")

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

        model = NeuralNet(X_train_tensor.shape[1]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience = 20
        epochs_no_improve = 0

        for epoch in range(100):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_loss += loss.item() * xb.size(0)

            avg_val_loss = val_loss / len(test_loader.dataset)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(torch.load("best_model.pt"))
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        cm = confusion_matrix(all_true, all_preds)
        conf_matrices.append(cm)
        labels_for_plots.append('+'.join(subset))

        print(f"Completed training on subset: {subset}, Best val loss: {best_val_loss:.4f}")

# Plot all confusion matrices after training all subsets

num_plots = len(conf_matrices)
cols = 3
rows = (num_plots + cols - 1) // cols
plt.figure(figsize=(cols*5, rows*4))

for i, (cm, label) in enumerate(zip(conf_matrices, labels_for_plots), 1):
    plt.subplot(rows, cols, i)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
    plt.title(label)
    plt.tight_layout()

plt.show()
accuracies = []
for cm in conf_matrices:
    correct = np.trace(cm)          # Sum of diagonal elements (true positives + true negatives)
    total = np.sum(cm)              # Total samples
    accuracy = correct / total if total > 0 else 0
    accuracies.append(accuracy)

# Create DataFrame
df = pd.DataFrame({
    'Label': labels_for_plots,
    'Accuracy': accuracies
})

print(df)