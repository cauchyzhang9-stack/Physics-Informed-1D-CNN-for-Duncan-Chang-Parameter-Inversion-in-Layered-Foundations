import os
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ================= 1. 配置区域 =================
INPUT_CSV = 'Input_Params_2400.csv'
PS_DIR = '2400_date'

# 修改了输出文件夹名称，专用于这组消融实验
OUTPUT_DIR = 'Training_Group3_CNN_LateClip'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

PS_POINTS = 50
EPOCHS = 300
BATCH_SIZE = 64
LR = 0.001

PHYSICS_BOUNDS = {
    1: {'K': (400, 1200), 'phi': (32, 45), 'c': (0, 2)},
    2: {'K': (200, 800), 'phi': (26, 35), 'c': (0, 20)},
    3: {'K': (50, 200), 'phi': (20, 28), 'c': (5, 25)},
    4: {'K': (200, 600), 'phi': (25, 35), 'c': (40, 120)},
    5: {'K': (50, 250), 'phi': (15, 25), 'c': (0, 15)}
}


# ================= 2. 数据加载与预处理 (特征剥离) =================
def resample_curve_robust(path, x_max, y_max, n_points):
    try:
        if not os.path.exists(path): return np.zeros(n_points)
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        if data.ndim < 2 or len(data) < 2: return np.zeros(n_points)
        x, y = data[:, 0], data[:, 1]
        if np.all(x == 0) or np.all(np.isnan(y)): return np.zeros(n_points)
        _, idx = np.unique(x, return_index=True)
        x, y = x[idx], y[idx]
        f = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
        x_new = np.linspace(0, x_max, n_points)
        return np.nan_to_num(f(x_new) / y_max, nan=0.0)
    except:
        return np.zeros(n_points)


def load_and_preprocess_data():
    print(">>> Loading Data (Ablation: No Prior Input, Late Clipping Only)...")
    df = pd.read_csv(INPUT_CSV)

    df['Top_c'] = df['Top_c'] / 1000.0
    df['Bot_c'] = df['Bot_c'] / 1000.0

    scalar_cols = ['Top_H', 'Top_n', 'Bot_n', 'Top_Cc', 'Bot_Cc']
    for col in scalar_cols:
        if col not in df.columns: df[col] = 0

    # ！！！核心修改：不再计算和拼接 top_dummies 和 bot_dummies ！！！
    # CNN 的标量输入只包含纯几何/物理指标，没有任何土壤类型信息
    X_scalars = df[scalar_cols].values

    # 但我们仍然需要提取真实的土壤类型，用于最后的强行截断
    Top_Types = df['Top_Soil_Type'].values
    Bot_Types = df['Bot_Soil_Type'].values

    X_curves = []
    print(f"Processing P-S curves...")
    for rid in tqdm(df['ID'].values):
        ps = resample_curve_robust(os.path.join(PS_DIR, f'PS_Curve_{int(rid)}.csv'), 150.0, 2000.0, PS_POINTS)
        X_curves.append(ps)
    X_curves = np.array(X_curves)

    param_names = ['Top_K', 'Top_phi', 'Top_c', 'Bot_K', 'Bot_phi', 'Bot_c']
    Y = df[param_names].values

    return X_curves, X_scalars, Y, param_names, Top_Types, Bot_Types


# ================= 3. 定义双分支 1D-CNN 架构 =================
class PhysicsInformedCNN(nn.Module):
    def __init__(self, num_scalars):
        super(PhysicsInformedCNN, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 12, 64),
            nn.ReLU()
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(num_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x_curve, x_scalar):
        curve_features = self.conv_branch(x_curve)
        scalar_features = self.scalar_branch(x_scalar)
        fused = torch.cat((curve_features, scalar_features), dim=1)
        output = self.fusion_layer(fused)
        return output


# ================= 4. 物理截断与画图引擎 =================
def apply_physics_constraints(Y_pred, Top_Types, Bot_Types):
    Y_constrained = Y_pred.copy()
    for i in range(len(Y_constrained)):
        top_t, bot_t = Top_Types[i], Bot_Types[i]
        if top_t in PHYSICS_BOUNDS:
            Y_constrained[i, 0] = np.clip(Y_constrained[i, 0], *PHYSICS_BOUNDS[top_t]['K'])
            Y_constrained[i, 1] = np.clip(Y_constrained[i, 1], *PHYSICS_BOUNDS[top_t]['phi'])
            Y_constrained[i, 2] = np.clip(Y_constrained[i, 2], *PHYSICS_BOUNDS[top_t]['c'])
        if bot_t in PHYSICS_BOUNDS:
            Y_constrained[i, 3] = np.clip(Y_constrained[i, 3], *PHYSICS_BOUNDS[bot_t]['K'])
            Y_constrained[i, 4] = np.clip(Y_constrained[i, 4], *PHYSICS_BOUNDS[bot_t]['phi'])
            Y_constrained[i, 5] = np.clip(Y_constrained[i, 5], *PHYSICS_BOUNDS[bot_t]['c'])
    return Y_constrained


def evaluate_and_plot(true_vals, pred_vals, param_names, filename):
    metrics = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        t_val, p_val = true_vals[:, i], pred_vals[:, i]
        r2 = r2_score(t_val, p_val)
        rmse = np.sqrt(mean_squared_error(t_val, p_val))
        metrics.append({'Param': name, 'R2': r2, 'RMSE': rmse})

        ax = axes[i]
        # 用深红色 (crimson) 标记这组“反面教材”
        ax.scatter(t_val, p_val, alpha=0.5, s=15, c='crimson', label='Late Clip (No Input Prior)')
        min_v, max_v = min(t_val.min(), p_val.min()), max(t_val.max(), p_val.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit')

        ax.set_title(f"{name} (Late Clip Ablation)\n$R^2$={r2:.3f}, RMSE={rmse:.2f}", fontsize=12)
        ax.set_xlabel("True Value", fontsize=10)
        if i % 3 == 0: ax.set_ylabel("Predicted Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    return pd.DataFrame(metrics)


# ================= 5. 主训练流水线 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_curves, X_scalars, Y, param_names, Top_Types, Bot_Types = load_and_preprocess_data()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scalars_norm = scaler_x.fit_transform(X_scalars)
    Y_norm = scaler_y.fit_transform(Y)

    indices = np.arange(len(Y))
    idx_train, idx_test = train_test_split(indices, test_size=0.1, random_state=42)

    X_c_train = torch.FloatTensor(X_curves[idx_train]).unsqueeze(1).to(device)
    X_s_train = torch.FloatTensor(X_scalars_norm[idx_train]).to(device)
    Y_train = torch.FloatTensor(Y_norm[idx_train]).to(device)

    X_c_test = torch.FloatTensor(X_curves[idx_test]).unsqueeze(1).to(device)
    X_s_test = torch.FloatTensor(X_scalars_norm[idx_test]).to(device)
    Y_test = Y[idx_test]

    # 提取测试集的土壤类型，用于最后的延迟截断
    Test_Top_Types = Top_Types[idx_test]
    Test_Bot_Types = Bot_Types[idx_test]

    train_dataset = TensorDataset(X_c_train, X_s_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PhysicsInformedCNN(num_scalars=X_scalars.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print(f">>> Starting CNN Training (Blind) for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for b_curves, b_scalars, b_y in train_loader:
            optimizer.zero_grad()
            preds = model(b_curves, b_scalars)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_c_test, X_s_test).cpu().numpy()

    # 1. 模型给出极不准确的盲猜
    Y_pred_raw = scaler_y.inverse_transform(preds_norm)

    # 2. 强行进行物理截断
    print(">>> Applying Delayed Physics Constraints (Late Clipping)...")
    Y_pred_final = apply_physics_constraints(Y_pred_raw, Test_Top_Types, Test_Bot_Types)

    print("\n[Final Evaluation for Ablation: Late Clip Only]")
    metrics_df = evaluate_and_plot(Y_test, Y_pred_final, param_names, 'Scatter_Ablation_LateClip.png')
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'Metrics_Ablation_LateClip.csv'), index=False)


if __name__ == '__main__':
    main()