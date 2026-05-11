import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from scipy import interpolate
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 1. 配置区域 =================
INPUT_CSV = 'Input_Params_2400.csv'
PS_DIR = '2400_date'

# 修改输出文件夹，专用于 XGBoost 的延迟截断消融实验
OUTPUT_DIR = 'Training_XGBoost_LateClip'
MODEL_NAME = 'xgboost_blind_lateclip_model.pkl'

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

PS_POINTS = 50

# 物理边界先验知识字典 (单位: kPa)
PHYSICS_BOUNDS = {
    1: {'K': (400, 1200), 'phi': (32, 45), 'c': (0, 2)},
    2: {'K': (200, 800), 'phi': (26, 35), 'c': (0, 20)},
    3: {'K': (50, 200), 'phi': (20, 28), 'c': (5, 25)},
    4: {'K': (200, 600), 'phi': (25, 35), 'c': (40, 120)},
    5: {'K': (50, 250), 'phi': (15, 25), 'c': (0, 15)}
}


# ================= 2. 核心工具函数 =================
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
        y_new = f(x_new)
        vec = y_new / y_max
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
    except:
        return np.zeros(n_points)


def load_ablation_data():
    print(">>> Loading Data (Ablation: Feature Stripped, No Categories)...")
    if not os.path.exists(INPUT_CSV): return None, None, None, None, None
    df = pd.read_csv(INPUT_CSV)

    # 量纲修复
    df['Top_c'] = df['Top_c'] / 1000.0
    df['Bot_c'] = df['Bot_c'] / 1000.0

    scalar_cols = ['Top_H', 'Top_n', 'Bot_n', 'Top_Cc', 'Bot_Cc']
    for col in scalar_cols:
        if col not in df.columns: df[col] = 0
    X_scalars = df[scalar_cols].values

    # ！！！核心修改：不再生成和拼接 top_dummies 和 bot_dummies ！！！
    # 但我们保留真实土壤类别用于最后的“延迟截断”
    Top_Types = df['Top_Soil_Type'].values
    Bot_Types = df['Bot_Soil_Type'].values

    target_cols = ['Top_K', 'Top_phi', 'Top_c', 'Bot_K', 'Bot_phi', 'Bot_c']
    Y = df[target_cols].values

    ids = df['ID'].values
    X_curves = []
    print(f"Processing {len(ids)} P-S curve samples...")
    for rid in tqdm(ids):
        rid = int(rid)
        ps = resample_curve_robust(os.path.join(PS_DIR, f'PS_Curve_{rid}.csv'), 150.0, 2000.0, PS_POINTS)
        X_curves.append(ps)

    # ！！！核心修改：输入特征X 中只有基础标量和沉降曲线 ！！！
    X = np.hstack([X_scalars, np.array(X_curves)])
    X = np.nan_to_num(X, nan=0.0)

    return X, Y, target_cols, Top_Types, Bot_Types


def apply_physics_constraints(Y_pred, Top_Types, Bot_Types):
    Y_constrained = Y_pred.copy()
    for i in range(len(Y_constrained)):
        top_t = Top_Types[i]
        bot_t = Bot_Types[i]
        if top_t in PHYSICS_BOUNDS:
            Y_constrained[i, 0] = np.clip(Y_constrained[i, 0], *PHYSICS_BOUNDS[top_t]['K'])
            Y_constrained[i, 1] = np.clip(Y_constrained[i, 1], *PHYSICS_BOUNDS[top_t]['phi'])
            Y_constrained[i, 2] = np.clip(Y_constrained[i, 2], *PHYSICS_BOUNDS[top_t]['c'])
        if bot_t in PHYSICS_BOUNDS:
            Y_constrained[i, 3] = np.clip(Y_constrained[i, 3], *PHYSICS_BOUNDS[bot_t]['K'])
            Y_constrained[i, 4] = np.clip(Y_constrained[i, 4], *PHYSICS_BOUNDS[bot_t]['phi'])
            Y_constrained[i, 5] = np.clip(Y_constrained[i, 5], *PHYSICS_BOUNDS[bot_t]['c'])
    return Y_constrained


# ================= 3. 统一的可视化出图引擎 =================
def evaluate_and_plot(true_vals, pred_vals, param_names, title_suffix, filename, color='crimson'):
    """
    通用出图函数：输入真实值和预测值，自动计算指标并输出 2x3 散点图
    """
    metrics = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    print(f"\n>>> Evaluation Results ({title_suffix}):")
    print(f"{'Parameter':<10} {'R2 Score':<10} {'MAPE':<10} {'MSE':<12} {'RMSE':<10}")
    print("-" * 55)

    for i, name in enumerate(param_names):
        t_val = true_vals[:, i]
        p_val = pred_vals[:, i]

        r2 = r2_score(t_val, p_val)
        mape = mean_absolute_percentage_error(t_val, p_val)
        mse = mean_squared_error(t_val, p_val)
        rmse = np.sqrt(mse)

        metrics.append({'Param': name, 'R2': r2, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse})
        print(f"{name:<10} {r2:<10.4f} {mape:<10.4f} {mse:<12.2f} {rmse:<10.2f}")

        ax = axes[i]
        # 使用你指定的颜色 (crimson 深红色作为反面教材标记)
        ax.scatter(t_val, p_val, alpha=0.5, s=15, c=color, label='Late Clip Predictions')

        min_v, max_v = min(t_val.min(), p_val.min()), max(t_val.max(), p_val.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit')

        ax.set_title(f"{name} ({title_suffix})\n$R^2$={r2:.3f}, RMSE={rmse:.2f}", fontsize=12)
        ax.set_xlabel("True Value (kPa / degree)", fontsize=10)
        if i % 3 == 0: ax.set_ylabel("Predicted Value", fontsize=10)

        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(plot_path, dpi=300)
    print(f"[OK] Plots saved to: {plot_path}")

    return pd.DataFrame(metrics)


# ================= 4. 主程序 =================
def main():
    X, Y, param_names, Top_Types, Bot_Types = load_ablation_data()
    if X is None: return
    print(f"Data Loaded. Blind Feature Shape (NO Category Dummies): {X.shape}")

    c_indices = [2, 5]
    Y_raw = Y.copy()
    Y_log = Y.copy()
    Y_log[:, c_indices] = np.log1p(Y[:, c_indices])

    indices = np.arange(len(X))
    X_train, X_test, Y_train_log, Y_test_log, idx_train, idx_test = train_test_split(
        X, Y_log, indices, test_size=0.1, random_state=42)
    Y_test_raw = Y_raw[idx_test]

    Test_Top_Types = Top_Types[idx_test]
    Test_Bot_Types = Bot_Types[idx_test]

    # --- 训练纯盲算 XGBoost ---
    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=42, tree_method='hist'
    ))

    print(">>> Training Blind XGBoost...")
    model.fit(X_train, Y_train_log)

    save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    joblib.dump(model, save_path)

    # --- 预测 ---
    Y_pred_log = model.predict(X_test)
    Y_pred_raw = Y_pred_log.copy()
    Y_pred_raw[:, c_indices] = np.expm1(Y_pred_log[:, c_indices])  # 盲算原始预测值

    # --- 延迟截断 (Late Clipping) ---
    print(">>> Applying Delayed Physics Constraints (Late Clipping)...")
    Y_pred_final = apply_physics_constraints(Y_pred_raw, Test_Top_Types, Test_Bot_Types)

    # ================= 5. 对比出图 =================
    # 出图 1：盲猜状态 (Raw Blind)
    metrics_raw = evaluate_and_plot(
        true_vals=Y_test_raw,
        pred_vals=Y_pred_raw,
        param_names=param_names,
        title_suffix="Raw Blind XGBoost",
        filename='Scatter_01_Raw_Blind.png',
        color='gray'  # 灰色表示原始无约束盲猜
    )

    # 出图 2：截断后灾难现场 (Late Clip Constrained)
    metrics_final = evaluate_and_plot(
        true_vals=Y_test_raw,
        pred_vals=Y_pred_final,
        param_names=param_names,
        title_suffix="XGBoost Late Clip Ablation",
        filename='Scatter_02_LateClip_Constrained.png',
        color='crimson'  # 深红色表示不合理截断
    )

    # 汇总保存指标
    metrics_raw['Stage'] = 'Raw_Blind'
    metrics_final['Stage'] = 'Late_Clip'
    pd.concat([metrics_raw, metrics_final]).to_csv(os.path.join(OUTPUT_DIR, 'Ablation_Comparison_Metrics.csv'), index=False)


if __name__ == '__main__':
    main()