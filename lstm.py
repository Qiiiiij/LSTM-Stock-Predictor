import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, output_size=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def load_and_preprocess_data():
    try:
        df = pd.read_csv('易明医药_002826_2024-06-01_至_2025-05-31.csv')
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        numeric_columns = ['开盘价(元)', '收盘价(元)']
        for col in numeric_columns:
            if df[col].isnull().any():
                # 时序数据使用前向/后向填充，避免均值填充污染序列
                df[col] = df[col].ffill().bfill()
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_model(model, train_loader, criterion, optimizer, num_epochs=300):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')
    return train_losses


def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    return np.array(predictions), np.array(actuals)


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def main():
    df = load_and_preprocess_data()
    if df is None:
        return

    features = ['开盘价(元)', '收盘价(元)']
    data = df[features].values

    # 先切分再归一化：Scaler 只拟合训练集，避免测试集信息泄漏
    train_size = int(len(data) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(data[:train_size])
    test_data = scaler.transform(data[train_size:])

    seq_length = 15

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=2, hidden_size=128, num_layers=3, output_size=2, dropout=0.1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=300)
    print("训练完成。")

    # 保存模型权重，下次可直接加载，无需重复训练
    torch.save(model.state_dict(), 'lstm_model.pt')
    print("模型权重已保存到 'lstm_model.pt'。")

    # 训练集评估
    train_predictions, train_actuals = evaluate_model(model, train_loader)
    train_predictions = scaler.inverse_transform(train_predictions)
    train_actuals = scaler.inverse_transform(train_actuals)
    train_rmse, train_mae = calc_metrics(train_actuals, train_predictions)

    # 测试集评估
    test_predictions, test_actuals = evaluate_model(model, test_loader)
    test_predictions = scaler.inverse_transform(test_predictions)
    test_actuals = scaler.inverse_transform(test_actuals)
    test_rmse, test_mae = calc_metrics(test_actuals, test_predictions)

    print(f"训练集 RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # 未来30个交易日预测 (2025年6月)
    model.eval()
    with torch.no_grad():
        scaled_full = np.vstack([train_data, test_data])
        input_seq = scaled_full[-seq_length:].copy()
        future_preds_scaled = []
        for _ in range(30):
            inp = torch.FloatTensor(input_seq.reshape(1, seq_length, -1)).to(device)
            pred = model(inp).cpu().numpy()[0]
            future_preds_scaled.append(pred)
            input_seq = np.vstack([input_seq, pred])[1:]

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled))

    last_date = df['日期'].iloc[-1]
    # 使用工作日(freq='B')近似交易日，避免把周末当作交易日
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq='B')

    future_df = pd.DataFrame({
        '日期': future_dates,
        '预测开盘价(元)': future_preds[:, 0],
        '预测收盘价(元)': future_preds[:, 1]
    })

    print("\n2025年6月预测结果：")
    print(future_df.round(2))

    future_df.to_csv('2025年6月股票预测结果.csv', index=False, encoding='utf-8-sig')
    print("未来30天预测结果已保存到 '2025年6月股票预测结果.csv' 文件中。")

    # 可视化结果
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['日期'], df['收盘价(元)'], label='历史收盘价', color='blue')
    plt.plot(future_df['日期'], future_df['预测收盘价(元)'], label='预测收盘价', linestyle='--', color='red')
    plt.title('易明医药（002826）2025年6月收盘价预测')
    plt.xlabel('日期')
    plt.ylabel('价格(元)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df['日期'], df['开盘价(元)'], label='历史开盘价', color='green')
    plt.plot(future_df['日期'], future_df['预测开盘价(元)'], label='预测开盘价', linestyle='--', color='orange')
    plt.title('易明医药（002826）2025年6月开盘价预测')
    plt.xlabel('日期')
    plt.ylabel('价格(元)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
