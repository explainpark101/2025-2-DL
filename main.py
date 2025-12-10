import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

# 설정
DATA_FILE = 'blackjack_probabilities_106cols_traindata.csv'
MODEL_SAVE_PATH = 'blackjack_model.pth'
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TEST_SIZE = 0.2
RANDOM_SEED = 42

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class BlackjackDataset(Dataset):
    """블랙잭 데이터셋 클래스"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BlackjackModel(nn.Module):
    """
    블랙잭 확률 예측 모델
    
    레이어 구조 선정 이유:
    - 입력: 104개 특성 (rem_* 52개 + curr_* 52개)
    - Hidden1 (256): 입력의 약 2.5배 크기로 충분한 표현력 확보
                     블랙잭의 복잡한 확률 관계를 학습하기 위해 넉넉한 크기 설정
    - Hidden2 (128): 중간 크기로 특징을 압축하면서 중요한 패턴 추출
                     점진적 감소로 정보 손실 최소화
    - Hidden3 (64): 최종 압축 레이어로 핵심 특징만 남김
                    과적합 방지와 일반화 성능 향상
    - Output (2): hit_bust_prob와 stay_dealer_win_prob 2개 확률값 출력
    
    Dropout(0.3) 사용 이유:
    - 과적합 방지를 위해 각 히든 레이어 후에 적용
    - 0.3은 일반적으로 사용되는 값으로, 적절한 정규화 효과 제공
    """
    def __init__(self, input_size=104, hidden1=256, hidden2=128, hidden3=64, output_size=2, dropout=0.3):
        super(BlackjackModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # 확률값이므로 0~1 사이로 제한
        return x


def load_and_preprocess_data(file_path):
    """
    CSV 파일을 로드하고 전처리
    같은 입력 조합에 대해 시뮬레이션 결과의 평균을 계산하여 확률로 변환
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 입력 특성 컬럼 추출 (rem_*와 curr_*)
    input_cols = [col for col in df.columns if col.startswith('rem_') or col.startswith('curr_')]
    
    # 출력 컬럼
    output_cols = ['player_burst', 'dealer_wins']
    
    print(f"Found {len(df)} rows")
    print(f"Input features: {len(input_cols)}")
    
    # 같은 입력 조합에 대해 그룹화하여 확률 계산
    print("Grouping by input features and calculating probabilities...")
    grouped = df.groupby(input_cols)[output_cols].mean().reset_index()
    
    # 확률값으로 변환 (이미 평균이므로 0~1 사이의 확률값)
    X = grouped[input_cols].values
    y = grouped[output_cols].values
    
    # 컬럼명 변경: player_burst -> hit_bust_prob, dealer_wins -> stay_dealer_win_prob
    y_df = pd.DataFrame(y, columns=['hit_bust_prob', 'stay_dealer_win_prob'])
    y = y_df.values
    
    print(f"After grouping: {len(X)} unique scenarios")
    print(f"Output shape: {y.shape}")
    print(f"hit_bust_prob range: [{y[:, 0].min():.4f}, {y[:, 0].max():.4f}]")
    print(f"stay_dealer_win_prob range: [{y[:, 1].min():.4f}, {y[:, 1].max():.4f}]")
    
    return X, y, input_cols


def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    """모델 학습"""
    criterion = nn.MSELoss()  # 확률 예측이므로 MSE 사용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print()
    
    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader):
    """모델 평가"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # 전체 예측과 실제값 합치기
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # MAE 계산
    mae = np.mean(np.abs(predictions - labels))
    
    print(f"\nTest Results:")
    print(f"  MSE Loss: {avg_loss:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"\n  hit_bust_prob:")
    print(f"    MAE: {np.mean(np.abs(predictions[:, 0] - labels[:, 0])):.6f}")
    print(f"  stay_dealer_win_prob:")
    print(f"    MAE: {np.mean(np.abs(predictions[:, 1] - labels[:, 1])):.6f}")
    
    return avg_loss, mae


def main():
    """메인 함수"""
    # 데이터 로드 및 전처리
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found!")
        print("Please run create_training_data.py first to generate the training data.")
        return
    
    X, y, input_cols = load_and_preprocess_data(DATA_FILE)
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Validation/Test 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = BlackjackDataset(X_train, y_train)
    val_dataset = BlackjackDataset(X_val, y_val)
    test_dataset = BlackjackDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 생성
    model = BlackjackModel(input_size=len(input_cols)).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 모델 학습
    print(f"\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # 테스트 평가
    print("\nEvaluating on test set...")
    test_loss, test_mae = evaluate_model(model, test_loader)
    
    # 모델 저장
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_cols': input_cols,
        'scaler': scaler,
        'model_config': {
            'input_size': len(input_cols),
            'hidden1': 256,
            'hidden2': 128,
            'hidden3': 64,
            'output_size': 2,
            'dropout': 0.3
        }
    }, MODEL_SAVE_PATH)
    
    print(f"Model saved successfully!")
    print(f"\nTraining completed!")


if __name__ == "__main__":
    main()
