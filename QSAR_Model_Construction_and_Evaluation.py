import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
# 1. Data Loading
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/ERR.csv', help='Input CSV file')
args = parser.parse_args()
data = pd.read_csv(args.input)
smiles_list = data['smiles'].tolist()
labels = data['label'].tolist()

# 2. Molecular descriptor extraction functions
def compute_ecfp6(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def compute_fcfp6(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True))

def compute_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol))

def compute_rdkit(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = Chem.RDKFingerprint(mol, fpSize=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 3. Batch descriptor calculation
print("Calculating molecular descriptors...")
ecfp6_list = [compute_ecfp6(s) for s in smiles_list]
fcfp6_list = [compute_fcfp6(s) for s in smiles_list]
maccs_list = [compute_maccs(s) for s in smiles_list]
rdkit_list = [compute_rdkit(s) for s in smiles_list]

valid_indices = [i for i, desc in enumerate(ecfp6_list) if desc is not None]
labels = [labels[i] for i in valid_indices]
ecfp6_list = [ecfp6_list[i] for i in valid_indices]
fcfp6_list = [fcfp6_list[i] for i in valid_indices]
maccs_list = [maccs_list[i] for i in valid_indices]
rdkit_list = [rdkit_list[i] for i in valid_indices]

X_ecfp6 = np.array(ecfp6_list)
X_fcfp6 = np.array(fcfp6_list)
X_maccs = np.array(maccs_list)
X_rdkit = np.array(rdkit_list)
y = np.array(labels)
print(f"Valid sample size: {len(y)}")

# 4. Descriptor combinations
descriptor_sets = {
    'ECFP6': X_ecfp6,
    'FCFP6': X_fcfp6,
    'MACCS': X_maccs,
    'RDKit': X_rdkit,
    'ECFP6+FCFP6': np.hstack([X_ecfp6, X_fcfp6]),
    'ECFP6+MACCS': np.hstack([X_ecfp6, X_maccs]),
    'ECFP6+RDKit': np.hstack([X_ecfp6, X_rdkit]),
    'FCFP6+MACCS': np.hstack([X_fcfp6, X_maccs]),
    'FCFP6+RDKit': np.hstack([X_fcfp6, X_rdkit]),
    'MACCS+RDKit': np.hstack([X_maccs, X_rdkit]),
    'ECFP6+FCFP6+MACCS': np.hstack([X_ecfp6, X_fcfp6, X_maccs]),
    'ECFP6+FCFP6+RDKit': np.hstack([X_ecfp6, X_fcfp6, X_rdkit]),
    'ECFP6+MACCS+RDKit': np.hstack([X_ecfp6, X_maccs, X_rdkit]),
    'FCFP6+MACCS+RDKit': np.hstack([X_fcfp6, X_maccs, X_rdkit]),
    'ALL': np.hstack([X_ecfp6, X_fcfp6, X_maccs, X_rdkit])
}

# 5. Machine learning models
ml_models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42)
}

def create_lstm_model(input_shape, num_classes=1):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# 6. 5-fold cross-validation evaluation function
def cross_val_evaluate_ml(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {'Recall': [], 'F1': [], 'Accuracy': [], 'AUC': []}
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        scores['Recall'].append(recall_score(y_test, y_pred))
        scores['F1'].append(f1_score(y_test, y_pred))
        scores['Accuracy'].append(accuracy_score(y_test, y_pred))
        scores['AUC'].append(roc_auc_score(y_test, y_prob))
    return {k: np.mean(v) for k,v in scores.items()}

def cross_val_evaluate_lstm(X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {'Recall': [], 'F1': [], 'Accuracy': [], 'AUC': []}
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Sequence format
        seq_length = min(10, len(X_train) // 10)
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length + 1):
                sequences.append(data[i:i+seq_length])
            return np.array(sequences)
        X_train_seq = create_sequences(X_train, seq_length)
        X_test_seq = create_sequences(X_test[:len(X_test)-len(X_test)%seq_length], seq_length)
        y_train_seq = y_train[seq_length-1:]
        y_test_seq = y_test[seq_length-1:len(y_test)-len(y_test)%seq_length]
        scaler = StandardScaler()
        X_train_seq_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_seq_reshaped).reshape(X_train_seq.shape)
        X_test_seq_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_test_scaled = scaler.transform(X_test_seq_reshaped).reshape(X_test_seq.shape)
        model = create_lstm_model((seq_length, X_train.shape[1]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        model.fit(X_train_scaled, y_train_seq, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        y_prob = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        scores['Recall'].append(recall_score(y_test_seq, y_pred))
        scores['F1'].append(f1_score(y_test_seq, y_pred))
        scores['Accuracy'].append(accuracy_score(y_test_seq, y_pred))
        scores['AUC'].append(roc_auc_score(y_test_seq, y_prob))
    return {k: np.mean(v) for k,v in scores.items()}

# 7. Results saving and visualization
results = []
model_objects = []  # Save model objects for later best model saving
print("Start 5-fold cross-validation model training and evaluation...")
for desc_name, X in descriptor_sets.items():
    print(f"Processing descriptor combination: {desc_name}")
    for model_name, model in ml_models.items():
        scores = cross_val_evaluate_ml(model, X, y, cv=5)
        result = {'Descriptor': desc_name, 'Model': model_name, **scores}
        results.append(result)
        model_objects.append({'Descriptor': desc_name, 'Model': model_name, 'ModelObj': model, 'AUC': scores['AUC'], 'X': X, 'y': y})
        print(f"  {model_name}: AUC={scores['AUC']:.3f}")
    # LSTM
    try:
        scores = cross_val_evaluate_lstm(X, y, cv=5)
        result = {'Descriptor': desc_name, 'Model': 'LSTM', **scores}
        results.append(result)
        # Note: LSTM is not saved as sklearn/xgboost object (save only if needed)
        print(f"  LSTM: AUC={scores['AUC']:.3f}")
    except Exception as e:
        print(f"  LSTM failed: {e}")

df_results = pd.DataFrame(results)
# Find the best model
best_row = df_results.sort_values('AUC', ascending=False).iloc[0]
print(f"\nBest model: Descriptor combination={best_row['Descriptor']}, Model={best_row['Model']}, Mean AUC={best_row['AUC']:.3f}")

# Save the best model to models/best_qsar_model.pkl
if not os.path.exists('models'):
    os.makedirs('models')
for m in model_objects:
    if m['Descriptor'] == best_row['Descriptor'] and m['Model'] == best_row['Model']:
        model = m['ModelObj']
        X_full = m['X']
        y_full = m['y']
        model.fit(X_full, y_full)
        # ---- 修改为带描述符类型信息 ----
        joblib.dump({'model': model, 'descriptor': best_row['Descriptor']}, 'models/best_qsar_model.pkl')
        print("Best model and descriptor info saved to models/best_qsar_model.pkl")
        break

# Visualization (AUC performance of different models/descriptors)
plt.figure(figsize=(15, 10))
sns.barplot(data=df_results, x='Descriptor', y='AUC', hue='Model')
plt.xticks(rotation=45)
plt.title('AUC performance of different descriptor combinations and ML models (5-fold CV)')
plt.tight_layout()
plt.show()