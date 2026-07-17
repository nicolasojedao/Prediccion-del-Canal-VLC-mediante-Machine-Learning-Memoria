import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



print("Cargando dataset caso XGBoost")


column_names = ['X', 'Y', 'Z', 'Alpha', 'Beta', 'H_LoS', 'H_NLoS', 'H_Scat', 'H0_Total', 'P_rx']
df = pd.read_csv("dataset_vlc_z_variable_10k_FINAL_V6.csv", header=None) 
df.columns = column_names

X_tx, Y_tx, Z_tx = 3.0, 0.5, 4.5
df['Distancia_3D'] = np.sqrt((df['X'] - X_tx)**2 + (df['Y'] - Y_tx)**2 + (df['Z'] - Z_tx)**2)
df['Inv_Dist_Sq'] = 1 / (df['Distancia_3D'] ** 2)
df['Cos_Alpha'] = np.cos(np.radians(df['Alpha']))
df['Sin_Alpha'] = np.sin(np.radians(df['Alpha']))
df['Cos_Beta'] = np.cos(np.radians(df['Beta']))


features = ["X", "Y", "Z", "Alpha", "Beta", "Distancia_3D", "Cos_Alpha", "Sin_Alpha", "Cos_Beta", "Inv_Dist_Sq"]

X = df[features]
y = np.log10(df["H0_Total"])

r2_train_list, r2_test_list, gap_list = [], [], []

print("Ejecutando 30 particiones")




for seed in range(30):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    xgboost_model = xgb.XGBRegressor(
        n_estimators=700,
        max_depth=11,
        learning_rate=0.07,
        min_child_weight=7,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.5,
        reg_lambda=10,
        gamma=0.0,
        tree_method="hist",
        random_state=seed,  
        n_jobs=-1
    )
    
    xgboost_model.fit(X_train_scaled, y_train)
    
    train_score = xgboost_model.score(X_train_scaled, y_train)
    test_score = xgboost_model.score(X_test_scaled, y_test)
    gap = train_score - test_score
    
    r2_train_list.append(train_score)
    r2_test_list.append(test_score)
    gap_list.append(gap)








print("\n" + "="*50)
print("   VALIDACION DE CAPACIDAD DE GENERALIZACION DEL MODELO XGBOOST")
print("="*50)
print(f"R2 Train promedio: {np.mean(r2_train_list):.4f} ± {np.std(r2_train_list):.4f}")
print(f"R2 Test promedio:  {np.mean(r2_test_list):.4f} ± {np.std(r2_test_list):.4f}")
print(f"Gap promedio:      {np.mean(gap_list):.4f} ± {np.std(gap_list):.4f}")
print("-" * 50)
print(f"R2 Test minimo:    {np.min(r2_test_list):.4f}")
print(f"R2 Test maximo:    {np.max(r2_test_list):.4f}")
print("="*50)