import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    print(f"🚀 Starting model training at {datetime.now()}")
    print("=" * 70)
    
    try:
        # Load data - CORRECTED PATH
        data_path = '../data/disease_data_cleaned.csv'  # Go up one level, then into data directory
        print(f"📊 Looking for data at: {os.path.abspath(data_path)}")
        
        if not os.path.exists(data_path):
            # Try alternative paths
            alternative_paths = [
                'data/disease_data_cleaned.csv',  # Same directory
                '../disease_data_cleaned.csv',    # Parent directory
                'disease_data_cleaned.csv'        # Current directory
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    print(f"📊 Found data at: {os.path.abspath(data_path)}")
                    break
        
        if not os.path.exists(data_path):
            print(f"❌ ERROR: Could not find 'disease_data_cleaned.csv'")
            print("💡 Please check: The file should be in one of these locations:")
            print("   - 'data/disease_data_cleaned.csv' (relative to train_model.py)")
            print("   - '../data/disease_data_cleaned.csv' (if models/train_model.py)")
            print("   - 'disease_data_cleaned.csv' (in same directory)")
            exit(1)
        
        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset with {len(df)} records")
        print(f"📋 Columns: {list(df.columns)}")
        
        # ==================== DATA CLEANING ====================
        print("\n🔧 DATA CLEANING:")
        
        # 1. Fix Year column - remove decimal .0
        df['Year'] = df['Year'].astype(int)
        print(f"  • Converted Year to integers: {df['Year'].min()} to {df['Year'].max()}")
        
        # 2. Check for and handle NaN values
        print(f"  • NaN values before cleaning:")
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    - {col}: {nan_count} NaN values")
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
        
        # 3. Fix column names with trailing spaces
        df.columns = [col.strip() for col in df.columns]
        
        # ==================== DATA DIAGNOSTICS ====================
        print("\n🔍 DATA DIAGNOSTICS:")
        print(f"Years in data: {sorted(df['Year'].unique())}")
        print(f"Time range: {df['Year'].min()} to {df['Year'].max()}")
        
        # Check data distribution
        print(f"\n📈 Cases by Year:")
        cases_by_year = df.groupby('Year')['Cases'].agg(['mean', 'sum'])
        for year, row in cases_by_year.iterrows():
            print(f"  • {year}: {row['mean']:.1f} avg, {row['sum']} total")
        
        print(f"\n💀 Deaths by Year:")
        if 'Deaths' in df.columns:
            deaths_by_year = df.groupby('Year')['Deaths'].agg(['mean', 'sum'])
            for year, row in deaths_by_year.iterrows():
                print(f"  • {year}: {row['mean']:.1f} avg, {row['sum']} total")
            total_deaths = df['Deaths'].sum()
            print(f"  • Total deaths in dataset: {total_deaths}")
            print(f"  • Percentage of records with deaths: {(df['Deaths'] > 0).mean()*100:.1f}%")
        else:
            print("  • No 'Deaths' column found!")
        
        # ==================== DATA PREPARATION ====================
        train_df = df.copy()
        
        # Encode categorical variables
        le_location = LabelEncoder()
        le_disease = LabelEncoder()
        
        train_df['Location_encoded'] = le_location.fit_transform(train_df['Location'])
        train_df['Disease_encoded'] = le_disease.fit_transform(train_df['Disease'])
        
        # Calculate CFR if not exists or fix existing CFR
        if 'CFR' not in train_df.columns and 'Deaths' in train_df.columns:
            train_df['CFR'] = (train_df['Deaths'] / train_df['Cases'].clip(lower=1)) * 100
            print(f"\n📝 Calculated CFR: {train_df['CFR'].mean():.2f}% average")
        elif 'CFR' in train_df.columns:
            # Fix any infinite or NaN CFR values
            train_df['CFR'] = (train_df['Deaths'] / train_df['Cases'].clip(lower=1)) * 100
            print(f"\n📝 Recalculated CFR: {train_df['CFR'].mean():.2f}% average")
        
        # ==================== FEATURE ENGINEERING ====================
        print("\n🔧 FEATURE ENGINEERING:")
        
        # 1. Time-based features
        train_df['Year_normalized'] = (train_df['Year'] - train_df['Year'].min()) / (train_df['Year'].max() - train_df['Year'].min())
        train_df['Year_squared'] = train_df['Year'] ** 2
        
        # 2. Growth rate features - handle Case Growth Rate column
        if 'Case Growth Rate' in train_df.columns:
            # Use existing column but handle NaN/inf
            train_df['Case_Growth_Rate'] = train_df['Case Growth Rate']
            print(f"  • Using existing Case Growth Rate column")
        else:
            # Calculate growth rate from previous year
            train_df = train_df.sort_values(['Location', 'Disease', 'Year'])
            train_df['Case_Growth_Rate'] = train_df.groupby(['Location', 'Disease'])['Cases'].pct_change()
            print(f"  • Calculated new Case_Growth_Rate")
        
        # Handle NaN/Inf in growth rate
        train_df['Case_Growth_Rate'] = train_df['Case_Growth_Rate'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 3. Rolling averages
        train_df['Cases_Rolling_Avg'] = train_df.groupby(['Location', 'Disease'])['Cases'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean()
        )
        
        # 4. Location-Disease interaction
        train_df['Location_Disease_Interaction'] = train_df['Location_encoded'] * train_df['Disease_encoded']
        
        # 5. Seasonal features (based on temperature)
        train_df['Is_Warm'] = (train_df['Temperature (°C)'] > 25).astype(int)
        train_df['Is_Rainy'] = (train_df['Rainfall (mm)'] > 2000).astype(int)
        
        # 6. Risk categories
        train_df['Risk_Category'] = pd.qcut(train_df['Climate Risk Index'], q=4, labels=[0, 1, 2, 3])
        
        # ==================== PREPARE FEATURES ====================
        # Base features (ensure they exist)
        available_features = []
        
        potential_features = [
            'Year', 'Rainfall (mm)', 'Temperature (°C)', 'Population Density', 
            'Climate Risk Index', 'Case_Growth_Rate', 'Year_normalized', 
            'Year_squared', 'Cases_Rolling_Avg', 'Location_Disease_Interaction',
            'Is_Warm', 'Is_Rainy', 'Risk_Category', 'Location_encoded', 'Disease_encoded'
        ]
        
        # Only use features that exist in dataframe
        for feat in potential_features:
            if feat in train_df.columns:
                available_features.append(feat)
        
        print(f"  • Using {len(available_features)} features: {available_features}")
        
        # Prepare feature matrix
        X = train_df[available_features].copy()
        
        # Final check for missing values
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
                print(f"  • Filled NaN in {col}")
        
        # ==================== CASES PREDICTION MODEL ====================
        print("\n" + "=" * 70)
        print("🤖 TRAINING CASES PREDICTION MODEL")
        print("=" * 70)
        
        y_cases = train_df['Cases'].values
        
        # Scale features
        scaler_cases = StandardScaler()
        X_scaled = scaler_cases.fit_transform(X)
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=min(3, len(np.unique(train_df['Year'])) - 1))
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n📊 Training with cross-validation...")
        
        # Cross-validation for cases model
        cv_scores = []
        cv_mae_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_cases[train_idx], y_cases[val_idx]
            
            # Train Random Forest
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_val)
            
            score = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(score)
            cv_mae_scores.append(mae)
        
        print(f"  • Cross-validation R² scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"  • Cross-validation MAE scores: {[f'{s:.2f}' for s in cv_mae_scores]}")
        print(f"  • Average CV R²: {np.mean(cv_scores):.4f}")
        print(f"  • Average CV MAE: {np.mean(cv_mae_scores):.2f}")
        
        # Final training on all data
        print("\n📈 Final model training...")
        cases_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        cases_model.fit(X_scaled, y_cases)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': cases_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 10 Important Features for Cases:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  • {row['feature']}: {row['importance']:.4f}")
        
        # ==================== DEATHS PREDICTION STRATEGY ====================
        print("\n" + "=" * 70)
        print("💀 TRAINING DEATHS PREDICTION MODEL")
        print("=" * 70)
        
        # Check if we have enough death data for direct prediction
        total_deaths = train_df['Deaths'].sum() if 'Deaths' in train_df.columns else 0
        death_records = (train_df['Deaths'] > 0).sum() if 'Deaths' in train_df.columns else 0
        
        print(f"  • Total deaths in dataset: {total_deaths}")
        print(f"  • Records with deaths: {death_records}/{len(train_df)}")
        print(f"  • Death percentage: {(death_records/len(train_df)*100):.1f}%")
        
        # DECISION: Since death data is sparse, use CFR method
        print("\n  ⚠️  Death data is sparse. Using CFR-based prediction method.")
        
        # Ensure CFR exists
        if 'CFR' not in train_df.columns and 'Deaths' in train_df.columns:
            train_df['CFR'] = (train_df['Deaths'] / train_df['Cases'].clip(lower=1)) * 100
        
        if 'CFR' in train_df.columns:
            y_cfr = train_df['CFR'].values
            
            # Train CFR model
            cfr_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            cfr_model.fit(X_scaled, y_cfr)
            
            # Evaluate CFR model
            cfr_predictions = cfr_model.predict(X_scaled)
            cfr_r2 = r2_score(y_cfr, cfr_predictions)
            cfr_mae = mean_absolute_error(y_cfr, cfr_predictions)
            
            print(f"  • CFR prediction R²: {cfr_r2:.4f}")
            print(f"  • CFR prediction MAE: {cfr_mae:.2f}%")
            print(f"  • Average CFR: {train_df['CFR'].mean():.2f}%")
            print(f"  • Min CFR: {train_df['CFR'].min():.2f}%")
            print(f"  • Max CFR: {train_df['CFR'].max():.2f}%")
            
            deaths_model = cfr_model
            use_cfr_method = True
        else:
            print("  ❌ No CFR data available. Cannot predict deaths.")
            deaths_model = None
            use_cfr_method = False
        
        # ==================== SAVE MODELS ====================
        print("\n" + "=" * 70)
        print("💾 SAVING MODELS")
        print("=" * 70)
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📂 Saving models to: {script_dir}")
        
        # Save cases model and scaler
        joblib.dump(cases_model, os.path.join(script_dir, 'cases_model.pkl'))
        joblib.dump(scaler_cases, os.path.join(script_dir, 'scaler_cases.pkl'))
        
        # Save CFR model
        if deaths_model is not None:
            joblib.dump(deaths_model, os.path.join(script_dir, 'cfr_model.pkl'))
        
        # Save encoders
        joblib.dump(le_location, os.path.join(script_dir, 'location_encoder.pkl'))
        joblib.dump(le_disease, os.path.join(script_dir, 'disease_encoder.pkl'))
        
        # Save feature names
        joblib.dump(available_features, os.path.join(script_dir, 'feature_names.pkl'))
        
        # Save comprehensive statistics
        location_stats = train_df.groupby('Location').agg({
            'Rainfall (mm)': ['mean', 'std'],
            'Temperature (°C)': ['mean', 'std'],
            'Population Density': 'mean',
            'Climate Risk Index': 'mean',
            'Cases': ['mean', 'median', 'std', 'sum'],
            'Deaths': 'sum' if 'Deaths' in train_df.columns else None,
            'CFR': 'mean' if 'CFR' in train_df.columns else None
        }).round(2)
        
        location_stats_dict = location_stats.to_dict('index')
        joblib.dump(location_stats_dict, os.path.join(script_dir, 'location_stats.pkl'))
        
        # Disease statistics
        disease_stats = train_df.groupby('Disease').agg({
            'Cases': ['mean', 'median', 'sum'],
            'Deaths': 'sum' if 'Deaths' in train_df.columns else None,
            'CFR': ['mean', 'median'] if 'CFR' in train_df.columns else None
        }).round(2)
        
        disease_stats_dict = disease_stats.to_dict('index')
        joblib.dump(disease_stats_dict, os.path.join(script_dir, 'disease_stats.pkl'))
        
        # Save prediction method info
        metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(train_df),
            'n_features': len(available_features),
            'years_range': [int(train_df['Year'].min()), int(train_df['Year'].max())],
            'locations': le_location.classes_.tolist(),
            'diseases': le_disease.classes_.tolist(),
            'death_prediction_method': 'cfr' if use_cfr_method else 'none',
            'has_death_data': 'Deaths' in train_df.columns,
            'total_deaths': int(total_deaths),
            'cv_scores': [float(s) for s in cv_scores],
            'cv_mae_scores': [float(s) for s in cv_mae_scores],
            'avg_cv_score': float(np.mean(cv_scores)),
            'avg_cv_mae': float(np.mean(cv_mae_scores)),
            'top_features': feature_importance.head(10).to_dict('records')
        }
        
        joblib.dump(metadata, os.path.join(script_dir, 'metadata.pkl'))
        
        # ==================== PREDICTION TEST ====================
        print("\n" + "=" * 70)
        print("🧪 PREDICTION TEST")
        print("=" * 70)
        
        # Test prediction for a sample
        sample_idx = 0
        sample_data = train_df.iloc[sample_idx]
        
        print(f"\n📅 Testing prediction for:")
        print(f"  • Year: {sample_data['Year']}")
        print(f"  • Location: {sample_data['Location']}")
        print(f"  • Disease: {sample_data['Disease']}")
        print(f"  • Actual Cases: {sample_data['Cases']}")
        if 'Deaths' in sample_data:
            print(f"  • Actual Deaths: {sample_data['Deaths']}")
        if 'CFR' in sample_data:
            print(f"  • Actual CFR: {sample_data['CFR']:.2f}%")
        
        # Prepare features for this sample
        X_sample = X.iloc[[sample_idx]].copy()
        X_sample_scaled = scaler_cases.transform(X_sample)
        
        # Predict cases
        predicted_cases = cases_model.predict(X_sample_scaled)[0]
        print(f"  • Predicted Cases: {predicted_cases:.0f}")
        print(f"  • Prediction Error: {abs(predicted_cases - sample_data['Cases']):.0f} cases")
        
        # Predict deaths via CFR
        if deaths_model is not None and use_cfr_method:
            predicted_cfr = deaths_model.predict(X_sample_scaled)[0]
            predicted_deaths = predicted_cases * (predicted_cfr / 100)
            
            print(f"  • Predicted CFR: {predicted_cfr:.2f}%")
            print(f"  • Estimated Deaths: {predicted_deaths:.1f}")
            
            if 'Deaths' in sample_data:
                print(f"  • Death Prediction Error: {abs(predicted_deaths - sample_data['Deaths']):.1f} deaths")
        
        # ==================== FINAL SUMMARY ====================
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        
        print(f"\n🎯 CASES MODEL PERFORMANCE:")
        print(f"  • Cross-validation R²: {np.mean(cv_scores):.4f}")
        print(f"  • Cross-validation MAE: {np.mean(cv_mae_scores):.2f} cases")
        print(f"  • Most important feature: {feature_importance.iloc[0]['feature']}")
        
        print(f"\n📊 DATA STATISTICS:")
        print(f"  • Total records: {len(train_df)}")
        print(f"  • Locations: {len(le_location.classes_)}")
        print(f"  • Diseases: {len(le_disease.classes_)}")
        print(f"  • Time span: {train_df['Year'].min()} to {train_df['Year'].max()}")
        print(f"  • Total cases: {train_df['Cases'].sum():,}")
        
        if 'Deaths' in train_df.columns:
            death_rate = (train_df['Deaths'].sum() / train_df['Cases'].sum() * 100) if train_df['Cases'].sum() > 0 else 0
            print(f"  • Overall death rate: {death_rate:.2f}%")
        
        print(f"\n💡 PREDICTION METHOD:")
        if use_cfr_method:
            print(f"  • Deaths = Cases × Predicted CFR")
            print(f"  • Average CFR in data: {train_df['CFR'].mean():.2f}%")
            print(f"  • CFR Range: {train_df['CFR'].min():.2f}% to {train_df['CFR'].max():.2f}%")
        else:
            print(f"  • Only cases prediction available")
        
        # List all saved files
        print(f"\n📁 Models saved in: {script_dir}")
        saved_files = [
            'cases_model.pkl', 'scaler_cases.pkl', 
            'cfr_model.pkl', 'location_encoder.pkl', 
            'disease_encoder.pkl', 'feature_names.pkl',
            'location_stats.pkl', 'disease_stats.pkl', 
            'metadata.pkl'
        ]
        
        for file in saved_files:
            file_path = os.path.join(script_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                print(f"  ✓ {file} ({file_size:.1f} KB)")
            else:
                print(f"  ✗ {file} (not created)")
        
        print(f"\n⏰ Training completed at {datetime.now()}")
        
        return {
            'cases_model': cases_model,
            'deaths_model': deaths_model,
            'le_location': le_location,
            'le_disease': le_disease,
            'scaler_cases': scaler_cases,
            'use_cfr_method': use_cfr_method,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    # Check current working directory
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📂 Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Train and save model
    results = train_and_save_model()