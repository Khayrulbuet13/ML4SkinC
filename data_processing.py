import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn import preprocessing
from sklearn.impute import KNNImputer
import warnings
import joblib

warnings.simplefilter("ignore")




# load data from csv files

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f}'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100*(start_mem - end_mem) / start_mem))
    
    return df


def load_data( Kaggle=False, debug=False):
    if Kaggle:
        train_meta = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv')
        test_meta = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')
    else:
        train_meta = pd.read_csv('dataset/dump/train-metadata.csv')
        test_meta = pd.read_csv('dataset/dump/test-metadata.csv')
    if debug:
        train_meta = train_meta[:80*1000]
    return reduce_mem_usage(train_meta), reduce_mem_usage(test_meta)



import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

def preprocess_and_impute_data(df, columns_to_drop):
    # Drop specified columns safely
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Specific handling for the 'age_approx' column using KNN
    if 'age_approx' in df.columns and df['age_approx'].isnull().any():
        print('Processing column: age_approx with KNN')
        knn_imputer = KNNImputer(n_neighbors=5)
        age_approx_values = df[['age_approx']]
        df['age_approx'] = knn_imputer.fit_transform(age_approx_values).ravel()  # Reshape to 1D array

    # Handle missing data for other numeric columns using mean imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'age_approx' in numeric_cols:
        numeric_cols.remove('age_approx')
    for col in numeric_cols:
        if df[col].isnull().any():
            print(f'Processing column: {col} with mean imputation')
            mean_imputer = SimpleImputer(strategy='mean')
            df[col] = mean_imputer.fit_transform(df[[col]]).ravel()  # Reshape to 1D array
    
    # Handle missing data for categorical columns using median imputation
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            print(f'Processing column: {col} with median imputation')
            median_imputer = SimpleImputer(strategy='most_frequent')
            df[col] = median_imputer.fit_transform(df[[col]]).ravel()  # Reshape to 1D array

    return df



def feature_engineering(df):
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4
    return df

import pandas as pd
from sklearn import preprocessing



from sklearn.metrics import roc_auc_score
def calculate_pauc(y_true, y_scores, tpr_threshold=0.8):
    # Calculate pAUC using sklearn's roc_auc_score with max_fpr
    partial_auc_scaled = roc_auc_score(y_true, y_scores, max_fpr=tpr_threshold)

    # Scale from [0.5, 1.0] to [0.0, 0.2]
    partial_auc = (partial_auc_scaled - 0.5) * 0.4
    return partial_auc



def train_model(train_df, categorical_columns):
    # Assuming 'target' is the column name for your target variable
    X = train_df.drop(columns=['target'])
    y = train_df['target']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    pauc_scores = []
    models = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            cat_features=categorical_columns,
            eval_metric='AUC',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)
        
        test_pred = model.predict_proba(X_test)[:, 1]
        
        models.append(model)

        pauc = calculate_pauc(y_test, test_pred)
        pauc_scores.append(pauc)

    print(f'Average pAUC score: {np.mean(pauc_scores):.4f}')
    
    return models

def predict(models, test_meta_df_le):
    submit_score = []
    for model in models:
        pred_ = model.predict_proba(test_meta_df_le)[:, 1]
        submit_score.append(pred_)
    submit_pred = np.mean(submit_score, axis=0)
    return submit_pred

def create_submission(test_id, submit_pred, filename='submission.csv'):
    submission = pd.DataFrame({
        'isic_id': test_id,
        'target': submit_pred
    })
    submission.to_csv(filename, index=False)
    return submission








import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def fit_and_save_encoders(train_df, categorical_columns, file_path='encoders.joblib'):
    """
    Fits LabelEncoders for each categorical column in train_df and saves the encoders.
    
    :param train_df: DataFrame containing the training data.
    :param categorical_columns: List of column names that are categorical.
    :param file_path: File path to save the encoders.
    :return: DataFrame with encoded categorical columns.
    """
    encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        # Fit the encoder and transform the data
        train_df[column] = le.fit_transform(train_df[column].astype(str))
        # Save the encoder in a dictionary
        encoders[column] = le
    # Save all encoders to disk
    joblib.dump(encoders, file_path)
    return train_df



def load_and_apply_encoders(test_df, categorical_columns, file_path='encoders.joblib'):
    """
    Loads encoders and applies them to the test_df, handling unseen categories.
    
    :param test_df: DataFrame containing the testing data.
    :param categorical_columns: List of column names that are categorical.
    :param file_path: File path where the encoders are saved.
    :return: DataFrame with encoded categorical columns.
    """
    # Load the saved encoders
    encoders = joblib.load(file_path)
    for column in categorical_columns:
        le = encoders[column]
        # Handle unseen categories by using 'transform' method and custom handling for unknown categories
        test_df[column] = test_df[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return test_df









train_df, test_df = load_data(Kaggle=True, debug=False)

test_id = test_df['isic_id']


# Drop columns that are not needed and handle missing data
missin_in_test = ['iddx_3', 'iddx_full', 'iddx_2', 'mel_mitotic_index', 
                  'iddx_1', 'lesion_id', 'tbp_lv_dnn_lesion_confidence', 
                  'iddx_5', 'iddx_4', 'mel_thick_mm']
columns_to_drop = missin_in_test + ['isic_id', 'patient_id', 'sex', 'anatom_site_general', 
                    'image_type', 'tbp_tile_type', 'attribution', 'copyright_license']
test_df = preprocess_and_impute_data(test_df, columns_to_drop)
train_df = preprocess_and_impute_data(train_df, columns_to_drop)


# Label encode categorical columns
categorical_cols = [ 'tbp_lv_location', 'tbp_lv_location_simple']
# Fit encoders to the training data and save them
train_df = fit_and_save_encoders(train_df, categorical_cols)

# Load the encoders and apply them to the testing data
test_df = load_and_apply_encoders(test_df, categorical_cols)




# Feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)





    
# Train the model and make predictions
models = train_model(train_df, categorical_columns=categorical_cols)
submit_pred = predict(models, test_df)

# Create a submission file
create_submission(test_id, submit_pred, filename='submission.csv')
    

