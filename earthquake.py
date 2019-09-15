
# x -> segment
# preds_lgb_test

#############
# Libraries #
#############

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from tsfresh.feature_extraction import feature_calculators
warnings.filterwarnings("ignore")

#########################
# Prepare training data #
#########################

train = pd.read_csv("./input/train/train.csv", dtype = {"acoustic_data": np.int16, "time_to_failure": np.float32})

segment_length = 150_000 # test segments are 150 000 each
num_segments = train.shape[0] // segment_length # results in 4194 segments for train data

x_train = pd.DataFrame(index = range(num_segments), dtype = np.float64)
y_train = pd.DataFrame(index = range(num_segments), dtype = np.float64, columns = ["time_to_failure"])

# ------------- Visualize a segment of data -------------
fig, ax1 = plt.subplots(figsize = (12, 6))
plt.title("A segment of data")
plt.plot(train["acoustic_data"].values[0:12582910], color = "blue")
ax1.set_ylabel("Acoustic data", color = "blue")
ax2 = ax1.twinx()
plt.plot(train["time_to_failure"].values[0:12582910], color = "green")
ax2.set_ylabel("Time to failure", color = "green")

#####################################
# Feature engineering training data #
#####################################

for i in range(num_segments):
    segment = train.iloc[(i * segment_length):(i * segment_length + segment_length)]
    y = segment["time_to_failure"].values[-1] # last value  
    y_train.loc[i, "time_to_failure"] = y
    segment = pd.Series(segment["acoustic_data"].values) 
    
    x_train.loc[i, "mean"] = segment.mean()
    x_train.loc[i, "max_to_min"] = segment.max() / np.abs(segment.min())    
    x_train.loc[i, "skew"] = segment.skew()       
    x_train.loc[i, "num_peaks_2"] = feature_calculators.number_peaks(segment, 2) # number of peaks of at least support 2
    
    x_train.loc[i, "std_first_50k"] = segment[:50000].std()
    x_train.loc[i, "std_last_50k"] = segment[-50000:].std()
    x_train.loc[i, "std_first_10k"] = segment[:10000].std()
    x_train.loc[i, "std_last_10k"] = segment[-10000:].std()
    
    x_train.loc[i, "avg_first_50k"] = segment[:50000].mean()
    x_train.loc[i, "avg_last_50k"] = segment[-50000:].mean()
    x_train.loc[i, "avg_first_10k"] = segment[:10000].mean()
    x_train.loc[i, "avg_last_10k"] = segment[-10000:].mean()
    
    x_train.loc[i, "min_first_50k"] = segment[:50000].min()
    x_train.loc[i, "min_last_50k"] = segment[-50000:].min()
    x_train.loc[i, "min_first_10k"] = segment[0:10000].min()
    x_train.loc[i, "min_last_10k"] = segment[-10000:].min()
    
    x_train.loc[i, "max_first_50k"] = segment[:50000].max()
    x_train.loc[i, "max_last_50k"] = segment[-50000:].max()
    x_train.loc[i, "max_first_10k"] = segment[:10000].max()
    x_train.loc[i, "max_last_10k"] = segment[-10000:].max()
    
    for window_size in [10, 100, 1000]: 
        roll_avg = segment.rolling(window_size).mean().dropna().values # calculate rolling avg with window size 10, 100, 1000
        roll_std = segment.rolling(window_size).std().dropna().values
        
        x_train.loc[i, f"avg_roll_avg_{window_size}"] = roll_avg.mean() # create features from the rolling values
        x_train.loc[i, f"std_roll_avg_{window_size}"] = roll_avg.std()
        x_train.loc[i, f"max_roll_avg_{window_size}"] = roll_avg.max()
        x_train.loc[i, f"min_roll_avg_{window_size}"] = roll_avg.min()
        x_train.loc[i, f"q01_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.01)
        x_train.loc[i, f"q05_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.05)
        x_train.loc[i, f"q95_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.95)
        x_train.loc[i, f"q99_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.99)
        x_train.loc[i, f"avg_change_abs_roll_avg_{window_size}"] = np.mean(np.diff(roll_avg))
        x_train.loc[i, f"avg_change_rate_roll_avg_{window_size}"] = np.mean(np.nonzero((np.diff(roll_avg) / roll_avg[:-1]))[0])
        
        x_train.loc[i, f"avg_roll_std_{window_size}"] = roll_std.mean()
        x_train.loc[i, f"std_roll_std_{window_size}"] = roll_std.std()
        x_train.loc[i, f"max_roll_std_{window_size}"] = roll_std.max()
        x_train.loc[i, f"min_roll_std_{window_size}"] = roll_std.min()
        x_train.loc[i, f"q01_roll_std_{window_size}"] = np.quantile(roll_std, 0.01)
        x_train.loc[i, f"q05_roll_std_{window_size}"] = np.quantile(roll_std, 0.05)
        x_train.loc[i, f"q95_roll_std_{window_size}"] = np.quantile(roll_std, 0.95)
        x_train.loc[i, f"q99_roll_std_{window_size}"] = np.quantile(roll_std, 0.99)
        x_train.loc[i, f"avg_change_abs_roll_std_{window_size}"] = np.mean(np.diff(roll_std))
        x_train.loc[i, f"avg_change_rate_roll_std_{window_size}"] = np.mean(np.nonzero((np.diff(roll_std) / roll_std[:-1]))[0])

#################
# Visualization #
#################

high_corr_cols = np.abs(x_train.corrwith(y_train["time_to_failure"])).sort_values().tail(14)
high_corr_cols_list = list(high_corr_cols.index)
print(high_corr_cols)

# ------------- Visualize top features -------------
plt.figure(figsize = (44, 24))
for i, col in enumerate(high_corr_cols_list):
    plt.subplot(7, 2, i + 1)
    plt.plot(x_train[col], color = "blue")
    plt.plot(y_train, color = "green")
    plt.title(col)
    plt.legend([col, "time_to_failure"], loc = "upper right")

############################
# Preprocess training data #
############################

# ------------- Mean imputation -------------
col_means_train = {}
for col in x_train.columns:
    if x_train[col].isnull().any():
        print(f"Imputing column {col}...")
        mean_value = x_train.loc[x_train[col] != -np.inf, col].mean()
        col_means_train[col] = mean_value
        x_train.loc[x_train[col] == -np.inf, col] = mean_value
        x_train[col] = x_train[col].fillna(mean_value)       

# ------------- Scaling -------------
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_train = pd.DataFrame(x_train_scaled, columns = x_train.columns)

#####################
# Prepare test data #
#####################

test_ids = pd.read_csv("./input/sample_submission.csv", index_col = "seg_id").index
x_test = pd.DataFrame(index = test_ids, columns = x_train.columns, dtype = np.float64)
      
# ------------- Visualize some test data -------------
plt.figure(figsize = (11, 8))
for i in range(12):
    segment = pd.read_csv("./input/test/" + test_ids[i] + ".csv")
    plt.subplot(6, 2, i + 1)
    plt.plot(segment["acoustic_data"])
plt.tight_layout()

for i, seg_id in enumerate(x_test.index):
    segment = pd.read_csv("./input/test/" + seg_id + ".csv")
    segment = pd.Series(segment["acoustic_data"].values)
    
    x_test.loc[seg_id, "mean"] = segment.mean()
    x_test.loc[seg_id, "max_to_min"] = segment.max() / np.abs(segment.min())
    x_test.loc[seg_id, "skew"] = segment.skew()
    x_test.loc[seg_id, "num_peaks_2"] = feature_calculators.number_peaks(segment, 2)
    
    x_test.loc[seg_id, "std_first_50k"] = segment[:50000].std()
    x_test.loc[seg_id, "std_last_50k"] = segment[-50000:].std()
    x_test.loc[seg_id, "std_first_10k"] = segment[:10000].std()
    x_test.loc[seg_id, "std_last_10k"] = segment[-10000:].std()
    
    x_test.loc[seg_id, "avg_first_50k"] = segment[:50000].mean()
    x_test.loc[seg_id, "avg_last_50k"] = segment[-50000:].mean()
    x_test.loc[seg_id, "avg_first_10k"] = segment[:10000].mean()
    x_test.loc[seg_id, "avg_last_10k"] = segment[-10000:].mean()
    
    x_test.loc[seg_id, "min_first_50k"] = segment[:50000].min()
    x_test.loc[seg_id, "min_last_50k"] = segment[-50000:].min()
    x_test.loc[seg_id, "min_first_10k"] = segment[:10000].min()
    x_test.loc[seg_id, "min_last_10k"] = segment[-10000:].min()
    
    x_test.loc[seg_id, "max_first_50k"] = segment[:50000].max()
    x_test.loc[seg_id, "max_last_50k"] = segment[-50000:].max()
    x_test.loc[seg_id, "max_first_10k"] = segment[:10000].max()
    x_test.loc[seg_id, "max_last_10k"] = segment[-10000:].max()
    
    for window_size in [10, 100, 1000]:
        roll_avg = segment.rolling(window_size).mean().dropna().values
        roll_std = segment.rolling(window_size).std().dropna().values
        
        x_test.loc[seg_id, f"avg_roll_avg_{window_size}"] = roll_avg.mean()
        x_test.loc[seg_id, f"std_roll_avg_{window_size}"] = roll_avg.std()
        x_test.loc[seg_id, f"max_roll_avg_{window_size}"] = roll_avg.max()
        x_test.loc[seg_id, f"min_roll_avg_{window_size}"] = roll_avg.min()
        x_test.loc[seg_id, f"q01_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.01)
        x_test.loc[seg_id, f"q05_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.05)
        x_test.loc[seg_id, f"q95_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.95)
        x_test.loc[seg_id, f"q99_roll_avg_{window_size}"] = np.quantile(roll_avg, 0.99)
        x_test.loc[seg_id, f"avg_change_abs_roll_avg_{window_size}"] = np.mean(np.diff(roll_avg))
        x_test.loc[seg_id, f"avg_change_rate_roll_avg_{window_size}"] = np.mean(np.nonzero((np.diff(roll_avg) / roll_avg[:-1]))[0])
        
        x_test.loc[seg_id, f"avg_roll_std_{window_size}"] = roll_std.mean()
        x_test.loc[seg_id, f"std_roll_std_{window_size}"] = roll_std.std()
        x_test.loc[seg_id, f"max_roll_std_{window_size}"] = roll_std.max()
        x_test.loc[seg_id, f"min_roll_std_{window_size}"] = roll_std.min()
        x_test.loc[seg_id, f"q01_roll_std_{window_size}"] = np.quantile(roll_std, 0.01)
        x_test.loc[seg_id, f"q05_roll_std_{window_size}"] = np.quantile(roll_std, 0.05)
        x_test.loc[seg_id, f"q95_roll_std_{window_size}"] = np.quantile(roll_std, 0.95)
        x_test.loc[seg_id, f"q99_roll_std_{window_size}"] = np.quantile(roll_std, 0.99)
        x_test.loc[seg_id, f"avg_change_abs_roll_std_{window_size}"] = np.mean(np.diff(roll_std))
        x_test.loc[seg_id, f"avg_change_rate_roll_std_{window_size}"] = np.mean(np.nonzero((np.diff(roll_std) / roll_std[:-1]))[0])
        
for col in x_test.columns:
    if x_test[col].isnull().any():
        print(f"Imputing column {col}...")
        x_test.loc[x_test[col] == -np.inf, col] = col_means_train[col] # if -Inf set train mean
        x_test[col] = x_test[col].fillna(col_means_train[col]) # if NA set train mean      

x_test_scaled = scaler.transform(x_test) # use the scaler fitted to training data
x_test = pd.DataFrame(x_test_scaled, columns = x_test.columns)

##################
# Specify models #
##################

num_folds = 5
folds = KFold(num_folds, shuffle = True, random_state = 2019)

def train_model(X = None, y = None, X_test = None, params = None, folds = folds,
                model_type = None, feat_importance = False):

    preds_oof_all = np.zeros(len(X)) # out-of-fold predictions
    preds_test_all = np.zeros(len(X_test)) # test set predictions
    errors_oof_all = [] # mean absolute error for out-of-fold predictions
    feature_importances = pd.DataFrame()
    
    for fold_i, (train_i, oof_i) in enumerate(folds.split(X)):
        x_train, x_oof = X.iloc[train_i], X.iloc[oof_i]
        y_train, y_oof = y.iloc[train_i], y.iloc[oof_i]
        
        # ---------- Model types ----------
        if model_type == "lgb":
            model = lgb.LGBMRegressor(**params, n_estimators = 50_000, n_jobs = -1)
            model.fit(
                    x_train,
                    y_train, 
                    eval_set = [(x_train, y_train), (x_oof, y_oof)],
                    eval_metric = "mae",
                    verbose = 10_000,
                    early_stopping_rounds = 200
            )            
            preds_oof = model.predict(x_oof)
            preds_test = model.predict(X_test, num_iteration = model.best_iteration_)
            
        if model_type == "xgb":
            xgb_train = xgb.DMatrix(x_train, y_train, feature_names = X.columns)
            xgb_oof = xgb.DMatrix(x_oof, y_oof, feature_names = X.columns)
            xgb_oof_nolabel = xgb.DMatrix(x_oof, feature_names = X.columns)
            xgb_test = xgb.DMatrix(X_test, feature_names = X.columns)
            model = xgb.train(
                    dtrain = xgb_train,
                    num_boost_round = 20_000,
                    evals = [(xgb_train, "train"), (xgb_oof, "valid_data")],
                    early_stopping_rounds = 200,
                    verbose_eval = 500,
                    params = params
            )
            preds_oof = model.predict(xgb_oof_nolabel, ntree_limit = model.best_ntree_limit)
            preds_test = model.predict(xgb_test, ntree_limit = model.best_ntree_limit)
        
        if model_type == "nusvr":
            model = NuSVR(gamma = "scale", nu = 0.7, tol = 0.01, C = 1.0)
            model.fit(x_train, y_train)          
            preds_oof = model.predict(x_oof).reshape(-1, )
            error_oof = mean_absolute_error(y_oof, preds_oof)
            print(f"Fold {fold_i}. MAE: {error_oof:.4f}.")        
            preds_test = model.predict(X_test).reshape(-1,)
        
        if model_type == "krr":
            model = KernelRidge(kernel = "rbf", alpha = 0.1, gamma = 0.01)
            model.fit(x_train, y_train)          
            preds_oof = model.predict(x_oof).reshape(-1, )
            error_oof = mean_absolute_error(y_oof, preds_oof)
            print(f"Fold {fold_i}. MAE: {error_oof:.4f}.")         
            preds_test = model.predict(X_test).reshape(-1,)
                    
        preds_oof_all[oof_i] = preds_oof.reshape(-1, ) # set out-of-fold preds to right index
        preds_test_all += preds_test # ???????????????
        error_oof = mean_absolute_error(y_oof, preds_oof)
        errors_oof_all.append(error_oof) # append errors from current fold
       
        # ---------- Feature importance in fold ----------
        if (model_type == "lgb" and feat_importance == True):
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_i + 1
            feature_importances = pd.concat([feature_importances, fold_importance], axis = 0)
            
    # ---------- Aggregating results over folds ----------        
    preds_test_all /= num_folds
    mean_error = np.mean(errors_oof_all)
    std_error = np.std(errors_oof_all)
    print(f"CV error mean: {mean_error:.4f}, std: {std_error:.4f}")
    
    # ---------- Feature importance total ----------
    if (model_type == "lgb" and feat_importance == True):
        feature_importances["importance"] /= num_folds
        cols = feature_importances[["feature", "importance"]].groupby("feature").mean().sort_values("importance", ascending = False)[0:30].index
        best_features = feature_importances.loc[feature_importances.feature.isin(cols)]
        plt.figure(figsize = (13, 7));
        sns.barplot(x = "importance", y = "feature", data=best_features.sort_values("importance", ascending = False));
        plt.title("LGB best features (avg over folds)");    
        return preds_oof_all, preds_test_all    
    else:
        return preds_oof_all, preds_test_all
    
##############
# Fit models #
##############

# ------------- 1. LGB -------------  
lgb_params = {
        "num_leaves": 128,
        "min_data_in_leaf": 79,
        "objective": "huber",
        "max_depth": -1,
        "learning_rate": 0.01,
        "boosting": "gbdt",
        "bagging_freq": 5,
        "bagging_fraction": 0.8127,
        "bagging_seed": 11,
        "metric": "mae",
        "verbosity": -1,
        "reg_alpha": 0.1303,
        "reg_lambda": 0.3603
}

preds_lgb_train, preds_lgb_test, = train_model(
        x_train,
        y_train,
        x_test,
        model_type = "lgb",
        params = lgb_params,
        feat_importance = True
)

# ------------- 2. XGB -------------
xgb_params = {
        "eta": 0.03,
        "max_depth": 9,
        "subsample": 0.85,
        "objective": "reg:linear",
        "eval_metric": "mae",
        "silent": True,
        "nthread": 4
}

preds_xgb_train, preds_xgb_test = train_model(
        x_train,
        y_train,
        x_test,
        model_type = "xgb",
        params = xgb_params
)

# ------------- 3. NuSVR -------------
preds_svr_train, preds_svr_test = train_model(
        x_train,
        y_train,
        x_test,
        model_type = "nusvr"
)

# ------------- 4. KRR -------------
preds_krr_train, preds_krr_test = train_model(
        x_train,
        y_train,
        x_test,
        model_type = "krr"
)
    
##############
# Prediction #
##############

# ------------- Blending -------------
preds_blend_train = (preds_lgb_train + preds_xgb_train + preds_svr_train + preds_krr_train) / 4
preds_blend_test = (preds_lgb_test + preds_xgb_test + preds_svr_test + preds_krr_test) / 4

# ------------- Model stacking -------------
stack_train = np.vstack([preds_lgb_train, preds_xgb_train, preds_svr_train, preds_krr_train]).transpose()
stack_train = pd.DataFrame(stack_train, columns = ["lgb", "xgb", "svr", "krr"])
stack_test = np.vstack([preds_lgb_test, preds_xgb_test, preds_svr_test, preds_krr_test]).transpose()
stack_test = pd.DataFrame(stack_test)

preds_stack_train, preds_stack_test = train_model(
        stack_train,
        y_train,
        stack_test,
        params = lgb_params,
        model_type = "lgb",
        feat_importance = True
)

##############
# Evaluation #
##############

plt.figure(figsize = (18, 8))
# ---------------- LGB ----------------
plt.subplot(2, 3, 1)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_lgb_train, color = "blue", label = "lgb", alpha = 0.5)
plt.legend(loc = "upper right")
plt.title("lgb")
# ---------------- XGB ----------------
plt.subplot(2, 3, 2)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_xgb_train, color = "teal", label = "xgb", alpha = 0.5)
plt.legend(loc = "upper right")
plt.title("xgb")
# ---------------- NuSVR ----------------
plt.subplot(2, 3, 3)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_svr_train, color = "red", label = "svr", alpha = 0.5)
plt.legend(loc = "upper right")
plt.title("svr")
# ---------------- KRR ----------------
plt.subplot(2, 3, 4)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_krr_train, color = "magenta", label = "krr", alpha = 0.5)
plt.legend(loc = "upper right");
plt.title("krr");
# ---------------- Stack ----------------
plt.subplot(2, 3, 5)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_stack_train, color = "gold", label = "stack", alpha = 0.7)
plt.legend(loc = "upper right")
plt.title("stack")
# ---------------- Blend ----------------
plt.subplot(2, 3, 6)
plt.plot(y_train, color = "green", label = "y_train")
plt.plot(preds_blend_train, color = "gold", label = "blend", alpha = 0.7)
plt.legend(loc = "upper right")
plt.title("blend")
# -------------------------------------
plt.suptitle("Predictions vs actual")

############################
# Model blending/averaging #
############################

test_df = pd.read_csv("./input/sample_submission.csv", index_col = "seg_id")

test_df["time_to_failure"] = preds_blend_test
test_df.to_csv("submission_blend.csv")
test_df["time_to_failure"] = preds_stack_test
test_df.to_csv("submission_stack.csv")
