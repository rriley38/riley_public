import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(use_csv=True, file_path=None, sql_conn_str=None, sql_query=None):
    if use_csv:
        df = pd.read_csv(file_path)
        print("✅ Data loaded from CSV.")
    else:
        engine = create_engine(sql_conn_str)
        df = pd.read_sql_query(sql_query, engine)
        print("✅ Data loaded from SQL.")
    return df

def check_data(df):
    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())
    duplicates = df.duplicated().sum()
    print(f"🔍 Number of duplicated rows: {duplicates}")

def clean_data(df, remove_duplicates=True, handle_missing='none'):
    if remove_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        print(f"✅ Removed {before - after} duplicated rows.")

    if handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'fill_zero':
        df = df.fillna(0)
    elif handle_missing == 'fill_mean':
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif handle_missing == 'fill_median':
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def eda(df):
    print(df.info())
    print(df.describe())
    df.hist(bins=20, figsize=(14, 8))
    plt.suptitle("Distributions of Numeric Variables")
    plt.show()

    sns.boxplot(data=df.select_dtypes(include=np.number))
    plt.title("Boxplot of Numeric Variables")
    plt.show()

def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    threshold = 0.7
    corr_pairs = corr.unstack().sort_values(kind="quicksort").drop_duplicates()
    high_corr = corr_pairs[(abs(corr_pairs) > threshold) & (abs(corr_pairs) < 1)]
    print("\n🔗 Strong correlations:")
    print(high_corr)

def trend_analysis(df, date_col=None, metric_col=None, x=None, y=None):
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        trend = df.groupby(df[date_col].dt.to_period('M')).sum(numeric_only=True)
        trend.index = trend.index.to_timestamp()
        trend.plot()
        plt.title("📈 Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Metrics")
        plt.show()

    if x and y and x in df.columns and y in df.columns:
        sns.regplot(data=df, x=x, y=y, line_kws={"color": "red"})
        plt.title(f"📈 Trend: {y} vs {x}")
        plt.show()
        corr_value = df[x].corr(df[y])
        print(f"🔗 Correlation between {x} and {y}: {corr_value:.2f}")

def outlier_analysis(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
    outliers = (z_scores > 3).sum()
    print(f"🔍 Outliers detected per column:\n{outliers}")

def categorical_distribution(df):
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"\n📊 Distribution for {col}:")
        print(df[col].value_counts(normalize=True) * 100)
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

def feature_importance(df, target):
    numeric = df.select_dtypes(include=np.number).dropna()
    features = numeric.drop(columns=[target])
    mi = mutual_info_regression(features, numeric[target])
    mi_scores = pd.Series(mi, index=features.columns).sort_values(ascending=False)
    print("\n📊 Feature Importance (Mutual Information):")
    print(mi_scores)
    mi_scores.plot(kind='bar')
    plt.title('Feature Importance (Mutual Information)')
    plt.show()

def time_series_decomposition(df, date_col, metric_col):
    df[date_col] = pd.to_datetime(df[date_col])
    ts = df.set_index(date_col).resample('M').sum(numeric_only=True)[metric_col]
    result = seasonal_decompose(ts, model='additive')
    result.plot()
    plt.show()

def clustering_analysis(df, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters
    sns.pairplot(df, hue='cluster')
    plt.show()
    return df

# Example usage (uncomment and adjust as needed):
# df = load_data(use_csv=True, file_path='your_file.csv')
# check_data(df)
# df = clean_data(df, remove_duplicates=True, handle_missing='fill_mean')
# eda(df)
# correlation_analysis(df)
# trend_analysis(df, date_col='date', metric_col='sales', x='ad_spend', y='sales')
# outlier_analysis(df)
# categorical_distribution(df)
# feature_importance(df, target='sales')
# time_series_decomposition(df, date_col='date', metric_col='sales')
# df = clustering_analysis(df, n_clusters=3)
