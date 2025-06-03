# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Configure plot styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# 🚀 Data Collection Function
def load_data(source='csv', file_path=None, sql_conn_str=None, sql_query=None):
    """
    Load data from a CSV file or SQL database.

    Parameters:
    - source: 'csv' or 'sql'
    - file_path: Path to the CSV file (if source == 'csv')
    - sql_conn_str: SQLAlchemy connection string (if source == 'sql')
    - sql_query: SQL query to execute (if source == 'sql')

    Returns:
    - pandas DataFrame
    """
    if source == 'csv':
        try:
            df = pd.read_csv(file_path)
            print("✅ Data loaded from CSV.")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            df = pd.DataFrame()
    elif source == 'sql':
        try:
            engine = create_engine(sql_conn_str)
            df = pd.read_sql_query(sql_query, engine)
            print("✅ Data loaded from SQL database.")
        except Exception as e:
            print(f"❌ Failed to load SQL data: {e}")
            df = pd.DataFrame()
    else:
        print("❌ Invalid source. Choose 'csv' or 'sql'.")
        df = pd.DataFrame()

    print(df.head())
    return df


# 🧹 Data Cleaning Function
def clean_data(df):
    """
    Basic data cleaning:
    - Removes duplicates
    - Displays missing values

    Returns:
    - Cleaned DataFrame
    """
    print("\n🧽 Starting data cleaning...")

    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    df = df.drop_duplicates()

    # Optional: Handle missing values here
    # Example:
    # df = df.fillna(0)

    print("\n✅ Data cleaning completed.")
    return df


# 📊 Exploratory Data Analysis Function
def perform_eda(df):
    """
    Perform basic Exploratory Data Analysis (EDA).
    """
    print("\n📊 Starting Exploratory Data Analysis...")

    print(df.describe())

    # Histograms
    df.hist(bins=20, figsize=(14, 8))
    plt.tight_layout()
    plt.show()

    # Boxplot
    sns.boxplot(data=df.select_dtypes(include=np.number))
    plt.title("Boxplot of Numeric Variables")
    plt.show()

    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    print("✅ EDA completed.")


# 🧠 Simple Analysis Function
def simple_analysis(df, x_col, y_col):
    """
    Plot scatter and calculate correlation between two variables.

    Parameters:
    - x_col: Name of X variable
    - y_col: Name of Y variable
    """
    if x_col in df.columns and y_col in df.columns:
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Relationship between {x_col} and {y_col}")
        plt.show()

        correlation = df[x_col].corr(df[y_col])
        print(f"🔗 Correlation between {x_col} and {y_col}: {correlation:.2f}")
    else:
        print(f"❌ Columns '{x_col}' or '{y_col}' not found in the dataset.")


# 📈 Main Execution
if __name__ == "__main__":
    # 🔍 Problem Definition
    print("\n🎯 Problem: Example Analysis - Replace with your objective.")

    # ✅ Load Data (choose 'csv' or 'sql')
    # CSV Example
    # df = load_data(source='csv', file_path='your_file.csv')

    # SQL Example
    df = load_data(
        source='sql',
        sql_conn_str='postgresql://username:password@host:port/database',
        sql_query="""
            SELECT *
            FROM your_table
            WHERE date >= '2024-01-01'
        """
    )

    if not df.empty:
        # 🧹 Clean Data
        df = clean_data(df)

        # 📊 Perform EDA
        perform_eda(df)

        # 🧠 Simple Analysis (replace with relevant columns)
        simple_analysis(df, x_col='variable_x', y_col='variable_y')

        # ✅ End
        print("\n🎉 Data analysis process completed successfully!")
    else:
        print("\n⚠️ No data to analyze.")
