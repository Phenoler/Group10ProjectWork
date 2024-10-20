import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from textblob import TextBlob
from statsmodels.formula.api import ols
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class SentimentAnalysis: 
    def __init__(self):
        self.df = None
        self.analyzer = SentimentIntensityAnalyzer()
        
    def load_data(self, path):
        self.df = pd.read_csv(path)

    def get_text_columns(self):
        text_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        average_length = self.df[text_columns].applymap(lambda x: len(x)).mean().values
        unique_entries = self.df[text_columns].nunique().values
        
        return pd.DataFrame({
            'Column Name': text_columns,
            'Average Entry Length': average_length,
            'Unique Entries': unique_entries
        })

    def vader_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        
        for text in data:
            score = self.analyzer.polarity_scores(text)['compound']
            scores.append(score)
            sentiments.append('positive' if score > 0 else ('negative' if score < 0 else 'neutral'))
        
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        subjectivity_scores = []

        for text in data:
            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)
            sentiments.append('positive' if blob.sentiment.polarity > 0 else ('negative' if blob.sentiment.polarity < 0 else 'neutral'))
            subjectivity_scores.append(blob.sentiment.subjectivity)

        return scores, sentiments, subjectivity_scores

    def distilbert_sentiment_analysis(self, data):
        if not pipeline:
            raise ImportError("Transformers library is not available.")
        
        sentiment_pipeline = pipeline("sentiment-analysis")
        scores = []
        sentiments = []
        
        for text in data:
            result = sentiment_pipeline(text)[0]
            scores.append(result['score'])
            sentiments.append(result['label'])
        
        return scores, sentiments

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_distribution(data, column):
    plt.figure(figsize=(10, 6))
    data[column].hist(bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def conduct_anova(data, group_col, value_col):
    # Conduct ANOVA
    model = ols(f'{value_col} ~ C({group_col})', data=data).fit()
    anova_table = stats.f_oneway(*(data[data[group_col] == group][value_col].dropna() for group in data[group_col].unique()))
    
    print(f'ANOVA F-statistic: {anova_table.statistic:.4f}, p-value: {anova_table.pvalue:.15f}')
    
    # Check normality and plot Q-Q plot
    plt.figure(figsize=(10, 6))
    stats.probplot(data[value_col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q plot for {value_col}')
    plt.axhline(0, color='red', linestyle='--')
    plt.axvline(0, color='red', linestyle='--')
    plt.show()
    
    # Normality test
    stat, p_value = stats.shapiro(data[value_col].dropna())
    if p_value > 0.05:
        print(f'{value_col} is normally distributed.')
    else:
        print(f'{value_col} is not normally distributed.')
        print("Performing Kruskal-Wallis Test instead...")
        kruskal_stat, kruskal_p = stats.kruskal(*(data[data[group_col] == group][value_col].dropna() for group in data[group_col].unique()))
        print(f'Kruskal-Wallis Statistic: {kruskal_stat:.4f}, p-value: {kruskal_p:.15f}')
        
        if kruskal_p < 0.05:
            print("Result is statistically significant.")
            print(f"There is a statistically significant difference in the average {value_col} across the categories of {group_col}.")
        else:
            print("Result is not statistically significant.")
            print(f"There is no statistically significant difference in the average {value_col} across the categories of {group_col}.")

def conduct_t_test(data, group_col, value_col):
    groups = [group[value_col].dropna() for name, group in data.groupby(group_col)]
    
    if len(groups) != 2:
        print("t-test requires exactly 2 groups.")
        return
    
    if len(groups[0]) < 2 or len(groups[1]) < 2:
        print("Each group must have at least 2 samples for t-test.")
        return

    t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
    print(f'T-test statistic: {t_stat:.4f}, p-value: {p_value:.15f}')

def conduct_chi_square(data, col1, col2):
    # Check if variables are categorical
    if not pd.api.types.is_categorical_dtype(data[col1]) and not pd.api.types.is_object_dtype(data[col1]):
        print(f"Variable '{col1}' is not categorical, unable to conduct chi-square test.")
        return
    if not pd.api.types.is_categorical_dtype(data[col2]) and not pd.api.types.is_object_dtype(data[col2]):
        print(f"Variable '{col2}' is not categorical, unable to conduct chi-square test.")
        return
    
    contingency_table = pd.crosstab(data[col1], data[col2])
    
    # Check expected frequencies
    chi2, p, _, expected = stats.chi2_contingency(contingency_table)
    
    if (expected < 5).any():
        print("Expected frequencies for chi-square test should be greater than 5, some cells do not meet this condition.")
        print(f"Expected frequencies:\n{expected}")
        return
    
    print(f'Chi-square statistic: {chi2:.4f}, p-value: {p:.15f}')

def conduct_regression(data, x_cols, y_col):
    X = data[x_cols].dropna()
    y = data[y_col].dropna()
    min_length = min(len(X), len(y))
    X = X[:min_length]
    y = y[:min_length]
    
    # Add constant term
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Output analysis conclusions
    print("\n--- Regression Analysis Conclusions ---")
    
    # R² and adjusted R²
    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    print(f"Model's R² value: {r_squared:.4f}")
    print(f"Adjusted R² value: {adjusted_r_squared:.4f}")
    
    # Coefficients and p-values
    for col in x_cols:
        coef = model.params[col]
        p_value = model.pvalues[col]
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"Independent variable '{col}' coefficient: {coef:.4f}, p-value: {p_value:.4f} ({significance})")
    
    print("\n--- Summary ---")
    if r_squared >= 0.85:
        print("Model fits well, explaining a high proportion of variance in the dependent variable.")
    else:
        print("Model fits poorly, further exploration of other independent variables may be needed.")

def process_group_column(data, group_col, selected_value):
    # Create new column, initialize as "other"
    new_col_name = f'{group_col}_processed'
    data[new_col_name] = 'other'
    
    # Update selected value to its original
    data.loc[data[group_col] == selected_value, new_col_name] = selected_value
    return new_col_name

def main():
    file_path = input("Please enter the path of the CSV file: ")
    data = load_data(file_path)

    print("The following variables are in the dataset:")
    print(data.columns.tolist())

    sa = SentimentAnalysis()
    sa.load_data(file_path)

    while True:
        print("\nEnter your choice:")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")

        choice = input("Please enter your choice (1-7): ")

        if choice == '1':
            print("Available variables:")
            for idx, col in enumerate(data.columns, start=1):
                print(f"{idx}. {col}")  # Print available variables with numbers
            print(f"{len(data.columns) + 1}. QUIT")
            
            while True:
                column_idx = input("Please enter the number of the variable to plot: ")
                
                if column_idx.isdigit() and 1 <= int(column_idx) <= len(data.columns):
                    column = data.columns[int(column_idx) - 1]
                    plot_distribution(data, column)
                    break  # Exit inner loop
                elif column_idx == str(len(data.columns) + 1):
                    break  # Return to main menu
                else:
                    print("Invalid variable number, please re-enter.")  # Error message

        elif choice == '2':
            print("For ANOVA, the available variables are:")
            for idx, col in enumerate(data.columns, start=1):
                print(f"{idx}. {col}")
            print(f"{len(data.columns) + 1}. QUIT")
            
            while True:
                continuous_var = input("Please enter the name of the continuous variable: ")
                if continuous_var not in data.columns:
                    print("Invalid variable name, please re-enter.")
                    continue

                categorical_var = input("Please enter the name of the categorical variable: ")
                if categorical_var not in data.columns:
                    print("Invalid variable name, please re-enter.")
                    continue

                print("Performing ANOVA on the selected variables...")
                conduct_anova(data, categorical_var, continuous_var)
                break  # Exit inner loop after valid input

        elif choice == '3':
            print("Available grouping variables:")
            for idx, col in enumerate(data.columns, start=1):
                print(f"{idx}. {col}")
            print(f"{len(data.columns) + 1}. QUIT")

            while True:
                group_col = input("Please enter the name of the grouping variable: ")
                if group_col not in data.columns:
                    print("Invalid variable name, please re-enter.")
                    continue

                # Get unique values of the grouping variable and display
                unique_values = data[group_col].unique()
                print("Available group values:")
                for idx, value in enumerate(unique_values, start=1):
                    print(f"{idx}. {value}")

                selected_idx = input("Please enter the number of the selected group value: ")
                if selected_idx.isdigit() and 1 <= int(selected_idx) <= len(unique_values):
                    selected_value = unique_values[int(selected_idx) - 1]

                    # Create new column, replacing values other than selected with 'other'
                    new_col_name = f"{group_col}_modified"
                    data[new_col_name] = data[group_col].apply(lambda x: x if x == selected_value else 'other')
                    print(f"New column '{new_col_name}' created, selected value {selected_value} retained, others are 'other'.")
                    print("Counts of the new column values:")
                    print(data[new_col_name].value_counts())  # Print counts of new column values

                    # Conduct t-test analysis
                    value_col = input("Please enter the name of the numeric variable: ")
                    if value_col in data.columns:
                        conduct_t_test(data, new_col_name, value_col)
                    else:
                        print("Invalid numeric variable name.")

                    # Drop the generated new column
                    data.drop(columns=new_col_name, inplace=True)
                    break
                else:
                    print("Invalid number, please re-enter.")

        elif choice == '4':
            print("For chi-square test, the available variables are:")
            for idx, col in enumerate(data.columns, start=1):
                print(f"{idx}. {col}")
            print(f"{len(data.columns) + 1}. QUIT")
    
            while True:
                col1 = input("Please enter the name of the first categorical variable: ")
                if col1 == "QUIT":
                    break
                if col1 not in data.columns:
                    print("Invalid variable name, please re-enter.")
                    continue
        
                col2 = input("Please enter the name of the second categorical variable: ")
                if col2 == "QUIT":
                    break
                if col2 not in data.columns:
                    print("Invalid variable name, please re-enter.")
                    continue
        
                conduct_chi_square(data, col1, col2)
                break  # Exit loop after valid input

        elif choice == '5':
            print("For regression analysis, the available variables are:")
            for idx, col in enumerate(data.columns, start=1):
                print(f"{idx}. {col}")

            while True:
                x_cols = input("Please enter the names of independent variables (comma separated): ").split(',')
                y_col = input("Please enter the name of the dependent variable: ")
                x_cols = [col.strip() for col in x_cols]  # Remove spaces

                if all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in x_cols) and y_col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[y_col]) and not pd.api.types.is_categorical_dtype(data[y_col]):
                        print("The dependent variable must be numeric or categorical.")
                        continue
            
                    conduct_regression(data, x_cols, y_col)
                    break  # Exit inner loop after valid input
                else:
                    print("Invalid variable name or independent variable does not meet requirements, please re-enter.")

        elif choice == '6':
            text_columns_df = sa.get_text_columns()
            print("Available text columns:")
            print(text_columns_df)
            
            column_name = input("Please enter the name of the text column to analyze: ")
            if column_name not in text_columns_df['Column Name'].values:
                print(f"{column_name} is not a valid text column.")
                continue
            
            analyzer_choice = input("Select analyzer (1: VADER, 2: TextBlob, 3: DistilBERT): ")
            
            if analyzer_choice == '1':
                scores, sentiments = sa.vader_sentiment_analysis(sa.df[column_name])
            elif analyzer_choice == '2':
                scores, sentiments, subjectivity = sa.textblob_sentiment_analysis(sa.df[column_name])
                print("Subjectivity scores:", subjectivity)
            elif analyzer_choice == '3':
                scores, sentiments = sa.distilbert_sentiment_analysis(sa.df[column_name])
            else:
                print("Invalid choice.")
                continue
            
            print("Scores:", scores)
            print("Sentiments:", sentiments)

        elif choice == '7':
            print("Exiting the program")
            break

        else:
            print("Invalid option, please re-select.")

if __name__ == "__main__":
    main()
