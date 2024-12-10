import pprint
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from lib.lib import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import norm
from skopt import BayesSearchCV
warnings.filterwarnings('ignore')

def normalize_data(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    normalized_data = 2 * (data - min_values) / (max_values - min_values) - 1
    return normalized_data, min_values, max_values

def true_pred_plot(y_test, y_test_pred, target_variable, font_size=12):

    # Create plots for each target variable
    x = np.array(y_test)
    y = np.array(y_test_pred)  # Adjust for 0-based indexing

    # Create scatter plot with regression line
    plt.figure(figsize=(8, 8))
    sns.regplot(x=x, y=y, line_kws={'color': 'black'})

    # Add a diagonal reference line
    plt.plot([min(x), max(x)], [min(x), max(x)], linestyle='--', color='gray', linewidth=2)

    plt.title(f'True vs Predicted Values for {target_variable}', fontsize=font_size)
    plt.xlabel(f'True Value ({target_variable})', fontsize=font_size)
    plt.ylabel(f'Predicted Value ({target_variable})', fontsize=font_size)

    # Calculate R2 score
    R2 = r2_score(x, y)

    # Annotate R2 score with adjusted font size
    plt.annotate(f'R2 = {R2:.3f}', xy=(0.1, .93), xycoords='axes fraction',
                    ha='left', va='center', bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}, fontsize=font_size)

    # Set font size for tick labels
    plt.xticks(fontsize=font_size, rotation=45)
    plt.yticks(fontsize=font_size)

    # Show the plot
    plt.show()
    
def calculate_aare(actual, predicted):
    absolute_relative_errors = np.abs((actual - predicted) / actual)
    aare = np.mean(absolute_relative_errors)
    return aare

def calculate_aard(actual, predicted):
    absolute_differences = np.abs(actual - predicted)
    relative_differences = np.abs(absolute_differences / actual)
    AARD = np.mean(relative_differences)
    return AARD

def absolute_relative_deviation(observed_values, reference_values):
    total_deviation_percentage = 0
    n = len(observed_values)
    for observed, reference in zip(observed_values, reference_values):
        deviation_percentage = abs(observed - reference) / reference * 100
        total_deviation_percentage += deviation_percentage
    average_deviation_percentage = total_deviation_percentage / n
    return average_deviation_percentage

def get_data_correlation(data):
    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"fontweight": "bold"})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Correlation Matrix', fontweight='bold')
    plt.show()
    
def data_descriptio(data):
    pprint.pprint(data.dtypes)
    pprint.pprint(data.describe())
    
def train_model(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # Print cross-validation scores using R^2 score
    print("Cross-validation scores (R^2):", cv_scores)
    print("Mean CV Score (R^2):", np.mean(cv_scores))
    print("STD CV Score (R^2):", np.std(cv_scores, axis=0))

    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test, y_test):
    model_prediction = model.predict(X_test)

    print("R^2 IFT:", r2_score(y_test, model_prediction))
    print("MAE IFT:",mean_absolute_error(y_test,model_prediction))
    print("MSE IFT:",root_mean_squared_error(y_test,model_prediction)** 2)
    print("RMSE IFT:",root_mean_squared_error(y_test,model_prediction))
    print("AARE IFT:", calculate_aare(y_test, model_prediction))
    return model_prediction, r2_score(y_test, model_prediction)

def fea(model,features, model_name = '', fontsize = 12):
    
    if model_name == "ann":
        input_layer_weights = model.coefs_[0]
        feature_importance = np.abs(input_layer_weights).mean(axis=1)
        total_importance = np.sum(feature_importance)
        normalized_importance = feature_importance / total_importance
        feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': normalized_importance})
    else:
        feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importances_df, x='Importance', y='Feature', palette='viridis')
    plt.xlabel('Feature Importance', fontsize=fontsize, fontweight='bold')
    plt.ylabel('Features', fontsize=fontsize, fontweight='bold')
    plt.title('Feature Importances', fontsize=fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize, fontweight='bold')
    plt.show()

def standardize_data(data):
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    standardized_data = (data - mean_values) / std_values
    return standardized_data

def TSNE_viz(data_one, data_two, data_zone=False):
    if len(data_two) > 0:
        # Concatenate the two datasets along axis 0
        combined_data = np.concatenate((data_one, data_two), axis=0)
    else:
        combined_data = data_one

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(combined_data)

    # Create labels for the datasets
    one_hot_salt_labels = ['Simulated'] * len(data_one)
    experimental_labels = ['Experimental'] * len(data_two)
    all_labels = np.array(one_hot_salt_labels + experimental_labels)

    # Create a DataFrame for seaborn
    T_SNE_result_DF = pd.DataFrame({
        't-SNE1': tsne_result[:, 0],
        't-SNE2': tsne_result[:, 1],
        'Dataset': all_labels
    })

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=T_SNE_result_DF, x='t-SNE1', y='t-SNE2', hue='Dataset', palette=['blue', 'red'])
    plt.title('t-SNE Visualization of Datasets', fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    if data_zone:
        # ellipse = Ellipse(xy=(5, 1), width=40, height=25, angle=0, edgecolor='green', fc='None', lw=2)
        # plt.gca().add_patch(ellipse)
        rectangle = Rectangle(xy=(-15, -10), width=40, height=20, angle=0, edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(rectangle)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize= 12)
    # plt.grid(True)
    plt.show()
    
    return T_SNE_result_DF

def denormalize_data(normalized_data, min_values, max_values):
    denormalized_data = (normalized_data + 1) * (max_values - min_values) / 2 + min_values
    return denormalized_data

def cal_average_percentage_error(predicted_value, original_value):
    percentage_errors = []

    for pred, orig in zip(predicted_value, original_value):
        loss_percentage = (abs(orig - pred) / orig) * 100
        percentage_errors.append(loss_percentage)

    average_percentage_error = sum(percentage_errors) / len(percentage_errors)
    std_dev_percentage_error = np.std(percentage_errors)

    return average_percentage_error, std_dev_percentage_error

def analyze_column_bins_and_errors(dataframe, columns_to_analyze, num_bins):
    """
    Analyzes columns in a DataFrame by binning their values and calculating average error percentages for each bin.
    
    Args:
    - dataframe: DataFrame containing the data to be analyzed
    - columns_to_analyze: List of column names to be analyzed
    - num_bins: Number of bins to create for each column
    
    Returns:
    - average_errors: A dictionary containing average error percentages for each column and bin
    """
    
    average_errors = {}
    
    # Iterate over each column
    for column in columns_to_analyze:
        # Create bins based on the current column
        dataframe[f'{column}_bins'] = pd.cut(dataframe[column], bins=num_bins)

        # Initialize a dictionary to store the errors for the current column
        column_errors = {}

        # Iterate over each bin in the current column
        for bin_label, group in dataframe.groupby(f'{column}_bins'):
            # Calculate the errors if the group is not empty
            if len(group[column]) > 0:
                errors = (abs(group['IFT'] - group['predicted_IFT']) / group['IFT']) * 100
                column_errors[bin_label] = np.mean(errors)

        # Store the errors for the current column
        average_errors[column] = column_errors

    return average_errors

def plot_prediction_interval(DF):
    z = norm.ppf(0.975)  # 95% confidence level
    residuals = DF['IFT'] - DF['predicted_IFT']
    model_var = np.var(residuals)

    prediction_interval_lower = DF['predicted_IFT'] - z * np.sqrt(model_var)
    prediction_interval_upper = DF['predicted_IFT'] + z * np.sqrt(model_var)

    plt.scatter(DF['predicted_IFT'], DF['IFT'], color='blue', label='Actual Values')
    plt.plot(DF['predicted_IFT'], DF['predicted_IFT'], color='red', label='Predicted Values')
    plt.fill_between(DF['predicted_IFT'], 
                     prediction_interval_lower, 
                     prediction_interval_upper, 
                     color='green', 
                     alpha=0.3, 
                     label='Prediction Interval')

    # Add labels and legend
    plt.xlabel('Predicted IFT')
    plt.ylabel('Actual IFT')
    plt.title('Prediction vs Actual IFT with Prediction Interval')
    plt.legend()

    # Show plot
    plt.show()
    
def plot_distribution_comparison(DF):
    # Set style
    sns.set_style('whitegrid')

    # Visualize distribution comparison
    sns.kdeplot(DF['IFT'], label='IFT', color='blue', linestyle='-', linewidth=2)
    sns.kdeplot(DF['predicted_IFT'], label='predicted_IFT', color='red', linestyle='--', linewidth=2)

    # Add title and labels
    plt.title('Distribution Comparison of IFT and Predicted IFT', fontsize=14, fontweight='bold')
    plt.xlabel('IFT Value', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add legend
    plt.legend(fontsize=14)

    # Show plot
    plt.show()
    
def plot_error_distribution(DF, error_brackets):
    DF['error'] = abs(DF['IFT'] - DF['predicted_IFT'])
    DF['error_percentage'] = (DF['error'] / DF['IFT']) * 100

    # Count data points in each error bracket
    error_counts = pd.cut(DF['error_percentage'], bins=error_brackets).value_counts()

    # Plotting
    plt.bar(error_counts.index.astype(str), error_counts.values, color='skyblue')
    plt.xlabel('Error Brackets', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Error Distribution', fontweight='bold')
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    for i, count in enumerate(error_counts):
        plt.text(i, count + 0.2, str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.ylim(0, max(error_counts.values) + 5)  # Adjust ylim for better visualization
    plt.show()
    return error_counts

def plot_model_confidence(scores):
    mean_accuracy, std_dev = scores.mean(), scores.std()
    lower_bound = mean_accuracy - (2 * std_dev)
    upper_bound = mean_accuracy + (2 * std_dev)

    # Create the box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(scores, vert=False, patch_artist=True, notch=True, medianprops={"linewidth": 2})

    # Add labels and title
    plt.xlabel("Accuracy", fontsize=20)
    plt.ylabel("Model Performance", fontsize=20)
    plt.title("Model Cross-Validated Performance (Avg: {:.2f}%)".format(mean_accuracy * 100), fontsize=20)

    # Plot Confidence Interval
    plt.axvline(x=lower_bound, color='red', linestyle='--', label='Conf. Interval (97%)')
    plt.axvline(x=upper_bound, color='red', linestyle='--')
    plt.text(lower_bound + 0.0005, 1, "{:.2f}%".format(lower_bound * 100), color='red', fontsize=20, verticalalignment='center', rotation=90)
    plt.text(upper_bound - 0.0025, 1, "{:.2f}%".format(upper_bound * 100), color='red', fontsize=20, verticalalignment='center', rotation=90)

    # Customizations
    plt.yticks(rotation=0, fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0.94, 1.0)  # Adjust x-axis limits to better fit the data range
    plt.legend(fontsize=20)
    plt.tight_layout()

    # Display the graph
    plt.show()
    

def plot_model_losses(models, metrics, values):
    models_new = [model.replace(' ', '\n') for model in models]  # Adjust model names for better visualization
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.2
    index = range(len(models))

    for i, metric in enumerate(metrics):
        ax.bar([x + i * bar_width for x in index], [values[model][i] for model in models], bar_width, label=metric)
        for j, value in enumerate([values[model][i] for model in models]):
            ax.text(j + i * bar_width - 0.05, value / 2, f'{value:.2f}', fontsize=14, fontweight='bold', rotation=90)

    ax.set_xlabel('Models', fontsize=18, fontweight='bold')
    ax.set_ylabel('Value', fontsize=18, fontweight='bold')
    ax.set_title('Performance Metrics for Different Models', fontsize=18, fontweight='bold')
    ax.set_xticks([i + 1 * bar_width for i in index])
    ax.set_xticklabels(models_new, fontsize=14, rotation=45)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

def optimize_model_RS(base_model, param_dist, X_train, y_train):
    # Perform randomized search cross-validation
    random_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist,
                                       n_iter=10, scoring='neg_mean_squared_error',
                                       cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_
    print("Best parameters found by RandomizedSearchCV: \n", best_params)
    return best_params

def optimize_model_GS(base_model, param_dist, X_train, y_train):
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_dist,
                               scoring='neg_mean_squared_error',
                               cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best parameters found by GridSearchCV:\n", best_params)
    return best_params

def optimize_model_BYS(base_model, param_dist, X_train, y_train):
    # Perform Bayesian optimization
    bayes_search = BayesSearchCV(estimator=base_model, search_spaces=param_dist,
                                 n_iter=32, scoring='neg_mean_squared_error',
                                 cv=5, n_jobs=-1, random_state=42)
    bayes_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    print("Best parameters found by BayesSearchCV:\n")
    print(best_params, "Best Score:", best_score)
    return best_params

def true_pred_plot_subplots(y_test, predictions, model_names, target_variable, font_size=12):
    """
    Plot True vs Predicted values for multiple models in separate subplots.

    Parameters:
        y_test: array-like
            True values of the target variable.
        predictions: list of array-like
            Predicted values from multiple models.
        model_names: list of str
            Names of the models for labeling.
        target_variable: str
            Name of the target variable.
        font_size: int, optional
            Font size for plot labels and annotations.
    """
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(15, 5), sharex=True, sharey=True)

    y_test = np.array(y_test).ravel()

    for i, (y_pred, model_name, ax) in enumerate(zip(predictions, model_names, axes)):
        y_pred = np.array(y_pred).ravel()

        # Scatter plot
        sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7)
        
        # Regression line
        sns.regplot(x=y_test, y=y_pred, ax=ax, scatter=False, color='black', line_kws={'linewidth': 1})

        # Add diagonal reference line
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', linewidth=2)

        # Title and labels
        ax.set_title(f"{model_name} - {target_variable}", fontsize=font_size)
        ax.set_xlabel(f"True Value ({target_variable})", fontsize=font_size)
        ax.set_ylabel(f"Predicted Value ({target_variable})", fontsize=font_size)

        # Calculate R2 score
        R2 = r2_score(y_test, y_pred)
        ax.annotate(f'R2 = {R2:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',
                    ha='left', va='center', fontsize=font_size,
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
    
    plt.tight_layout()
    plt.show()
    
def true_pred_plot_multiple(y_test, predictions, model_names, target_variable, font_size=12):
    """
    Plot True vs Predicted values for multiple models on the same figure.

    Parameters:
        y_test: array-like
            True values of the target variable.
        predictions: list of array-like
            Predicted values from multiple models.
        model_names: list of str
            Names of the models for labeling.
        target_variable: str
            Name of the target variable.
        font_size: int, optional
            Font size for plot labels and annotations.
    """
    plt.figure(figsize=(10, 10))
    
    # Ensure y_test is a numpy array
    y_test = np.array(y_test).ravel()
    print(y_test.shape)
    
    colors = ['blue', 'green', 'red']  # Colors for each model
    
    for i, (y_pred, model_name, color) in enumerate(zip(predictions, model_names, colors)):
        y_pred = np.array(y_pred)
        
        # Scatter plot
        sns.scatterplot(x=y_test, y=y_pred, label=f"{model_name}", color=color, alpha=0.7)
        
        # Add regression line for each model
        sns.regplot(x=y_test, y=y_pred, scatter=False, color=color, line_kws={'label': f'{model_name} fit', 'lw': 1})
        
        # Calculate R2 score
        R2 = r2_score(y_test, y_pred)
        plt.annotate(f'{model_name} R2 = {R2:.3f}', 
                     xy=(0.1, 0.93 - i * 0.05), xycoords='axes fraction',
                     ha='left', va='center', fontsize=font_size,
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

    # Add diagonal reference line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', linewidth=2, label='Ideal')

    plt.title(f'True vs Predicted Values for {target_variable}', fontsize=font_size + 2)
    plt.xlabel(f'True Value ({target_variable})', fontsize=font_size)
    plt.ylabel(f'Predicted Value ({target_variable})', fontsize=font_size)
    
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    
    # Show the plot
    plt.show()