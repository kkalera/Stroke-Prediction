import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    Normalizer,
    FunctionTransformer,
    MinMaxScaler,
)
from sklearn.impute import KNNImputer, SimpleImputer

COLOR_MAIN = "#69b3a2"
COLOR_CONTRAST = "#B3697A"


def get_cmap():
    """
    Returns a matplotlib colormap with a main color and a contrast color.

    Returns:
    matplotlib.colors.LinearSegmentedColormap: The matplotlib colormap.
    """
    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [
        [norm(-1.0), COLOR_CONTRAST],
        [norm(0.0), "#ffffff"],
        [norm(1.0), COLOR_MAIN],
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


def countplot(
    data: pd.DataFrame,
    column_name: str,
    title: str = "Countplot",
    hue: str = None,
    ax=None,
    figsize=(10, 5),
    bar_labels: bool = False,
    bar_label_kind: str = "percentage",
    horizontal: bool = False,
):
    """
    Generate a countplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the countplot. Defaults to "Countplot".
        hue (str, optional): The column name to use for grouping the countplot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        bar_labels (bool, optional): Whether to add labels to the bars. Defaults to False.
        bar_label_kind (str, optional): The kind of labels to add to the bars. Can be "percentage" or "count". Defaults to "percentage".

    Returns:
        matplotlib.axes.Axes: The axis object containing the countplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) if ax is None else (plt.gcf(), ax)

    if hue:
        if horizontal:
            sns.countplot(
                data=data,
                y=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=[COLOR_MAIN, COLOR_CONTRAST],
                hue=hue,
            )
        else:
            sns.countplot(
                data=data,
                x=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=[COLOR_MAIN, COLOR_CONTRAST],
                hue=hue,
            )
    else:
        if horizontal:
            sns.countplot(data=data, y=column_name, ax=ax, color=COLOR_MAIN)
        else:
            sns.countplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN)

    ## Add bar labels
    if bar_labels:
        for container in ax.containers:
            if bar_label_kind == "percentage":
                ax.bar_label(container, fmt=lambda x: f" {x / len(data):.1%}")
            else:
                ax.bar_label(container, fmt=lambda x: f" {x}")

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def boxplot(
    data: pd.DataFrame,
    column_name: str,
    title: str = "Boxplot",
    ax=None,
    figsize=(10, 5),
):
    """
    Create a boxplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to create the boxplot for.
        title (str, optional): The title of the boxplot. Defaults to "Boxplot".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the boxplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    sns.boxplot(
        data=data,
        y=column_name,
        ax=ax,
        color=COLOR_MAIN,
    )

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def histplot(
    data: pd.DataFrame,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 5),
    kde: bool = False,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.histplot(
            data=data,
            x=column_name,
            ax=ax,
            color=COLOR_MAIN,
            palette=palette,
            hue=hue,
            kde=kde,
        )
    else:
        sns.histplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN, kde=kde)

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def plot_distribution_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    histplot(
        data=data,
        column_name=column_name,
        title="",
        ax=ax[0],
    )
    boxplot(
        data=data,
        column_name=column_name,
        title="",
        ax=ax[1],
    )
    fig.suptitle(title, fontsize=16)


def plot_distribution_and_ratio(
    data,
    ratio: pd.Series,
    column_name: str,
    hue: str,
    title: str = "Distribution and Ratio",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    horizontal: bool = False,
    label_rotation: int = 0,
):
    """
    Plot the distribution and ratio of a categorical variable.

    Parameters:
    - data: The DataFrame containing the data.
    - ratio: The ratio of the categories.
    - column_name: The name of the categorical column.
    - hue: The column to use for grouping the data.
    - title: The title of the plot (default: "Distribution and Ratio").
    - ax: The matplotlib axes object to plot on (default: None).
    - figsize: The figure size (default: (10, 5)).
    - width_ratios: The width ratios of the subplots (default: [3, 1.25]).
    - horizontal: Whether to plot the bars horizontally (default: False).
    - label_rotation: The rotation angle of the tick labels (default: 0).
    """
    fig, ax = plt.subplots(
        figsize=figsize, nrows=1, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    countplot(
        data=data,
        column_name=column_name,
        hue=hue,
        title="Distribution",
        bar_labels=True,
        ax=ax.flatten()[0],
        horizontal=horizontal,
    )
    if horizontal:
        sns.barplot(
            y=ratio.index,
            x=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    else:
        sns.barplot(
            x=ratio.index,
            y=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    ax[1].set_title("Ratio")

    if label_rotation:
        if horizontal:
            for t1, t2 in zip(ax[0].get_yticklabels(), ax[1].get_yticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)
        else:
            for t1, t2 in zip(ax[0].get_xticklabels(), ax[1].get_xticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)


def correlation_matrix(corr, title="Correlation Matrix"):
    """
    Plot a correlation matrix heatmap.

    Parameters:
    corr (numpy.ndarray): The correlation matrix.
    title (str): The title of the plot. Default is "Correlation Matrix".

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = get_cmap()

    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1, fmt=".2f")
    fig.suptitle(title, fontsize=16)


def get_correlations(data: pd.DataFrame):
    """
    Calculate the correlation matrix for the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The correlation matrix of the input DataFrame.
    """
    df_corr = data.copy()
    df_corr.drop(columns=["id", "bmi_class"], inplace=True, axis=1)
    df_corr["hypertension"] = df_corr["hypertension"].cat.rename_categories(
        {"No": False, "Yes": True}
    )
    df_corr["heart_disease"] = df_corr["heart_disease"].cat.rename_categories(
        {"No": False, "Yes": True}
    )
    df_corr["ever_married"] = df_corr["ever_married"].cat.rename_categories(
        {"No": False, "Yes": True}
    )
    df_corr["stroke"] = df_corr["stroke"].cat.rename_categories(
        {"No": False, "Yes": True}
    )
    df_dummies = pd.get_dummies(
        df_corr[["gender", "work_type", "Residence_type", "smoking_status"]]
    )
    df_corr = pd.concat([df_corr, df_dummies], axis=1)
    df_corr.drop(
        columns=["gender", "work_type", "Residence_type", "smoking_status"],
        axis=1,
        inplace=True,
    )
    correlations = df_corr.corr()
    correlations.drop(columns=df_dummies.columns, inplace=True, axis=0)
    return correlations


def prep_data(data: pd.DataFrame):
    """
    Preprocesses the given DataFrame by dropping unnecessary columns and filtering out invalid data.

    Args:
        data (pd.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    assert isinstance(data, pd.DataFrame)
    df = data.copy()
    df.drop(columns=["id"], inplace=True, axis=1)
    df.drop(df.loc[df["gender"] == "Other"].index, inplace=True)
    df.drop(
        df.loc[(df["age"] > 16) & (df["work_type"] == "Never_worked")].index,
        inplace=True,
        axis=0,
    )
    df.loc[df["work_type"] == "Never_worked", "work_type"] = "children"
    return df


def plot_shap_values(model, X, explainer=None, feature_names=None, plot_size=(7, 5)):
    """
    Plots the SHAP values for a given model and test data.

    Parameters:
    model (object): The trained model object.
    x_train (array-like): The training data used to train the model.
    x_test (array-like): The test data for which SHAP values will be computed and plotted.

    Returns:
    None
    """
    if explainer == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_size=plot_size, feature_names=feature_names)


def plot_roc_curve(model, x_test, y_test, ax=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(x_test))
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6)) if ax is None else plt.sca(ax)
    plt.plot(fpr, tpr, color=COLOR_MAIN, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color=COLOR_CONTRAST, lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")


def test_model(model, x_train, x_test, y_train, y_test, logger):
    """
    Evaluate the performance of a machine learning model on training and test data.

    Args:
        model: The trained machine learning model.
        x_train: The input features of the training data.
        x_test: The input features of the test data.
        y_train: The target labels of the training data.
        y_test: The target labels of the test data.
        logger: The logger object for logging the evaluation results.

    Returns:
        None
    """
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    logger.info(f"Training scores:\n")
    logger.info(f"    - Accuracy: {accuracy_score(y_train, pred_train):.3f}")
    logger.info(
        "    - Precision:"
        f" {precision_score(y_train, pred_train, average='binary', zero_division=0):.3f}"
    )
    logger.info(
        f"    - Recall: {recall_score(y_train, pred_train, average='binary'):.3f}\n"
    )

    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test, average="binary", zero_division=0)
    rec = recall_score(y_test, pred_test, average="binary")

    logger.info(f"Test scores:\n")
    logger.info(f"    - Accuracy: {acc:.3f}")
    logger.info(f"    - Precision: {prec:.3f}")
    logger.info(f"    - Recall: {rec:.3f}")


def plot_roc_and_confusion_matrix(model, X_test, y_test):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and the confusion matrix for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        figsize=(10, 5), nrows=1, ncols=2, gridspec_kw={"width_ratios": [3, 2]}
    )
    plt.grid(False)
    plot_roc_curve(model, X_test, y_test, ax=ax[0])
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels={0: "no stroke", 1: "stroke"}
    ).plot(include_values=True, cmap="Blues", ax=ax.flatten()[1])


def get_pipeline(model):
    """
    Create a data preprocessing pipeline for a given model.

    Parameters:
    model (object): The machine learning model to be used in the pipeline.

    Returns:
    pipeline (object): The data preprocessing pipeline.

    """
    # Do binary encoding for the binary categories
    binary_columns = ["gender", "ever_married", "Residence_type", "hypertension"]
    binary_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("binary", OrdinalEncoder()),
        ]
    )

    # Do one-hot encoding for the categorical categories
    categorical_columns = ["work_type", "smoking_status"]
    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="error")),
        ]
    )

    # Do standard scaling for the numerical columns
    numerical_columns = ["age", "bmi", "avg_glucose_level"]
    numerical_transformer = Pipeline(
        steps=[
            ("impute", KNNImputer()),
            ("scale", StandardScaler()),
            ("normalize", Normalizer()),
        ]
    )

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_columns),
            ("categorical", categorical_transformer, categorical_columns),
            ("numerical", numerical_transformer, numerical_columns),
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def get_reduced_pipeline(model):
    """
    Create a reduced pipeline for machine learning.

    Parameters:
    - model: The machine learning model to be used in the pipeline.

    Returns:
    - pipeline: The reduced pipeline with preprocessing and the specified model.
    """
    # Do standard scaling for the numerical columns
    numerical_columns = ["age", "bmi", "avg_glucose_level"]
    numerical_transformer = Pipeline(
        steps=[
            ("impute", KNNImputer()),
            ("scale", StandardScaler()),
            ("normalize", Normalizer()),
        ]
    )

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def calculate_outlier_range(data):
    """
    Calculate the lower and upper bounds for identifying outliers using the IQR method.

    Parameters:
    data (pandas.Series): The data for which to calculate the outlier range.

    Returns:
    tuple: A tuple containing the lower and upper bounds for identifying outliers.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return (lower_bound, upper_bound)
