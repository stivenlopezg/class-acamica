import shap
import numpy as np
import pandas as pd
import seaborn as sns
from pdpbox import pdp
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize': (12, 8)})


def barplot(data: pd.DataFrame, column: str):
    """
    Take a dataframe and graph the barchart of the variable of interest
    :param data: dataframe
    :param column: variable of interest
    :return:
    """
    aux = data[column].value_counts().sort_values(ascending=True)
    bars = tuple(aux.index.tolist())
    values = aux.values.tolist()
    y_pos = np.arange(len(bars))
    colors = ['lightblue'] * len(bars)
    colors[-1] = 'blue'
    plt.figure(figsize=(12, 8))
    plt.barh(y_pos, values, color=colors)
    plt.title(f'{column} bar chart')
    plt.yticks(y_pos, bars)
    return plt.show()


def histogram(data: pd.DataFrame, column: str, **kwargs):
    """

    :param data:
    :param column:
    :param kwargs:
    :return:
    """
    return sns.distplot(a=data[column], **kwargs)


def boxplot(data: pd.DataFrame, column: str, label: str = None, **kwargs):
    """

    :param data:
    :param column:
    :param label:
    :param kwargs:
    :return:
    """
    if label is None:
        return sns.catplot(data=data, y=column, kind='box', height=8, aspect=10.4/8, **kwargs)
    else:
        return sns.catplot(data=data, x=column, y=label, kind='box', height=8, aspect=10.4/8, **kwargs)


def pointplot(data: pd.DataFrame, x: str, y: str, **kwargs):
    """

    :param data:
    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    return sns.catplot(data=data, kind='point', x=x, y=y, height=8, aspect=10.4/8, **kwargs)


def scatterplot(data: pd.DataFrame, x: str, y: str, **kwargs):
    """

    :param data:
    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    return sns.relplot(x=x, y=y, data=data, kind='scatter', **kwargs)


def plot_imputations(df: pd.DataFrame, column: str, method: str = 'median', **kwargs):
    plt.subplot(1, 2, 1)
    sns.distplot(df[column], **kwargs)
    plt.subplot(1, 2, 2)
    if method == 'median':
        sns.distplot(df[column].fillna(df[column].fillna(df[column].median())), **kwargs)
    elif method == 'mean':
        sns.distplot(df[column].fillna(df[column].fillna(df[column].mean())), **kwargs)
    else:
        sns.distplot(df[column].fillna(df[column].fillna(df[column].mode())), **kwargs)
    return plt.show()


def plot_feature_importance(model, data: pd.DataFrame, **kwargs):
    """
    Graph the importance of the variables
    :param model: model
    :param data: DataFrame
    :return:
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap.summary_plot(shap_values, data, plot_type='bar', plot_size=(14, 10), **kwargs)


def partial_dependence_plot(model, dataset: pd.DataFrame, model_features: list, objective: str, **kwargs):
    """

    :param model:
    :param dataset:
    :param model_features:
    :param objective:
    :return:
    """
    pdp_data = pdp.pdp_isolate(model=model, dataset=dataset, model_features=model_features, feature=objective)
    pdp.pdp_plot(pdp_data, objective, figsize=(10, 8), **kwargs)
    return plt.show()
