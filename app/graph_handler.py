# app/graph_handler.py

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64


def parse_excel(file_path):
    """
    Parses the Excel file and returns a Pandas DataFrame.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error parsing Excel file: {e}")
        return None


def generate_matplotlib_graph(df, chart_type='line', x_col=None, y_col=None):
    """
    Generates a Matplotlib graph and returns it as a base64-encoded string.
    """
    plt.figure(figsize=(10, 5))
    if chart_type == 'line':
        plt.plot(df[x_col], df[y_col], marker='o')
    elif chart_type == 'bar':
        plt.bar(df[x_col], df[y_col])
    elif chart_type == 'scatter':
        plt.scatter(df[x_col], df[y_col])
    plt.title(f"{chart_type.capitalize()} Chart")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64


def generate_plotly_graph(df, chart_type='line', x_col=None, y_col=None):
    """
    Generates a Plotly graph and returns it as an HTML div.
    """
    if chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type.capitalize()} Chart")
    elif chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type.capitalize()} Chart")
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type.capitalize()} Chart")

    graph_html = fig.to_html(full_html=False)
    return graph_html
