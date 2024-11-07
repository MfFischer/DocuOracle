# docuoracle_app/graph_handler.py

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import logging
from typing import Optional, Union, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class GraphHandler:
    SUPPORTED_CHART_TYPES = {
        # Basic Charts
        'line': 'Line Chart',
        'bar': 'Bar Chart',
        'scatter': 'Scatter Plot',
        'pie': 'Pie Chart',
        'donut': 'Donut Chart',
        'area': 'Area Chart',

        # Statistical Charts
        'histogram': 'Histogram',
        'box': 'Box Plot',
        'violin': 'Violin Plot',
        'density': 'Density Plot',

        # Comparison Charts
        'grouped_bar': 'Grouped Bar Chart',
        'stacked_bar': 'Stacked Bar Chart',
        'radar': 'Radar Chart',

        # Distribution Charts
        'heatmap': 'Heatmap',
        'bubble': 'Bubble Chart',
        'contour': 'Contour Plot',

        # Time Series
        'time_series': 'Time Series',
        'candlestick': 'Candlestick Chart',

        # Specialized Charts
        'treemap': 'Treemap',
        'sunburst': 'Sunburst Chart',
        'funnel': 'Funnel Chart',
        'waterfall': 'Waterfall Chart'
    }

    def generate_plotly_graph(
            self,
            df: pd.DataFrame,
            chart_type: str,
            x_col: str = None,
            y_col: str = None,
            title: str = None,
            **kwargs
    ) -> Optional[str]:
        """Generate an interactive Plotly graph based on chart type."""
        try:
            if chart_type not in self.SUPPORTED_CHART_TYPES:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            fig = None

            # Basic Charts
            if chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col,
                              title=title,
                              markers=True,
                              line_shape=kwargs.get('line_shape', 'linear'))

            elif chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col,
                             title=title,
                             color=kwargs.get('color_col'),
                             barmode=kwargs.get('barmode', 'relative'))

            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col,
                                 title=title,
                                 color=kwargs.get('color_col'),
                                 size=kwargs.get('size_col'),
                                 hover_data=kwargs.get('hover_data'))

            elif chart_type == 'pie':
                fig = px.pie(df, values=y_col, names=x_col,
                             title=title,
                             hole=0)

            elif chart_type == 'donut':
                fig = px.pie(df, values=y_col, names=x_col,
                             title=title,
                             hole=0.3)

            elif chart_type == 'area':
                fig = px.area(df, x=x_col, y=y_col,
                              title=title,
                              line_shape=kwargs.get('line_shape', 'linear'))

            # Statistical Charts
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=x_col,
                                   title=title,
                                   nbins=kwargs.get('nbins', 30),
                                   histnorm=kwargs.get('histnorm'))

            elif chart_type == 'box':
                fig = px.box(df, x=x_col, y=y_col,
                             title=title,
                             points=kwargs.get('points', 'outliers'))

            elif chart_type == 'violin':
                fig = px.violin(df, x=x_col, y=y_col,
                                title=title,
                                box=kwargs.get('box', True))

            # Comparison Charts
            elif chart_type == 'grouped_bar':
                fig = px.bar(df, x=x_col, y=y_col,
                             title=title,
                             color=kwargs.get('group_col'),
                             barmode='group')

            elif chart_type == 'stacked_bar':
                fig = px.bar(df, x=x_col, y=y_col,
                             title=title,
                             color=kwargs.get('stack_col'),
                             barmode='stack')

            elif chart_type == 'radar':
                fig = go.Figure(data=go.Scatterpolar(
                    r=df[y_col],
                    theta=df[x_col],
                    fill='toself'
                ))

            # Distribution Charts
            elif chart_type == 'heatmap':
                fig = px.imshow(df.pivot(
                    index=kwargs.get('y_axis'),
                    columns=kwargs.get('x_axis'),
                    values=kwargs.get('values')
                ))

            elif chart_type == 'bubble':
                fig = px.scatter(df, x=x_col, y=y_col,
                                 size=kwargs.get('size_col'),
                                 color=kwargs.get('color_col'),
                                 title=title)

            # Time Series
            elif chart_type == 'time_series':
                fig = px.line(df, x=x_col, y=y_col,
                              title=title,
                              markers=True)
                fig.update_xaxes(rangeslider_visible=True)

            elif chart_type == 'candlestick':
                fig = go.Figure(data=[go.Candlestick(
                    x=df[x_col],
                    open=df[kwargs.get('open_col')],
                    high=df[kwargs.get('high_col')],
                    low=df[kwargs.get('low_col')],
                    close=df[kwargs.get('close_col')]
                )])

            # Specialized Charts
            elif chart_type == 'treemap':
                fig = px.treemap(df,
                                 path=kwargs.get('path'),
                                 values=kwargs.get('values'),
                                 title=title)

            elif chart_type == 'sunburst':
                fig = px.sunburst(df,
                                  path=kwargs.get('path'),
                                  values=kwargs.get('values'),
                                  title=title)

            elif chart_type == 'funnel':
                fig = go.Figure(go.Funnel(
                    y=df[x_col],
                    x=df[y_col],
                ))

            elif chart_type == 'waterfall':
                fig = go.Figure(go.Waterfall(
                    name="Waterfall",
                    orientation="v",
                    measure=["relative"] * len(df),
                    x=df[x_col],
                    y=df[y_col],
                ))

            if fig:
                # Common layout updates
                fig.update_layout(
                    title=title or f"{self.SUPPORTED_CHART_TYPES[chart_type]}",
                    template="plotly_white",
                    height=kwargs.get('height', 600),
                    width=kwargs.get('width', 800),
                    showlegend=kwargs.get('showlegend', True),
                    hovermode=kwargs.get('hovermode', 'closest')
                )

                # Add hover templates if specified
                if kwargs.get('hover_template'):
                    fig.update_traces(hovertemplate=kwargs.get('hover_template'))

                # Add annotations if specified
                if kwargs.get('annotations'):
                    fig.update_layout(annotations=kwargs.get('annotations'))

                return fig.to_html(
                    full_html=False,
                    include_plotlyjs=True,
                    config={
                        'responsive': True,
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d']
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error generating Plotly graph: {str(e)}")
            return None

    def get_available_charts(self) -> Dict[str, str]:
        """Return dictionary of available chart types."""
        return self.SUPPORTED_CHART_TYPES


# Create singleton instance
graph_handler = GraphHandler()


def generate_plotly_graph(*args, **kwargs) -> Optional[str]:
    return graph_handler.generate_plotly_graph(*args, **kwargs)


def get_available_charts() -> Dict[str, str]:
    return graph_handler.get_available_charts()


def parse_excel(filepath: str) -> pd.DataFrame:
    """
    Parse Excel or CSV file into a pandas DataFrame

    Args:
        filepath (str): Path to the Excel or CSV file

    Returns:
        pd.DataFrame: Parsed data
    """
    try:
        if filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}")
    except Exception as e:
        logger.error(f"Error parsing file {filepath}: {str(e)}")
        raise


def generate_matplotlib_graph(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str,
        **kwargs
) -> Optional[str]:
    """
    Generate a static Matplotlib graph

    Args:
        df (pd.DataFrame): Input data
        chart_type (str): Type of chart to generate
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        **kwargs: Additional arguments for customization

    Returns:
        Optional[str]: Base64 encoded image string
    """
    try:
        plt.figure(figsize=(10, 6))

        if chart_type == 'line':
            plt.plot(df[x_col], df[y_col])
        elif chart_type == 'bar':
            plt.bar(df[x_col], df[y_col])
        elif chart_type == 'scatter':
            plt.scatter(df[x_col], df[y_col])
        else:
            raise ValueError(f"Unsupported chart type for Matplotlib: {chart_type}")

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(kwargs.get('title', f'{chart_type.title()} Chart'))

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(image_png).decode()

    except Exception as e:
        logger.error(f"Error generating Matplotlib graph: {str(e)}")
        return None


# Add this to your existing graph_handler.py
def generate_analysis_graphs(analysis_result: dict) -> dict:
    """Generate graphs from analysis results."""
    try:
        graphs = {}

        # Check if analysis result contains data for visualization
        if not isinstance(analysis_result, dict):
            return None

        # Process numerical data if available
        if 'data' in analysis_result and isinstance(analysis_result['data'], pd.DataFrame):
            df = analysis_result['data']

            # Generate different types of graphs based on data
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                # Time series if index is datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    graphs[f'{col}_timeseries'] = generate_plotly_graph(
                        df,
                        'time_series',
                        x_col=df.index.name or 'Date',
                        y_col=col,
                        title=f'{col} Over Time'
                    )

                # Distribution plot
                graphs[f'{col}_distribution'] = generate_plotly_graph(
                    df,
                    'histogram',
                    x_col=col,
                    title=f'Distribution of {col}'
                )

                # Box plot for numerical columns
                graphs[f'{col}_box'] = generate_plotly_graph(
                    df,
                    'box',
                    y_col=col,
                    title=f'Box Plot of {col}'
                )

        return graphs

    except Exception as e:
        logger.error(f"Error generating analysis graphs: {str(e)}")
        return None
