import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import io
import base64
import logging
from typing import Optional, Dict, Union, List

# Set up logging
logger = logging.getLogger(__name__)


class GraphHandler:
    """Handles generation and management of various types of graphs and visualizations."""

    def __init__(self):
        self.SUPPORTED_CHART_TYPES = {
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
            'waterfall': 'Waterfall Chart',

            # RAG-specific Charts
            'embedding_scatter': 'Document Embedding Visualization',
            'similarity_heatmap': 'Document Similarity Heatmap',
            'relevance_bar': 'Document Relevance Scores',
            'source_network': 'Source Document Network',
            'topic_clusters': 'Topic Clustering Visualization'
        }

    def get_available_charts(self) -> Dict[str, str]:
        """Return dictionary of available chart types."""
        return self.SUPPORTED_CHART_TYPES

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
                    template="plotly_dark",
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

    @staticmethod
    def generate_rag_visualization(

            embeddings: np.ndarray,
            texts: List[str],
            query_embedding: Optional[np.ndarray] = None,
            relevance_scores: Optional[List[float]] = None,
            title: str = "Document Embeddings Visualization"
    ) -> Optional[str]:
        """Generate visualization of document embeddings with RAG."""
        try:
            # Reduce dimensionality for visualization
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)

            # Create the base scatter plot
            fig = go.Figure()

            # Add document points
            fig.add_trace(go.Scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=relevance_scores if relevance_scores else None,
                    colorscale='Viridis',
                    showscale=True if relevance_scores else False,
                    colorbar=dict(title="Relevance Score") if relevance_scores else None
                ),
                text=texts,
                hovertemplate="<b>Text:</b> %{text}<br>" +
                              "<b>Relevance:</b> %{marker.color:.2f}<extra></extra>" if relevance_scores else None,
                name='Documents'
            ))

            # Add query point if provided
            if query_embedding is not None:
                query_reduced = tsne.fit_transform(query_embedding.reshape(1, -1))
                fig.add_trace(go.Scatter(
                    x=[query_reduced[0, 0]],
                    y=[query_reduced[0, 1]],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star'
                    ),
                    text=['Query'],
                    name='Query'
                ))

            # Update layout
            fig.update_layout(
                title=title,
                template='plotly_dark',
                showlegend=True,
                hovermode='closest',
                height=600,
                width=800
            )

            return fig.to_html(
                full_html=False,
                include_plotlyjs=True,
                config={'responsive': True}
            )

        except Exception as e:
            logger.error(f"Error generating RAG visualization: {str(e)}")
            return None

    def generate_rag_analysis_summary(
            self,
            df: pd.DataFrame,
            rag_results: Dict,
            query: str
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Generate comprehensive analysis summary with RAG results."""
        try:
            summary = {
                'visualizations': [],
                'insights': [],
                'sources': []
            }

            # Generate standard visualizations
            standard_viz = self.generate_visualizations(df)
            if standard_viz:
                summary['visualizations'].extend(standard_viz)

            # Add RAG-specific visualizations
            if 'embeddings' in rag_results:
                rag_viz = self.generate_rag_visualization(
                    embeddings=rag_results['embeddings'],
                    texts=rag_results.get('texts', []),
                    relevance_scores=rag_results.get('relevance_scores'),
                    title=f"Document Embeddings for Query: {query[:30]}..."
                )
                if rag_viz:
                    summary['visualizations'].append({
                        'type': 'rag_embedding',
                        'plot': rag_viz,
                        'description': 'Document embedding visualization with relevance to query'
                    })

            # Add similarity heatmap if available
            if 'similarity_matrix' in rag_results:
                fig = px.imshow(
                    rag_results['similarity_matrix'],
                    title='Document Similarity Heatmap',
                    labels=dict(color="Similarity Score")
                )
                fig.update_layout(template='plotly_dark')
                summary['visualizations'].append({
                    'type': 'similarity_heatmap',
                    'plot': fig.to_html(full_html=False, include_plotlyjs=True),
                    'description': 'Similarity scores between documents'
                })

            # Add source documents with relevance scores
            if 'sources' in rag_results:
                summary['sources'] = [{
                    'text': source.get('text', ''),
                    'relevance': source.get('relevance', 0),
                    'metadata': source.get('metadata', {})
                } for source in rag_results['sources']]

            # Generate insights visualization
            if 'answer' in rag_results:
                fig = go.Figure()
                fig.add_annotation(
                    text=rag_results['answer'],
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(
                    template='plotly_dark',
                    height=200,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                summary['insights'].append({
                    'type': 'rag_answer',
                    'content': fig.to_html(full_html=False, include_plotlyjs=True),
                    'description': 'RAG Analysis Answer'
                })

            return summary

        except Exception as e:
            logger.error(f"Error generating RAG analysis summary: {str(e)}")
            return {'visualizations': [], 'insights': [], 'sources': []}

    @staticmethod
    def generate_visualizations(df: pd.DataFrame) -> list:
        """Generate automatic visualizations based on DataFrame content."""
        try:
            visualizations = []

            # Numeric column analysis
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Distribution plots for numeric columns
                for col in numeric_cols[:2]:  # Limit to first 2 columns
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                    fig.update_layout(template='plotly_dark')
                    visualizations.append({
                        'type': 'distribution',
                        'plot': fig.to_html(full_html=False, include_plotlyjs=True),
                        'description': f'Distribution of {col}'
                    })

                # Correlation matrix if multiple numeric columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title='Correlation Matrix',
                        color_continuous_scale='RdBu'
                    )
                    fig.update_layout(template='plotly_dark')
                    visualizations.append({
                        'type': 'correlation',
                        'plot': fig.to_html(full_html=False, include_plotlyjs=True),
                        'description': 'Correlation matrix between numeric variables'
                    })

                    # Time series analysis if date/time columns exist
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0 and len(numeric_cols) > 0:
                    date_col = date_cols[0]
                    for num_col in numeric_cols[:2]:
                        fig = px.line(
                            df,
                            x=date_col,
                            y=num_col,
                            title=f'{num_col} over time'
                        )
                        fig.update_layout(template='plotly_dark')
                        visualizations.append({
                            'type': 'time_series',
                            'plot': fig.to_html(full_html=False, include_plotlyjs=True),
                            'description': f'Time series of {num_col}'
                        })

                # Categorical analysis
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols[:2]:
                        value_counts = df[col].value_counts()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f'Distribution of {col}'
                        )
                        fig.update_layout(template='plotly_dark')
                        visualizations.append({
                            'type': 'categorical',
                            'plot': fig.to_html(full_html=False, include_plotlyjs=True),
                            'description': f'Distribution of {col} categories'
                        })

            return visualizations

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return []

    def get_rag_supported_visualizations(self) -> Dict[str, str]:
        """Get RAG-specific visualization types."""
        rag_charts = {
            'embedding_scatter': 'Document Embedding Visualization',
            'similarity_heatmap': 'Document Similarity Heatmap',
            'relevance_bar': 'Document Relevance Scores',
            'source_network': 'Source Document Network',
            'topic_clusters': 'Topic Clustering Visualization'
        }
        return {**self.SUPPORTED_CHART_TYPES, **rag_charts}

    @staticmethod
    def generate_matplotlib_graph(

            df: pd.DataFrame,
            chart_type: str,
            x_col: str,
            y_col: str,
            **kwargs
    ) -> Optional[str]:
        """Generate a static Matplotlib graph."""
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

    @staticmethod
    def parse_excel(filepath: str) -> pd.DataFrame:
        """Parse Excel or CSV file into a pandas DataFrame."""
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


# Create the singleton instance
_graph_handler = GraphHandler()

# Export the singleton instance
def get_graph_handler() -> GraphHandler:
    """Get the singleton instance of GraphHandler."""
    return _graph_handler

# Helper functions that use the singleton
def get_available_charts() -> List[Dict[str, str]]:
    """Return list of available chart types."""
    handler = get_graph_handler()
    return [
        {'value': chart_type, 'label': label}
        for chart_type, label in handler.SUPPORTED_CHART_TYPES.items()
    ]

def generate_plotly_graph(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: Optional[str] = None,
        title: str = "",
        theme: str = 'dark'
) -> Optional[str]:
    """Generate a Plotly graph based on the specified parameters."""
    handler = get_graph_handler()
    return handler.generate_plotly_graph(
        df=df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        title=title,
        theme=theme
    )

def generate_matplotlib_graph(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str,
        **kwargs
) -> Optional[str]:
    """Generate a Matplotlib graph."""
    handler = get_graph_handler()
    return handler.generate_matplotlib_graph(
        df=df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        **kwargs
    )

def generate_rag_visualization(
        embeddings: np.ndarray,
        texts: List[str],
        query_embedding: Optional[np.ndarray] = None,
        relevance_scores: Optional[List[float]] = None,
        title: str = "Document Embeddings Visualization"
) -> Optional[str]:
    """Generate RAG visualization."""
    handler = get_graph_handler()
    return handler.generate_rag_visualization(
        embeddings=embeddings,
        texts=texts,
        query_embedding=query_embedding,
        relevance_scores=relevance_scores,
        title=title
    )

def generate_visualizations(df: pd.DataFrame) -> list:
    """Generate automatic visualizations based on DataFrame content."""
    handler = get_graph_handler()
    return handler.generate_visualizations(df)

def parse_excel(filepath: str) -> pd.DataFrame:
    """Parse Excel or CSV file into a pandas DataFrame."""
    handler = get_graph_handler()
    return handler.parse_excel(filepath)

# Export all the public components
__all__ = [
    'get_graph_handler',
    'get_available_charts',
    'generate_plotly_graph',
    'generate_matplotlib_graph',  # Added this
    'generate_rag_visualization', # Added this
    'generate_visualizations',
    'parse_excel',
    'GraphHandler'
]