"""LangChain tools: typed requests in, verified facts and chart artifacts out."""
import json
from typing import Literal
from langchain.tools import tool, ToolRuntime
from core import analytics


def output(artifact):
    return json.dumps(artifact['facts'], allow_nan=False, default=str), artifact


@tool(response_format='content_and_artifact')
def analyze_relationship(x: str, y: str, runtime: ToolRuntime):
    """Calculate the relationship between two numeric columns and supply an interactive scatter plot.

    Use for questions about relationships, associations or correlations. Returns complete-pair counts,
    exclusions, Pearson r, means, and an automatic chart. It does not test causality or significance.
    """
    return output(analytics.relationship(runtime.state['frame'], x, y))


@tool(response_format='content_and_artifact')
def explain_distribution(column: str, runtime: ToolRuntime,
                         chart_type: Literal['histogram', 'pie', 'bar', 'box', 'none'] = 'histogram', bins: int = 8):
    """Summarize a column using computed statistics and an optional interactive distribution chart.

    Numeric pie charts use explicitly labeled equal-width ranges, not one slice per measurement.
    For a numeric lookup such as mean or median choose chart_type='none'.
    """
    return output(analytics.distribution(runtime.state['frame'], column, chart_type, bins))


@tool(response_format='content_and_artifact')
def compare_groups(category: str, runtime: ToolRuntime, measurement: str | None = None,
                   aggregation: Literal['mean', 'median', 'count', 'sum', 'min', 'max'] = 'mean',
                   chart_type: Literal['bar', 'pie'] = 'bar'):
    """Compare a numeric measurement across categories with a table and interactive chart.

    For category counts choose aggregation='count', measurement=None. A pie requires counts;
    use bars to compare means. There can be at most 30 groups.
    """
    return output(analytics.compare_groups(runtime.state['frame'], category, measurement, aggregation, chart_type))


@tool(response_format='content_and_artifact')
def inspect_dataset(runtime: ToolRuntime):
    """Compute a dataset overview, numeric summaries and missing-value counts for teaching."""
    return output(analytics.overview(runtime.state['frame']))


@tool(response_format='content_and_artifact')
def query_table(sql: str, runtime: ToolRuntime, chart_type: Literal['none', 'bar', 'line', 'scatter', 'pie'] = 'none',
                x: str | None = None, y: str | None = None):
    """Run a single read-only DuckDB SELECT on the uploaded table named healthcare.

    Use for filters, time trends, grouped/binned relationships, counts and calculations beyond
    the dedicated tools. Quote column names with double quotes. No file/network access or writes.
    Aggregate/limit results to 2,000 rows. To visualize, give chart_type and x/y result column names.
    A pie must use meaningful parts of a whole, such as record counts, never mean measurements.
    """
    return output(analytics.sql_analysis(runtime.state['frame'], sql, chart_type, x, y))


TOOLS = [analyze_relationship, explain_distribution, compare_groups, inspect_dataset, query_table]
