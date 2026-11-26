"""Composable calculations and chart tools. Results are owned by a dataset and chat."""
from dataclasses import dataclass
import json
from typing import Literal

import numpy as np
import pandas as pd
from langchain.tools import tool, ToolRuntime

from core import analytics
from core.profile import profile_frame
from core.query import query_frames


@dataclass
class ToolContext:
    frame: object
    workspace: object
    dataset_id: str
    chat_id: str


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save(runtime, artifact):
    context = runtime.context
    artifact = clean(artifact)
    key = context.workspace.put_artifact(context.dataset_id, context.chat_id, artifact)
    facts = artifact.get('facts', {})
    content = {'result_id': key, **facts}
    if artifact.get('table'):
        content['table_preview'] = {**artifact['table'], 'rows': artifact['table']['rows'][:20]}
    if artifact.get('charts'):
        content['chart_type'] = artifact['charts'][0]['type']
    return json.dumps(content, allow_nan=False, default=str), {'result_id': key}


def get_frame(runtime, source_result_id=None):
    if not source_result_id:
        return runtime.context.frame.copy()
    context = runtime.context
    artifact = context.workspace.artifact(source_result_id, context.dataset_id, context.chat_id)
    table = artifact.get('data') or artifact.get('table')
    if not table:
        raise ValueError('This result is not a table. Query or calculate a table first.')
    return pd.DataFrame(table['rows'], columns=table['columns'])


def filtered(frame, filters):
    if len(filters or []) > 10:
        raise ValueError('Use at most ten filters.')
    for condition in filters or []:
        column, op, value = condition.get('column'), condition.get('op'), condition.get('value')
        if column not in frame:
            raise ValueError('Filter column not found.')
        series = frame[column]
        if op == 'eq': mask = series == value
        elif op == 'ne': mask = series != value
        elif op == 'gt': mask = series > value
        elif op == 'gte': mask = series >= value
        elif op == 'lt': mask = series < value
        elif op == 'lte': mask = series <= value
        elif op == 'in' and isinstance(value, list): mask = series.isin(value)
        elif op == 'not_null': mask = series.notna()
        else: raise ValueError('Supported filter operations: eq, ne, gt, gte, lt, lte, in, not_null.')
        frame = frame.loc[mask.fillna(False)]
    return frame


@tool(response_format='content_and_artifact')
def inspect_dataset(runtime: ToolRuntime[ToolContext]):
    """Inspect column names, data types, record counts, missing cells and numeric summaries. No charts."""
    return save(runtime, analytics.overview(runtime.context.frame))


@tool(response_format='content_and_artifact')
def query_data(sql: str, runtime: ToolRuntime[ToolContext]):
    """Run one read-only DuckDB SELECT on healthcare. Filter/group/aggregate; quote column names.

    Returns a result_id and a table for separate chart tools. Limit/aggregate to 2,000 rows.
    No file/network access, no writes. SQL calculations must use actual dataset columns.
    """
    result = query_frames(sql, {'healthcare': runtime.context.frame})
    if len(result) > 2000:
        raise ValueError('Aggregate or LIMIT your result to 2,000 rows.')
    table = {'columns': list(result.columns), 'rows': json.loads(result.to_json(orient='values', date_format='iso'))}
    return save(runtime, {'facts': {'operation': 'query', 'result_rows': len(result), 'columns': list(result.columns),
                                  'preview_truncated': len(result) > 20}, 'table': table, 'data': table})


@tool(response_format='content_and_artifact')
def calculate_statistics(operation: Literal['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'variance',
                          'quantiles', 'frequencies', 'percentages', 'pearson', 'spearman'],
                         columns: list[str], runtime: ToolRuntime[ToolContext],
                         group_by: str | None = None, filters: list[dict] | None = None,
                         source_result_id: str | None = None):
    """Calculate statistics without drawing charts. Returns result_id, computed facts and a table.

    Choose operation explicitly. Numeric columns for summaries/correlations; one categorical column
    for frequencies/percentages. Pearson or Spearman with 2+ columns returns all pairwise correlations
    and complete-pair counts (at most 12 columns). Optional group_by compares summaries by category.
    Filters are {column, op, value}; ops eq/ne/gt/gte/lt/lte/in/not_null. Filters combine with AND.
    std/variance use sample denominator n-1. Quantiles are 0,25,50,75,100 percent.
    For charts call a chart tool separately. Does not perform regression, significance or causal tests.
    """
    frame = filtered(get_frame(runtime, source_result_id), filters)
    if not columns or len(columns) > 12 or len(set(columns)) != len(columns) or any(c not in frame for c in columns):
        raise ValueError('Choose 1 to 12 different existing columns.')
    if group_by is not None and group_by not in frame:
        raise ValueError('Grouping column not found.')
    if group_by in columns:
        raise ValueError('Choose a grouping column different from the numeric measurements.')
    if frame.empty:
        raise ValueError('No records match these filters.')
    facts = {'operation': 'statistics', 'statistic': operation, 'columns': columns, 'group_by': group_by,
             'filters': filters or [], 'rows_total': len(runtime.context.frame), 'rows_selected': len(frame)}
    if operation in ('pearson', 'spearman'):
        if len(columns) < 2 or group_by:
            raise ValueError('Correlations need 2+ numeric columns; use filters for group-specific correlations.')
        values = pd.DataFrame({c: analytics.numeric(frame, c) for c in columns})
        corr = values.corr(method=operation, min_periods=2)
        pairs = []
        for i, x in enumerate(columns):
            for y in columns[i + 1:]:
                pair = values[[x, y]].dropna()
                pairs.append({'x': x, 'y': y, 'r': clean(corr.loc[x, y]), 'rows_used': len(pair),
                              'rows_excluded': len(frame) - len(pair)})
        facts.update(method=operation, pairs=pairs, matrix=clean(corr.values.tolist()),
                     interpretation_limit='Descriptive pairwise association only. No adjustment for other variables, significance test or causality.')
        table = {'columns': ['x', 'y', 'correlation', 'complete_pairs', 'excluded'],
                 'rows': [[p['x'], p['y'], p['r'], p['rows_used'], p['rows_excluded']] for p in pairs]}
        # Preserve explicit evidence fields used by the existing consistency checks.
        if len(columns) == 2 and operation == 'pearson':
            pair_facts = analytics.relationship(frame, *columns)['facts']
            facts.update(pair_facts)
            facts['statistic'] = operation
        return save(runtime, {'facts': facts, 'table': table})
    if operation in ('frequencies', 'percentages'):
        if len(columns) != 1 or group_by:
            raise ValueError('Choose one categorical column; use query_data for grouped frequency tables.')
        counts = frame[columns[0]].dropna().astype(str).value_counts()
        if len(counts) > 2000:
            raise ValueError('Too many distinct values. Bin or group them with query_data first.')
        result = pd.DataFrame({columns[0]: counts.index, 'count': counts.values,
                               'percentage': (counts.values / counts.sum() * 100).round(4)})
        facts.update(rows_used=int(counts.sum()), rows_excluded=int(frame[columns[0]].isna().sum()))
    else:
        values = pd.DataFrame({c: analytics.numeric(frame, c) for c in columns})
        facts['missing_by_column'] = {c: int(values[c].isna().sum()) for c in columns}
        groups = [(None, values)]
        if group_by:
            values[group_by] = frame[group_by].astype('string').fillna('(missing)')
            groups = list(values.groupby(group_by, sort=False))
            if len(groups) > 100:
                raise ValueError('Use at most 100 groups.')
        rows = []
        for group, selected in groups:
            for column in columns:
                valid = selected[column].dropna()
                if operation == 'quantiles':
                    for q in (0, .25, .5, .75, 1):
                        rows.append([group, column, q, clean(valid.quantile(q)), len(valid), len(selected) - len(valid)])
                else:
                    method = 'var' if operation == 'variance' else operation
                    value = getattr(valid, method)() if len(valid) else None
                    rows.append([group, column, clean(value), len(valid), len(selected) - len(valid)])
        result = pd.DataFrame(rows, columns=['group', 'column', 'quantile', 'value', 'records_used', 'excluded']
                              if operation == 'quantiles' else ['group', 'column', 'value', 'records_used', 'excluded'])
    table = {'columns': list(result.columns), 'rows': json.loads(result.to_json(orient='values'))}
    return save(runtime, {'facts': facts, 'table': table, 'data': table})


def draw(runtime, kind, x=None, y=None, source_result_id=None, group_by=None, bins=8):
    frame = get_frame(runtime, source_result_id)
    if kind == 'heatmap':
        if not source_result_id:
            raise ValueError('Calculate correlations first and supply the result_id.')
        context = runtime.context
        prior = context.workspace.artifact(source_result_id, context.dataset_id, context.chat_id)
        facts = prior['facts']
        if facts.get('statistic') not in ('pearson', 'spearman'):
            raise ValueError('A heatmap requires a calculated correlation matrix.')
        labels = facts['columns']
        points = [[j, i, value] for i, row in enumerate(facts['matrix']) for j, value in enumerate(row) if value is not None]
        chart = {'type': 'heatmap', 'title': facts['statistic'].capitalize() + ' correlations',
                 'labels': labels, 'points': points, 'caption': 'Pairwise correlations; missing values are excluded per pair. No adjustment for other variables or causal inference.'}
        return save(runtime, {'facts': {'operation': 'chart', 'type': kind, 'source_result_id': source_result_id}, 'charts': [chart]})
    if x not in frame or (y is not None and y not in frame):
        raise ValueError('Choose existing source columns for x and y.')
    if kind in ('histogram', 'box') or (kind == 'pie' and y is None):
        artifact = analytics.distribution(frame, x, kind, bins)
        chart = artifact.get('charts', [])
        if not chart:
            raise ValueError(artifact['facts'].get('limitation', 'No chartable values.'))
        return save(runtime, {'facts': artifact['facts'], 'charts': chart})
    if y is None:
        raise ValueError('Provide a numeric y column. Query or calculate category counts/averages first.')
    if kind == 'scatter':
        values = pd.DataFrame({'x': analytics.numeric(frame, x), 'y': analytics.numeric(frame, y)})
        if group_by:
            if group_by not in frame:
                raise ValueError('Grouping column not found.')
            values['group'] = frame[group_by].astype('string').fillna('(missing)')
            if values['group'].nunique() > 12:
                raise ValueError('Use at most 12 scatter groups.')
        values = values.dropna(subset=['x', 'y'])
        if values.empty:
            raise ValueError('No complete finite pairs are available.')
        sampled = values.sample(2000, random_state=7) if len(values) > 2000 else values
        chart = {'type': 'scatter', 'title': f'{analytics.label(y)} by {analytics.label(x)}',
                 'x_label': x, 'y_label': y, 'points': sampled[['x', 'y']].values.tolist(),
                 'total_points': len(values), 'displayed_points': len(sampled),
                 'caption': 'Each point is a complete record. Chart points may be sampled; calculations use their own stated records.'}
        if group_by:
            chart['groups'] = [{'name': str(group), 'points': selected[['x', 'y']].values.tolist()}
                               for group, selected in sampled.groupby('group')]
        return save(runtime, {'facts': {'operation': 'chart', 'type': kind, 'rows_used': len(values),
                'rows_excluded': len(frame) - len(values), 'x': x, 'y': y}, 'charts': [chart]})
    values = pd.DataFrame({'label': frame[x].astype('string'), 'value': analytics.numeric(frame, y)}).dropna()
    if values.empty or len(values) > 100:
        raise ValueError('Chart needs 1 to 100 rows. Aggregate or bin the data first.')
    if kind == 'pie' and ((values['value'] < 0).any() or values['value'].sum() <= 0):
        raise ValueError('Pie values must be nonnegative parts of a positive whole. Use bars for means.')
    chart = {'type': kind, 'title': f'{analytics.label(y)} by {analytics.label(x)}',
             'x_label': x, 'y_label': y, 'labels': values['label'].tolist(), 'values': values['value'].tolist(),
             'caption': 'Visualization of the selected data or saved calculation result; missing values excluded.'}
    return save(runtime, {'facts': {'operation': 'chart', 'type': kind, 'source_result_id': source_result_id}, 'charts': [chart]})


def chart_tool(name, kind, description):
    @tool(name, response_format='content_and_artifact')
    def chart(runtime: ToolRuntime[ToolContext], x: str | None = None, y: str | None = None,
              source_result_id: str | None = None, group_by: str | None = None, bins: int = 8):
        """Draw a chart from validated dataset columns or a previous result_id; no invented values."""
        return draw(runtime, kind, x, y, source_result_id, group_by, bins)
    chart.description = description
    return chart


TOOLS = [inspect_dataset, query_data, calculate_statistics,
    chart_tool('scatter_plot', 'scatter', 'Draw two numeric columns x/y as points. Optional group_by category (max 12). source_result_id optional. No correlation calculation.'),
    chart_tool('pie_chart', 'pie', 'Show parts of a whole. x alone bins a numeric variable or counts categories. For a saved frequency/query table supply source_result_id, x=labels, y=count/percentage. Never use means as pie slices.'),
    chart_tool('bar_chart', 'bar', 'Compare categories from a computed table. Specify x labels, numeric y, and optional source_result_id. Calculate group means/counts first.'),
    chart_tool('line_chart', 'line', 'Plot ordered query results. Specify x, numeric y and source_result_id. Query must order time/sequence first; do not imply a time trend from unordered rows.'),
    chart_tool('histogram', 'histogram', 'Count a numeric column x into bins (2-20). Optional source_result_id. Use to inspect distribution.'),
    chart_tool('box_plot', 'box', 'Show five-number summary of numeric x. Full-range whiskers, not Tukey outlier thresholds. Optional source_result_id.'),
    chart_tool('correlation_heatmap', 'heatmap', 'Visualize an existing Pearson/Spearman correlation matrix. Required source_result_id from calculate_statistics; no new calculation.')]
