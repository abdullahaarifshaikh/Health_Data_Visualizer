"""Deterministic calculations and declarative chart data, independent of any LLM."""
import json
import re
import numpy as np
import pandas as pd
from core.profile import profile_frame
from core.query import query_frames


def label(column):
    return column.replace('_', ' ').strip().capitalize()


def records(frame):
    return json.loads(frame.to_json(orient='values', date_format='iso'))


def numeric(frame, column):
    if column not in frame.columns:
        raise ValueError(f'Unknown column {column!r}. Choose a column from the dataset schema.')
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise ValueError(f'{column} is not a numeric measurement.')
    return frame[column].astype(float).replace([np.inf, -np.inf], np.nan)


def relationship(frame, x, y):
    if x == y:
        raise ValueError('Choose two different measurements to study a relationship.')
    values = pd.DataFrame({x: numeric(frame, x), y: numeric(frame, y)}).dropna()
    n = len(values)
    if n < 2:
        return {'facts': {'operation': 'relationship', 'x': x, 'y': y, 'rows_used': n,
                          'rows_excluded': len(frame) - n, 'limitation': 'At least two complete finite pairs are needed.'}}
    variable = values[x].nunique() > 1 and values[y].nunique() > 1
    r = float(values[x].corr(values[y])) if variable else None
    sampled = values.sample(n=2000, random_state=7) if n > 2000 else values
    chart = {'type': 'scatter', 'title': f'{label(y)} and {label(x)}', 'x_label': label(x),
             'y_label': label(y), 'points': records(sampled), 'total_points': n,
             'displayed_points': len(sampled), 'caption': 'Each point is one complete record. Hover to inspect values; drag the zoom controls to explore.'}
    if variable:
        slope, intercept = np.polyfit(values[x], values[y], 1)
        ends = [float(values[x].min()), float(values[x].max())]
        chart['trend'] = [[v, float(slope * v + intercept)] for v in ends]
    facts = {'operation': 'relationship', 'x': x, 'y': y, 'rows_total': len(frame),
             'rows_used': n, 'rows_excluded': len(frame) - n, 'pearson_r': r,
             'squared_pearson_r': r * r if r is not None else None,
             'x_range': [float(values[x].min()), float(values[x].max())],
             'y_range': [float(values[y].min()), float(values[y].max())],
             'x_mean': float(values[x].mean()), 'y_mean': float(values[y].mean()),
             'constant_variable': not variable, 'chart_sampled': n > 2000,
             'study_design': 'Not supplied. Do not infer cross-sectional, longitudinal, or one record per person.',
             'predictive_validation': 'None performed. Do not judge usefulness for individual prediction or clinical importance.',
             'interpretation_limit': 'Descriptive association only. No causal inference, significance test, or adjustment for confounding was performed.'}
    return {'facts': facts, 'charts': [chart]}


def distribution(frame, column, chart_type='histogram', bins=8):
    if column not in frame.columns:
        raise ValueError('Choose an existing dataset column.')
    if chart_type not in ('histogram', 'pie', 'bar', 'box', 'none'):
        raise ValueError('Supported distributions: histogram, pie, bar, box, none.')
    bins = max(2, min(int(bins), 20))
    series = frame[column]
    facts = {'operation': 'distribution', 'column': column, 'rows_total': len(frame)}
    if pd.api.types.is_numeric_dtype(series):
        valid = numeric(frame, column).dropna()
        facts.update(rows_used=len(valid), rows_excluded=len(frame) - len(valid))
        if valid.empty:
            facts['limitation'] = 'No finite numeric values are available.'
            return {'facts': facts}
        facts['statistics'] = {k: float(v) for k, v in valid.describe().items() if pd.notna(v)}
        counts, edges = np.histogram(valid, bins=bins)
        names = [f'{edges[i]:.2f}–{edges[i+1]:.2f}' for i in range(len(counts))]
        chart = {'type': 'pie' if chart_type == 'pie' else 'bar', 'title': f'Distribution of {label(column)}',
                 'labels': names, 'values': counts.tolist(), 'x_label': label(column), 'y_label': 'Record count',
                 'caption': 'Numeric values are grouped into equal-width ranges. Ranges include the lower edge and exclude the upper edge, except the last range which includes both. These are data bins, not clinical thresholds.'}
        facts['bin_counts'] = [{'range': name, 'count': int(count), 'percent': round(float(count) / len(valid) * 100, 1)} for name, count in zip(names, counts)]
        if chart_type == 'box':
            chart.update(type='box', values=[float(v) for v in valid.quantile([0, .25, .5, .75, 1])],
                         caption='Five-number summary: minimum, lower quartile, median, upper quartile, maximum. Whiskers show the full range, not Tukey outlier thresholds.')
    else:
        valid = series.dropna().astype(str)
        counts = valid.value_counts()
        top = counts.head(19)
        names, amounts = top.index.tolist(), top.astype(int).tolist()
        if len(counts) > 19:
            names.append('(remaining categories)')
            amounts.append(int(counts.iloc[19:].sum()))
        facts.update(rows_used=len(valid), rows_excluded=len(frame) - len(valid),
                     category_counts=dict(zip(names, amounts)))
        chart = {'type': 'pie' if chart_type == 'pie' else 'bar', 'title': f'{label(column)} counts',
                 'labels': names, 'values': amounts, 'x_label': label(column), 'y_label': 'Record count',
                 'caption': 'Missing values are excluded. Small categories are combined when there are more than 19 categories.'}
    return {'facts': facts, 'charts': [] if chart_type == 'none' else [chart]}


def compare_groups(frame, category, measurement=None, aggregation='mean', chart_type='bar'):
    if category not in frame.columns:
        raise ValueError('Choose an existing category column.')
    if aggregation not in ('mean', 'median', 'count', 'sum', 'min', 'max'):
        raise ValueError('Unsupported aggregation.')
    if chart_type not in ('bar', 'pie'):
        raise ValueError('Choose bar or pie for a group comparison.')
    if chart_type == 'pie' and aggregation != 'count':
        raise ValueError('A pie chart needs counts that form a whole. Use a bar chart for averages.')
    categories = frame[category].astype('string').fillna('(missing)')
    grouped = pd.DataFrame({'category': categories})
    excluded = 0
    if aggregation == 'count':
        result = grouped.groupby('category').size().rename('value').reset_index()
    else:
        if measurement is None:
            raise ValueError('Choose the numeric measurement to compare.')
        grouped['measurement'] = numeric(frame, measurement)
        excluded = int(grouped['measurement'].isna().sum())
        result = grouped.dropna().groupby('category')['measurement'].agg(aggregation).rename('value').reset_index()
    if len(result) > 30:
        raise ValueError('There are more than 30 groups. Choose a broader category or use a bounded SQL aggregation.')
    facts = {'operation': 'groups', 'category': category, 'measurement': measurement,
             'aggregation': aggregation, 'rows_excluded': excluded,
             'groups': json.loads(result.to_json(orient='records'))}
    chart = {'type': chart_type, 'title': f'{aggregation.capitalize()} {label(measurement) if measurement else "record count"} by {label(category)}',
             'labels': result['category'].astype(str).tolist(), 'values': result['value'].tolist(),
             'x_label': label(category), 'y_label': f'{aggregation} {label(measurement)}' if measurement else 'Record count',
             'caption': 'Compare descriptive summaries. Differences between groups do not establish causes.'}
    return {'facts': facts, 'charts': [chart], 'table': {'columns': result.columns.tolist(), 'rows': records(result)}}


def overview(frame):
    profile = profile_frame(frame)
    profile['numeric_summary'] = json.loads(frame.select_dtypes(include='number').describe().to_json()) if len(frame.select_dtypes(include='number').columns) else {}
    return {'facts': {'operation': 'overview', **profile},
            'table': {'columns': ['Variable', 'Type', 'Missing', 'Unique values'],
                      'rows': [[c['name'], c['dtype'], c['missing'], c['unique']] for c in profile['columns']]}}


def sql_analysis(frame, sql, chart_type='none', x=None, y=None):
    result = query_frames(sql, {'healthcare': frame})
    if len(result) > 2000:
        raise ValueError('Aggregate or limit the SQL result to 2,000 rows for a readable response.')
    facts = {'operation': 'query', 'result_rows': len(result), 'columns': result.columns.tolist(),
             'result_preview': records(result.head(25)), 'preview_truncated': len(result) > 25}
    artifact = {'facts': facts, 'table': {'columns': result.columns.tolist(), 'rows': records(result.head(200))}}
    if chart_type != 'none':
        if chart_type not in ('bar', 'line', 'scatter', 'pie') or x not in result.columns or y not in result.columns:
            raise ValueError('Specify a supported chart type and existing x/y result columns.')
        valid = result[[x, y]].copy()
        valid[y] = numeric(result, y)
        valid = valid.dropna()
        chart = {'type': chart_type, 'title': f'{label(y)} by {label(x)}',
                 'x_label': label(x), 'y_label': label(y), 'caption': 'Chart of the computed query result; missing values excluded.'}
        if chart_type == 'scatter':
            valid[x] = numeric(result, x)
            valid = valid.dropna()
            chart['points'] = records(valid)
        else:
            if chart_type == 'pie' and ((valid[y] < 0).any() or valid[y].sum() <= 0):
                raise ValueError('Pie chart values must be nonnegative and sum to more than zero.')
            chart.update(labels=valid[x].astype(str).tolist(), values=valid[y].tolist())
        artifact['charts'] = [chart]
    return artifact


def mentioned_columns(question, frame):
    normalized = ' '.join(re.sub(r'[^a-z0-9 ]', ' ', question.lower().replace('_', ' ')).split())
    columns = []
    for col in frame.columns:
        words = str(col).lower().replace('_', ' ').split()
        aliases = [' '.join(words), ' '.join(w for w in words if w not in ('bpm', 'mmhg', 'years', 'hours', 'kg', 'cm'))]
        if any(alias and re.search(r'\b' + re.escape(alias) + r'\b', normalized) for alias in aliases):
            columns.append(str(col))
    return columns


def direct_relationship(question, frame):
    """Shortcut only explicit, unfiltered relationships between two named numeric columns."""
    normalized = re.sub(r'[^a-z0-9 ]', ' ', question.lower().replace('_', ' '))
    normalized = ' '.join(normalized.split())
    if not re.search(r'\b(relationship|correlation|association|related)\b', normalized):
        return None
    if re.search(r'\b(that|this) correlation\b', normalized):
        return None
    if re.search(r'\b(where|only|among|older|younger|above|below|group|class|female|male|exclude|filter|if)\b|[<>=]', question.lower()):
        return None
    columns = mentioned_columns(question, frame.select_dtypes(include='number'))
    return columns if len(columns) == 2 else None


def direct_distribution(question, frame):
    """Recognize an explicit chart request for exactly one named column, without filters."""
    normalized = ' '.join(re.sub(r'[^a-z0-9 ]', ' ', question.lower().replace('_', ' ')).split())
    if re.search(r'\b(where|only|among|older|younger|above|below|group|class|female|male|exclude|filter|if)\b|[<>=]', question.lower()):
        return None
    chart_type = next((name for word, name in [('pie', 'pie'), ('histogram', 'histogram'), ('boxplot', 'box'), ('distribution', 'histogram')]
                       if re.search(r'\b' + word + r'\b', normalized)), None)
    if not chart_type:
        return None
    columns = mentioned_columns(question, frame)
    return {'column': columns[0], 'chart_type': chart_type} if len(columns) == 1 else None
