def profile_frame(frame):
    return {'rows': len(frame), 'column_count': len(frame.columns),
            'missing_cells': int(frame.isna().sum().sum()),
            'columns': [{'name': str(c), 'dtype': str(frame[c].dtype),
                         'missing': int(frame[c].isna().sum()),
                         'unique': int(frame[c].nunique())} for c in frame.columns]}
