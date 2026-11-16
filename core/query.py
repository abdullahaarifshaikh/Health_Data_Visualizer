"""Restrict the SQL callback that PandasAI's Docker adapter runs on the host."""
import threading
import duckdb


def query_frames(query, frames):
    connection = duckdb.connect(config={'enable_external_access': False,
                                        'memory_limit': '256MB', 'threads': 2})
    timer = threading.Timer(15, connection.interrupt)
    try:
        statements = connection.extract_statements(query)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError('Only a single SELECT query is supported.')
        for name, frame in frames.items():
            connection.register(name, frame)
        timer.start()
        result = connection.sql(query).limit(100001).df()
        if len(result) > 100000:
            raise ValueError('Query result exceeds 100,000 rows. Aggregate the data first.')
        return result
    finally:
        timer.cancel()
        connection.close()
