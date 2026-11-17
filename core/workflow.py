"""Default educational workflow; PandasAI remains an explicitly selectable backend."""
import os
from core.profile import profile_frame
if os.getenv('ANALYSIS_BACKEND', 'education') == 'pandasai':
    from core.pandasai_workflow import graph, build_graph
else:
    from core.education import graph, build_graph
