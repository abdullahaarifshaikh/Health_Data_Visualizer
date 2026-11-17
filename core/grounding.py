"""Conservative checks for known unsupported claims; not a general fact-checker."""
import re


def unsupported_claims(answer, facts):
    text = answer.lower().replace('\u2011', '-').replace('\u202f', ' ')
    issues = []
    relationships = [f for f in facts if f.get('operation') == 'relationship']
    if relationships:
        if re.search(r'\b(?:for each|per (?:extra |additional )?(?:hour|year)|each (?:extra|additional) hour)\b|\bbpm (?:less|more|lower|higher)\b', text):
            issues.append('A correlation was presented as a per-unit effect without a computed slope.')
        if re.search(r'\b(?:cross[- ]sectional|longitudinal|single snapshot|just a snapshot|snapshot of)\b|one (?:record|row) per (?:person|patient)', text):
            issues.append('The study design or observation unit was inferred without metadata.')
        if re.search(r'(?:points|records|cloud) (?:are |is )?(?:spread|scattered|sit|cluster)|(?:you.ll|you will) see points', text):
            issues.append('The explanation claimed to visually inspect an unseen chart.')
        if re.search(r'little predictive power|not a useful predictor|practically meaningless|practically negligible|meaningful for most individuals', text):
            issues.append('Predictive or individual importance was inferred without validation.')
        names = ' '.join(str(f.get(k, '')) for f in relationships for k in ('x', 'y'))
        if 'resting heart' in text and 'resting' not in names.lower():
            issues.append('The measurement was relabeled as resting without supporting metadata.')
    if any(f.get('operation') == 'distribution' for f in facts):
        if re.search(r'(?:slice|wedge).{0,30}(?:on the left|on the right|at the top)|what the slice looks like', text):
            issues.append('A slice location was invented without inspecting the rendered chart.')
    return issues
