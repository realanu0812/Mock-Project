import json
import os

def force_hashable(val):
    if isinstance(val, (dict, list, set)):
        return json.dumps(val, ensure_ascii=False)
    return val

combined_path = os.path.join('nutrition_insights', 'data', 'combined.json')

with open(combined_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for rec in data:
    if 'top_comments' in rec:
        rec['top_comments'] = force_hashable(rec['top_comments'])

with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('Patched combined.json: all top_comments fields are now JSON strings.')
