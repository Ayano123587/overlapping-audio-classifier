"""Visible notebook algorithms must stay aligned with production code."""
import ast
import json
from pathlib import Path


def test_visible_audio_core_matches_source_and_notebooks_keep_real_outputs():
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/02_complete_natural_audio_workflow.ipynb').read_text(encoding='utf-8'))
    visible={node.name:node for cell in notebook['cells'] if cell['cell_type']=='code'
             for node in ast.parse(''.join(cell['source'])).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    contracts={'transfer':['windows','Head','masked_loss','predict_head'],
               'metrics':['f1_per_class','average_precision'],'natural':['scores'],
               'learned':['fit_head'],'experiment':['StrongCNN'],
               'postprocess':['checked_probabilities','convex_mix','temporal_filter','apply_policy','component_predictions']}
    for module,names in contracts.items():
        nodes={node.name:node for node in ast.parse((root/f'src/overlap_audio/{module}.py').read_text(encoding='utf-8')).body
               if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for name in names:assert ast.dump(visible[name])==ast.dump(nodes[name]),name
    images=0
    for path in (root/'notebooks').glob('*.ipynb'):
        data=json.loads(path.read_text(encoding='utf-8'))
        for cell in data['cells']:
            if cell['cell_type']!='code':continue
            assert cell['execution_count'] is not None
            for output in cell.get('outputs',[]):
                assert output['output_type']!='error'
                images+=int('image/png' in output.get('data',{}))
    assert images>=8
