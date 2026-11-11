"""Check the actual FM, teacher and DMD callers against the shared media API."""
import ast
import inspect
from pathlib import Path
import unittest

class MediaContractTests(unittest.TestCase):
    def test_production_media_calls_bind(self):
        root = Path(__file__).resolve().parents[1] / 'pipelines' / 'h3'
        source = ast.parse((root / 'train_stage2.py').read_text())
        definition = next(node for node in source.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'deliver')
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]),
                     str(root / 'train_stage2.py'), 'exec'), namespace)
        signature = inspect.signature(namespace['deliver'])
        checked = {}
        for name in ('train_stage2.py', 'evaluate_stage3_teacher.py', 'train_stage3.py'):
            count = 0
            for call in ast.walk(ast.parse((root / name).read_text())):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == 'deliver'):
                    continue
                with self.subTest(caller=name, line=call.lineno):
                    self.assertTrue(all(k.arg is not None for k in call.keywords))
                    signature.bind(*[None for _ in call.args],
                                   **{k.arg: None for k in call.keywords})
                count += 1
            checked[name] = count
        self.assertTrue(all(checked.values()), checked)


if __name__ == '__main__':
    unittest.main()
