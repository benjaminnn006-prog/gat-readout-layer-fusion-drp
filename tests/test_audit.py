import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from src import collect
from src.common import write_json,fingerprint,sha256
from src.protocol import build_tasks


class AuditTests(unittest.TestCase):
    def test_smoke_cannot_become_formal_by_changing_folder(self):
        task = build_tasks()[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'metrics.json',{'protocol_signature':'fixture','task':task,'mode':'smoke'})
            with patch.object(collect,'task_root',return_value=root):
                with self.assertRaisesRegex(ValueError,'identity'):
                    collect.audit_result(task,'fixture')

    def test_partial_training_rejected(self):
        task = build_tasks()[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'metrics.json',{'protocol_signature':'fixture','task':task,'mode':'formal',
                       'publication_eligible':False,'epochs':80,'completed_epochs':79})
            with patch.object(collect,'task_root',return_value=root):
                with self.assertRaisesRegex(ValueError,'Partial'):
                    collect.audit_result(task,'fixture')

    def test_artifact_tampering_rejected(self):
        task = build_tasks()[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'metadata.json',{'sample_counts':[10,3,3],'compact_sha256':'fixture'})
            (root/'best_model.pt').write_bytes(b'test fixture only, not a trained model')
            identity = {'protocol_signature':'fixture','task':task,'mode':'formal','epochs':80,
                        'sample_counts':[10,3,3],'feature_sha256':'fixture'}
            result = {**identity,'publication_eligible':True,'completed_epochs':80,
                      'run_signature':fingerprint(identity),
                      'artifacts_sha256':{'best_model.pt':sha256(root/'best_model.pt')}}
            write_json(root/'metrics.json',result)
            (root/'best_model.pt').write_bytes(b'modified fixture')
            with patch.object(collect,'task_root',return_value=root),patch.object(collect,'split_root',return_value=root):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    collect.audit_result(task,'fixture')


if __name__=='__main__':
    unittest.main()
