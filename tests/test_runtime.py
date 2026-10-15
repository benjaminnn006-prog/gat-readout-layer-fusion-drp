"""CPU-only tiny fixtures; never create a formal publication result."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from src import train
from src.common import ROOT,read_json
from src.protocol import config


@unittest.skipUnless((ROOT/'cache'/'audit.json').exists(),'Run preparation before runtime integration checks')
class RuntimeTests(unittest.TestCase):
    def test_interrupted_smoke_resumes_exactly_on_cpu(self):
        args = argparse.Namespace(task='warm__compact__full__r0',mode='smoke',device='cpu',
                                  epochs=2,max_samples=16)
        original_predict = train.predict
        interrupted = [False]
        def interrupt_first_validation(*values):
            result = original_predict(*values)
            if not interrupted[0]:
                interrupted[0] = True
                train.STOP_REQUESTED = True
            return result
        with tempfile.TemporaryDirectory(prefix='gat_resume_') as temporary:
            resumed = Path(temporary)/'resumed'
            continuous = Path(temporary)/'continuous'
            with patch.object(train,'task_root',return_value=resumed):
                with patch.object(train,'predict',side_effect=interrupt_first_validation):
                    with self.assertRaisesRegex(RuntimeError,'resumable'):
                        train.run(args)
                self.assertTrue((resumed/'last.pt').exists())
                self.assertFalse((resumed/'metrics.json').exists())
                train.run(args)
            with patch.object(train,'task_root',return_value=continuous):
                train.run(args)
            left = pd.read_csv(resumed/'test_predictions.csv')
            right = pd.read_csv(continuous/'test_predictions.csv')
            self.assertTrue(left.equals(right),'CPU RNG/optimizer resume changed predictions')
            self.assertFalse(read_json(resumed/'metrics.json')['publication_eligible'])
            self.assertEqual(read_json(resumed/'metrics.json')['resumes'],1)

    def test_formal_mode_rejects_cpu_and_reserved_gpu(self):
        args = argparse.Namespace(task='warm__compact__full__r0',mode='formal',device='cpu',
                                  epochs=None,max_samples=None)
        with self.assertRaisesRegex(ValueError,'CUDA'):
            train.run(args)
        args.device='cuda:0'
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'6'}):
            with self.assertRaisesRegex(ValueError,'allowed physical GPU'):
                train.run(args)


if __name__=='__main__':
    unittest.main()
