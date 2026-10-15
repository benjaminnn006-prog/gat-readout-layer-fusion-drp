import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from src.model import MedicalGAT,AttentionReadout,VARIANTS
from src.data import collate_pairs
from src.prepare import cold_frames
from src.protocol import build_tasks,config
from reference_model import GATNet


def graph(nodes=5,cell_dim=128):
    edges = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]],dtype=torch.long)
    return Data(x=torch.randn(nodes,768),edge_index=edges,GEF=torch.randn(1,cell_dim),
                ic50=torch.tensor([0.5]),row_index=torch.tensor([0]))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_baseline_exact_original_parameterization_and_output(self):
        torch.manual_seed(12)
        original = GATNet(768,128).eval()
        replacement = MedicalGAT(128,'baseline').eval()
        replacement.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in original.parameters()),3284993)
        self.assertEqual(sum(p.numel() for p in replacement.parameters()),3284993)
        batch = Batch.from_data_list([graph(),graph()])
        with torch.no_grad():
            self.assertTrue(torch.equal(original(batch),replacement(batch)))

    def test_all_variants_finite_forward_and_backward(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = MedicalGAT(128,variant)
                batch = Batch.from_data_list([graph(),graph()])
                output = model(batch)
                self.assertEqual(tuple(output.shape),(2,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for name,param in model.named_parameters():
                    self.assertIsNotNone(param.grad,name)
                    self.assertTrue(torch.isfinite(param.grad).all(),name)

    def test_attention_normalizes_separately_per_graph(self):
        pool = AttentionReadout(12,5).eval()
        x = torch.randn(9,12)
        batch = torch.tensor([0]*4+[1]*5)
        _,weights = pool(x,batch,True)
        for group in range(2):
            self.assertAlmostEqual(float(weights[batch==group].sum()),1.0,places=6)
        first,_ = pool(x[:4],torch.zeros(4,dtype=torch.long),True)
        together,_ = pool(x,batch,True)
        self.assertTrue(torch.allclose(first[0],together[0],atol=1e-6))

    def test_readout_invariant_to_node_permutation(self):
        pool = AttentionReadout(12,5).eval()
        x,batch = torch.randn(8,12),torch.zeros(8,dtype=torch.long)
        order = torch.randperm(8)
        self.assertTrue(torch.allclose(pool(x,batch),pool(x[order],batch[order]),atol=1e-6))

    def test_full_graph_output_invariant_to_node_permutation(self):
        model = MedicalGAT(128,'full').eval()
        first = graph()
        order = torch.tensor([3,0,4,1,2])
        inverse = torch.argsort(order)
        second = Data(x=first.x[order],edge_index=inverse[first.edge_index],GEF=first.GEF)
        with torch.no_grad():
            left = model(Batch.from_data_list([first]))
            right = model(Batch.from_data_list([second]))
        self.assertTrue(torch.allclose(left,right,rtol=1e-5,atol=1e-6))

    def test_layer_weights_sum_one_and_drug_branch_ignores_cell(self):
        model = MedicalGAT(128,'full').eval()
        batch = Batch.from_data_list([graph(),graph()])
        with torch.no_grad():
            before,details = model.drug_vector(batch,True)
            batch.GEF = torch.randn_like(batch.GEF)*100
            after = model.drug_vector(batch)
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.allclose(details['layer_weights'].sum(1),torch.ones(2),atol=1e-6))

    def test_single_modality_ignores_removed_modality(self):
        drug_model = MedicalGAT(128,'drug_only').eval()
        cell_model = MedicalGAT(128,'cell_only').eval()
        batch = Batch.from_data_list([graph(),graph()])
        with torch.no_grad():
            drug_before = drug_model(batch)
            batch.GEF = torch.randn_like(batch.GEF)
            self.assertTrue(torch.equal(drug_before,drug_model(batch)))
            cell_before = cell_model(batch)
            batch.x = torch.randn_like(batch.x)
            self.assertTrue(torch.equal(cell_before,cell_model(batch)))

    def test_capacity_chosen_without_metric_and_close(self):
        model = MedicalGAT(128,'capacity')
        self.assertLess(abs(model.capacity_extra_parameters-model.capacity_target_parameters)/
                        model.capacity_target_parameters,0.01)

    def test_batch_prediction_indices_not_incremented(self):
        items = [graph(),graph()]
        items[0].row_index = torch.tensor([5])
        items[1].row_index = torch.tensor([8])
        self.assertEqual(collate_pairs(items).row_index.tolist(),[5,8])

    def test_cold_split_cell_and_pair_isolation(self):
        full = pd.DataFrame([{'Cell':str(i),'Drug':str(d),'IC50':float(i)/100}
                             for i in range(100) for d in range(2)])
        frames,ids = cold_frames(full,config())
        self.assertEqual([len(x) for x in ids],[70,15,15])
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse(set(frames[i].Cell)&set(frames[j].Cell))
        self.assertEqual(sum(len(f) for f in frames),len(full))

    def test_task_matrix_and_repeat_alignment(self):
        tasks = build_tasks()
        self.assertEqual(len(tasks),80)
        self.assertEqual(len({t['id'] for t in tasks}),80)
        for t in tasks:
            self.assertEqual(t['seed'],42+t['repeat'])
            self.assertEqual(t['split'],t['repeat'] if t['regime']=='warm' else 0)
        self.assertEqual(sum(t['regime']=='cold_cell' for t in tasks),20)
        self.assertEqual(sum(t['feature']=='raw' for t in tasks),10)


if __name__=='__main__':
    unittest.main()
