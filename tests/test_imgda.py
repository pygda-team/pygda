"""API contracts for the released ImGDA core; no downloads or paper-score assertions."""
import tempfile
from pathlib import Path
import pickle
import unittest

import torch
from torch_geometric.data import Data

from pygda.datasets import load_imgda_pair, prepare_imgda_source
from pygda.models import BaseGDA, ImGDA


def pair(device):
    generator = torch.Generator().manual_seed(9)
    edges = torch.stack((torch.arange(12),torch.roll(torch.arange(12),-1)))
    source = Data(x=torch.randn(12,4,generator=generator),y=torch.arange(3).repeat_interleave(4),
                  edge_index=edges,global_adj=edges,global_norm=torch.ones(12),
                  imgda_class_counts=torch.tensor([4,4,4]))
    target = Data(x=torch.randn(12,4,generator=generator),edge_index=edges,
                  global_adj=edges,global_norm=torch.ones(12))
    return source.to(device),target.to(device)


class ImGDAAPITests(unittest.TestCase):
    def devices(self):
        return ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

    def model(self,device,**kwargs):
        return ImGDA(4,8,3,epoch=2,k=2,seed=0,device=device,verbose=0,**kwargs)

    def test_public_api_and_configuration(self):
        self.assertTrue(issubclass(ImGDA,BaseGDA))
        for config in ({'batch_size':2},{'num_layers':3},{'num_neigh':2},{'dynamic_temperature':0},{'unexpected':1}):
            with self.assertRaises((ValueError,TypeError)):
                ImGDA(4,8,3,**config)
        with self.assertRaises(RuntimeError):
            self.model('cpu').predict(pair('cpu')[1])

    def test_fit_target_labels_and_input_integrity(self):
        for device in self.devices():
            with self.subTest(device=device):
                source,target = pair(device)
                before = {k:v.clone() for k,v in source if isinstance(v,torch.Tensor)}
                before_target = {k:v.clone() for k,v in target if isinstance(v,torch.Tensor)}
                callbacks = []
                first = self.model(device)
                self.assertIs(first.fit(source,target,lambda epoch,losses,logits:callbacks.append(epoch)),first)
                target.y = torch.tensor([-100],device=device) # deliberately invalid labels, ignored by fit
                second = self.model(device).fit(source,target)
                self.assertEqual(callbacks,[0,1])
                for a,b in zip(first.model.parameters(),second.model.parameters()):
                    self.assertTrue(torch.allclose(a,b,atol=1e-5,rtol=1e-4))
                self.assertTrue(all(torch.equal(source[k],v) for k,v in before.items()))
                self.assertTrue(all(torch.equal(target[k],v) for k,v in before_target.items()))
                self.assertTrue(all(d.training for d in first.model['encoder'].dropout_layers))

    def test_predict_uses_new_graph_and_cache(self):
        for device in self.devices():
            with self.subTest(device=device):
                source,target = pair(device)
                model = self.model(device).fit(source,target)
                torch.manual_seed(7)
                original,labels = model.predict(target)
                self.assertIsNone(labels)
                changed = target.clone()
                changed.x = changed.x+3
                changed.global_norm = changed.global_norm*.5
                changed.y = torch.arange(12,device=device)%3
                torch.manual_seed(7)
                updated,labels = model.predict(changed)
                self.assertEqual(tuple(updated.shape),(12,3))
                self.assertTrue(torch.equal(labels,changed.y))
                self.assertFalse(torch.allclose(original,updated))
                self.assertTrue(torch.equal(model.model['encoder'].conv_layers[0].cache_dict['prediction'][1],changed.global_norm))

    def test_invalid_cache_and_class_population(self):
        source,target = pair('cpu')
        for kind in ('weights','bounds','counts','singleton'):
            bad = source.clone()
            if kind=='weights': bad.global_norm[0]=float('nan')
            if kind=='bounds': bad.global_adj[0,0]=12
            if kind=='counts': bad.imgda_class_counts=torch.tensor([1,1,1])
            if kind=='singleton': bad.y=torch.tensor([0]*5+[1]*6+[2])
            with self.assertRaises(ValueError): self.model('cpu').fit(bad,target)

    def test_loader_masks_and_cache_preservation(self):
        source,target = pair('cpu')
        source.train_mask=torch.ones(12,dtype=torch.bool)
        target.y=torch.arange(12)%3
        with tempfile.TemporaryDirectory(prefix='imgda-test-') as name:
            root=Path(name)
            for domain,graph in (('S',source),('T',target)):
                path=root/'data'/domain/'processed'
                path.mkdir(parents=True)
                torch.save((graph,None),path/'data.pt')
            (root/'tmp').mkdir()
            for domain,graph in (('S1',source),('T',target)):
                with (root/'tmp'/(domain+'.pkl')).open('wb') as stream:
                    pickle.dump((graph.global_adj,graph.global_norm),stream)
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            selected,unlabeled=load_imgda_pair(root,'S','T',1,'cpu')
            self.assertIsNone(unlabeled.y)
            self.assertTrue(torch.equal(selected.y,source.y))
            self.assertTrue(torch.equal(selected.imgda_original_node_ids,torch.arange(12)))
            self.assertTrue(torch.equal(selected.global_norm,source.global_norm))
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError): load_imgda_pair('/unused','../S','T')


if __name__=='__main__':
    unittest.main(verbosity=2)
