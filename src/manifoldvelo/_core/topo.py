"""Restore the successful monotone-circle backend with an exact known root.

The old Manifold-Topo architecture is retained. The root is the only new
supervision: the input chart is rotated to zero and the map offset is fixed.
"""
import json
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
from .common import write,sha
from .cycle import CycleConfig,CycleTrainer,initial_phase
from .cycle_base.models import circle
from .cycle_base.losses import masked_mse
from .temporal import MonotoneCircle
from .temporal import Stability

class TopoTrainer(CycleTrainer):
    def reference_prediction(self):
        if not hasattr(self,'warp'):return super().reference_prediction()
        with torch.no_grad():p=self.warp(self._tensor(self.base_phase))
        z=circle(p);return self._prediction(z,z,self.cell_ids)

    def predict(self,S,genes,cell_ids):
        _,positions=self.transform.apply(S,genes)
        distances=cdist(positions,self.graph.positions)
        nearest=np.argsort(distances,axis=1)[:,:10]
        d=np.take_along_axis(distances,nearest,1);scale=np.maximum(d[:,-1:],1e-8)
        weights=np.exp(-d**2/scale**2)
        p=(np.angle(np.sum(weights*np.exp(2j*np.pi*self.base_phase)[nearest],axis=1))/(2*np.pi))%1
        with torch.no_grad():phase=self.warp(self._tensor(p))
        z=circle(phase);return self._prediction(z,z,np.asarray(cell_ids,str))

    @classmethod
    def load(cls,path,device='cpu'):
        from pathlib import Path
        obj=super().load(path,device)
        obj.base_phase=np.load(Path(path)/'base_phase.npy')
        obj.warp=MonotoneCircle(obj.base_phase).to(obj.device)
        obj.warp.load_state_dict(torch.load(Path(path)/'warp.pt',map_location=obj.device,weights_only=True))
        obj.warp.eval();return obj

def run_topo(a,job,out,device):
    from .serialization import save_model
    cfg=CycleConfig(neighbors=10,graph_pcs=10,seed=job['seed'],batch_size=1024,
        warmup_steps=1,joint_steps=0,decoder_hidden=128,collocation_points=192,
        residual=True,residual_hidden=64,residual_rank=min(18,len(a['genes'])),residual_bound=8.,
        curvature_weight=1e-8,dynamics_weight=.001,residual_weight=1e-5,rate_weight=1e-5,
        anchor_weight=.001,geometry_weight=0.,weight_decay=1e-6,root_weight=.1,root_phase=.05)
    p0,diagnostic=initial_phase(a,cfg);base=((p0-cfg.root_phase)%1).astype(np.float32);base[int(a['root_index'])]=0
    trainer=TopoTrainer(cfg,device=device).fit(a['S'],a['U'],a['genes'],a['cell_ids'],a['W_0_mask'],
        n_selected_genes=int(a['n_selected_genes']),initial_phase=p0,
        root_mask=np.arange(len(a['S']))==int(a['root_index']),verbose=False)
    model=trainer.model
    for p in model.encoder.parameters():p.requires_grad_(False)
    trainer.base_phase=base;warp=trainer.warp=MonotoneCircle(base).to(device)
    with torch.no_grad():warp.offset.fill_(cfg.root_phase)
    warp.offset.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]+[p for p in warp.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam([{'params':[p for p in model.parameters() if p.requires_grad],'lr':.002},
        {'params':[p for p in warp.parameters() if p.requires_grad],'lr':.004}],weight_decay=cfg.weight_decay)
    stable=Stability(minimum=5000,loss_tol=.0002,phase_tol=.0005,curve_tol=.003,param_tol=.01)
    rng=np.random.default_rng(job['seed']+9102);base_tensor=trainer._tensor(base);target=trainer._tensor(p0)
    maximum=int(job.get('steps',18000));started=time.time();history=[];converged=False
    write(out/'training_protocol.json',dict(backend='old Manifold-Topo monotone circle with exact known-root offset',
        config=cfg.to_dict(),maximum_steps=maximum,stability=stable.settings,root_identity_only=True,
        labels_used=False,temporal_source_sha256=sha(__import__(__package__+'.temporal',fromlist=['x']).__file__)))
    write(out/'INITIALIZATION.json',diagnostic)
    for step in range(1,maximum+1):
        if step==1501:trainer._initialize_kinetics_from_curve()
        grid=trainer._tensor((np.arange(cfg.collocation_points)+rng.random())/cfg.collocation_points)
        optimizer.zero_grad(set_to_none=True);p=warp(base_tensor)
        predicted=model.decoder(circle(p));reconstruction=masked_mse(predicted,trainer.y,trainer.mask,model.state_scale)
        curve,dy=model.curve(grid,create_graph=True);dyn=model.dynamics;rhs=dyn.clock_rhs(grid,curve)
        dynamic=((dy-rhs)/model.state_scale).square().mean()
        curvature=((dy.roll(-1,0)-dy)*len(grid)/model.state_scale).square().mean()
        anchor=(1-torch.cos(2*np.pi*(p-target))).mean()
        residual=(dyn.residual(grid)/dyn.alpha_scale).square().mean()
        rate=(torch.log(dyn.beta/dyn.beta_initial).square().mean()+torch.log(dyn.gamma/dyn.gamma_initial).square().mean()
              +(dyn.W*(1-dyn.W_0_mask)).square().mean())
        loss=reconstruction+cfg.anchor_weight*anchor+1e-5*warp.smoothness()+cfg.curvature_weight*curvature
        if step>1500:loss=loss+cfg.dynamics_weight*dynamic+cfg.residual_weight*residual+cfg.rate_weight*rate
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Topo loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);optimizer.step()
        if step%500==0:
            with torch.no_grad():
                phase=warp(base_tensor);fixed=trainer._tensor(np.arange(192)/192)
                fixed_curve=model.decode_phase(fixed)
                vector=torch.cat([v.detach().flatten() for v in params]).cpu().numpy()
            row=stable.update(step,float(loss.detach()),phase.cpu().numpy(),fixed_curve.cpu().numpy(),vector)
            row.update(seconds=time.time()-started,reconstruction=float(reconstruction.detach()),dynamics=float(dynamic.detach()))
            history.append(row);print('TOPO',step,row['loss'],row['converged'],flush=True)
            torch.save(dict(model=model.state_dict(),warp=warp.state_dict(),optimizer=optimizer.state_dict(),step=step),out/'continuation.pt')
            if step>=5000 and len(stable.records)>=4:
                recent=[r['loss'] for r in stable.records[-4:]]
                if max(recent)-min(recent)<.002:
                    for group in optimizer.param_groups:group['lr']=max(1e-5,group['lr']*.8)
            if step>1500 and row['converged']:converged=True;break
    model.eval();trainer.history=history
    np.savez_compressed(out/'prediction.npz',**trainer.reference_prediction())
    save_model(trainer,out/'model');np.save(out/'model/base_phase.npy',base)
    torch.save({k:v.detach().cpu() for k,v in warp.state_dict().items()},out/'model/warp.pt')
    write(out/'diagnostics.json',dict(steps=step,converged=converged,root_phase=float(trainer.reference_prediction()['phase'][int(a['root_index'])]),
        strict_circular_order=True,query='S-only circular interpolation in the training spectral chart, then learned monotone map'))
    write(out/'status.json',dict(complete=True,steps=step,converged=converged,stop='stability' if converged else 'budget'))
    return trainer
