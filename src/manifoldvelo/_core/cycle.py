"""Old GraphSAGE periodic model with explicit, independently tested variants."""
from dataclasses import dataclass
import math
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import write
from .cycle_base.config import ManifoldConfig as BaseConfig
from .cycle_base.models import ManifoldModel, circle, phase, circular_delta
from .cycle_base.losses import graph_loss, masked_mse
from .cycle_base.graph import fit_transform, build_reference_graph
from .cycle_base import trainer as engine
from .rooting import root_penalty

@dataclass
class CycleConfig(BaseConfig):
    initialization: str='spectral'
    harmonics: int=0
    u_weight: float=1.
    direction_weight: float=0.
    robust_delta: float=0.
    log_reconstruction: bool=False

class FourierDecoder(nn.Module):
    def __init__(self,mean,scale,selected,harmonics):
        super().__init__();self.harmonics=harmonics
        self.register_buffer('scale',scale.clone())
        active=torch.ones_like(mean);active[selected:len(mean)//2]=0
        self.register_buffer('active',active)
        self.linear=nn.Linear(2*harmonics,len(mean))
        with torch.no_grad():
            self.linear.weight.mul_(.01)
            self.linear.bias.copy_(torch.log((mean/scale).clamp_min(1e-4)))

    def forward(self,z):
        s,c=z[...,0],z[...,1];sk,ck=s,c;features=[]
        for k in range(self.harmonics):
            features.extend([sk,ck]);sk,ck=sk*c+ck*s,ck*c-sk*s
        return torch.exp(self.linear(torch.stack(features,-1)).clamp(-16,16))*self.scale*self.active

class CycleModel(ManifoldModel):
    def __init__(self,genes,selected,cfg,mean,scale,forbidden,initial):
        super().__init__(genes,selected,cfg,mean,scale,forbidden,initial)
        if cfg.harmonics:self.decoder=FourierDecoder(mean,scale,selected,cfg.harmonics)

def objective(model,x,y,mask,edges,negative,initial_z,batch,p_grid,cfg,joint,root_mask=None):
    z,raw,_=model.encoder.reference(x,edges);predicted=model.decoder(z[batch]);zero=predicted.sum()*0
    if cfg.u_weight==1 and cfg.robust_delta==0 and not cfg.log_reconstruction:
        reconstruction=masked_mse(predicted,y[batch],mask[batch],model.state_scale)
    else:
        if cfg.log_reconstruction:
            if not hasattr(model,'_log_scale'):
                logged=torch.log1p(y);count_all=mask.sum(0).clamp_min(1)
                mean_log=(logged*mask).sum(0)/count_all
                model._log_scale=torch.sqrt(((logged-mean_log).square()*mask).sum(0)/count_all).clamp_min(.1).detach()
            residual=(torch.log1p(predicted)-torch.log1p(y[batch]))/model._log_scale
        else:residual=(predicted-y[batch])/model.state_scale
        if cfg.robust_delta:
            d=cfg.robust_delta
            error=2*d*d*(torch.sqrt(1+(residual/d).square())-1)*mask[batch]
        else:error=residual.square()*mask[batch]
        count=mask[batch].sum(0);active=count>0
        weights=torch.ones_like(model.state_scale);weights[:model.n_genes]=cfg.u_weight
        reconstruction=((error.sum(0)/count.clamp_min(1))[active]*weights[active]).sum()/weights[active].sum()
    terms=dict(reconstruction=reconstruction,geometry=graph_loss(z,edges,negative,cfg.negative_margin),
        anchor=(2*math.pi*circular_delta(phase(z),phase(initial_z))).square().mean(),
        radius=(raw.square().sum(-1)-1).square().mean(),dynamics=zero,residual=zero,curvature=zero,
        rate_prior=zero,clock=zero,direction=zero,
        root=root_penalty(phase(z),root_mask,cfg.root_phase) if cfg.root_weight else zero)
    if joint:
        curve,dy=model.curve(p_grid,create_graph=True);rho=model.clock(p_grid);active=model.decoder.active.bool()
        kinetic=model.dynamics.clock_rhs(p_grid,curve)
        terms['dynamics']=(((dy/rho[:,None]-kinetic)/model.state_scale)[:,active]).square().mean()
        curvature=(dy.roll(-1,0)-dy)*len(p_grid)/model.state_scale
        terms['curvature']=curvature[:,active].square().mean();dyn=model.dynamics
        terms['residual']=(dyn.residual(p_grid)/dyn.alpha_scale).square().mean()
        terms['rate_prior']=(torch.log(dyn.beta/dyn.beta_initial).square().mean()+torch.log(dyn.gamma/dyn.gamma_initial).square().mean()
            +(dyn.W*(1-dyn.W_0_mask)).square().mean()+(dyn.phase_W/dyn.alpha_scale[:,None]).square().mean())
        terms['clock']=torch.log(rho).square().mean()
        if cfg.direction_weight:
            tangent=(dy/rho[:,None]/model.state_scale)[:,active];velocity=(kinetic/model.state_scale)[:,active]
            terms['direction']=(1-F.cosine_similarity(tangent,velocity,dim=-1,eps=1e-6)).mean()
    anchor=cfg.anchor_weight if joint else cfg.warmup_anchor_weight
    total=(terms['reconstruction']+cfg.geometry_weight*terms['geometry']+anchor*terms['anchor']+
        cfg.radius_weight*terms['radius']+cfg.dynamics_weight*terms['dynamics']+cfg.residual_weight*terms['residual']+
        cfg.curvature_weight*terms['curvature']+cfg.rate_weight*terms['rate_prior']+cfg.clock_weight*terms['clock']+
        cfg.root_weight*terms['root']+cfg.direction_weight*terms['direction'])
    return total,terms

# Bind only this private, copied training engine. Historical packages are unchanged.
engine.ManifoldConfig=CycleConfig
engine.ManifoldModel=CycleModel
engine.objective=objective

class CycleTrainer(engine.ManifoldTrainer):
    def predict_curve(self,S,genes,cell_ids,anchor=.2):
        """Optional S-only curve projection; never accepts query U or labels."""
        encoded=super().predict(S,genes,cell_ids)
        order={str(g):i for i,g in enumerate(genes)}
        S=np.asarray(S)[:,[order[str(g)] for g in self.genes]]
        grid=np.arange(720,dtype=np.float32)/720
        with torch.no_grad():
            curve=self.model.decode_phase(self._tensor(grid))[:,self.model.n_genes:]
            scale=self.model.state_scale[self.model.n_genes:]
            decoded=(curve/scale).cpu().numpy().astype(np.float64)
        # Explicit float64 differences give query batch / singleton invariance.
        positions=[]
        for row,p0 in zip(S,encoded['phase']):
            error=np.mean((decoded-row.astype(np.float64)/scale.cpu().numpy())**2,axis=1)
            delta=(grid-p0+.5)%1-.5
            index=np.argmin(error+anchor*(2*np.pi*delta)**2)
            positions.append(grid[index])
        p=self._tensor(np.asarray(positions,np.float32));z=circle(p)
        return self._prediction(z,z,np.asarray(cell_ids))

CYCLE_VARIANTS={
    'cyc_legacy':{},
    'cyc_pca':{'initialization':'pca'},
    'cyc_weakdyn':{'dynamics_weight':.01},
    'cyc_uquarter':{'u_weight':.25},
    'cyc_fourier2':{'harmonics':2,'curvature_weight':1e-5},
    'cyc_fourier3':{'harmonics':3,'curvature_weight':1e-5},
    'cyc_fourier2_pca':{'harmonics':2,'curvature_weight':1e-5,'initialization':'pca'},
    'cyc_direction':{'dynamics_weight':.01,'direction_weight':.03},
    'cyc_loose':{'anchor_weight':.002,'geometry_weight':.01},
    'cyc_fourier2_s':{'harmonics':2,'curvature_weight':1e-5,'u_weight':.25},
    'cyc_robust':{'robust_delta':1.},
    'cyc_robust_pca':{'robust_delta':1.,'initialization':'pca'},
    'cyc_pcs5':{'graph_pcs':5},
    'cyc_pcs10':{'graph_pcs':10},
    'cyc_neigh60':{'neighbors':60},
    'cyc_neigh15':{'neighbors':15},
    'cyc_long':{'joint_steps':6900},
    'cyc_weak_anchor':{'anchor_weight':.002},
    'cyc_log':{'log_reconstruction':True},
    'cyc_log_pca':{'log_reconstruction':True,'initialization':'pca'},
    'cyc_log_fourier2':{'log_reconstruction':True,'harmonics':2,'curvature_weight':1e-5},
}

def initial_phase(a,cfg):
    _,_,positions=fit_transform(a['S'],a['genes'],cfg)
    graph=build_reference_graph(positions,cfg.neighbors,cfg.seed)
    if cfg.initialization=='pca':
        coords=positions[:,:2]/np.maximum(positions[:,:2].std(0),1e-8)
        angle=np.arctan2(coords[:,0],coords[:,1])
    elif cfg.initialization=='spectral':angle=np.arctan2(graph.initial_z[:,0],graph.initial_z[:,1])
    else:raise ValueError(cfg.initialization)
    selected=int(a['n_selected_genes']);S,U=a['S'][:,:selected],a['U'][:,:selected]
    design=np.column_stack([np.ones(len(S)),np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])
    coefficients=np.linalg.lstsq(design,S,rcond=None)[0]
    tangent=2*np.pi*np.column_stack([np.zeros(len(S)),np.cos(angle),-np.sin(angle),2*np.cos(2*angle),-2*np.sin(2*angle)])@coefficients
    gamma=np.clip(2*np.pi*U.mean(0)/np.maximum(S.mean(0),.01),.1,100)
    kinetic=2*np.pi*U-gamma*S;scale=np.maximum(S.std(0),cfg.state_scale_floor)
    score=float(np.sum(tangent*kinetic/scale**2))
    phase0=(angle/(2*np.pi)*(1 if score>=0 else -1))%1
    phase0=((phase0-phase0[int(a['root_index'])]+cfg.root_phase)%1).astype(np.float32)
    return phase0,dict(method=cfg.initialization,orientation_score=score,graph=graph.diagnostics)

def run_cycle(a,job,out,device):
    from .serialization import save_model
    parameters=dict(CYCLE_VARIANTS[job.get('template',job['variant'])]);parameters.update(job.get('overrides',{}))
    settings=dict(neighbors=30,graph_pcs=20,warmup_steps=300,joint_steps=2400,
        root_weight=.1,root_phase=.05,seed=job['seed']);settings.update(parameters)
    cfg=CycleConfig(**settings)
    if (out/'model/checkpoint.pt').exists():
        return CycleTrainer.load(out/'model',device=device)
    p0,diagnostic=initial_phase(a,cfg);root=np.arange(len(a['S']))==a['root_index']
    write(out/'INITIALIZATION.json',diagnostic)
    np.savez_compressed(out/'initialization.npz',cell_ids=a['cell_ids'],phase=p0)
    write(out/'training_protocol.json',dict(config=cfg.to_dict(),fixed_budget=True,
        root_identity_only=True,reference_labels_used=False,
        backend='Old GraphSAGE circle with optional decoder/loss variants; topology specified for this control'))
    model=CycleTrainer(cfg,device=device).fit(a['S'],a['U'],a['genes'],a['cell_ids'],a['W_0_mask'],
        n_selected_genes=int(a['n_selected_genes']),initial_phase=p0,root_mask=root)
    prediction=model.reference_prediction()
    np.savez_compressed(out/'prediction.npz',**prediction)
    write(out/'diagnostics.json',model.diagnostics(integrate=False))
    save_model(model,out/'model')
    write(out/'status.json',dict(complete=True,steps=len(model.history),stop='fixed budget',convergence_claimed=False))
    return model
