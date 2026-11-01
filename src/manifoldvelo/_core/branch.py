"""Development variants of the successful open-chart v2 model."""
from dataclasses import dataclass
import copy
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .cycle_base.graph import names
from .branch_base.em import EMBranchConfig,EMBranchManifoldTrainer

@dataclass
class BranchConfig(EMBranchConfig):
    direction_weight: float=0.
    shared_time_weight: float=0.
    maximum_projection_step: float=1.
    u_weight: float=1.
    shared_velocity_weight: float=0.

class BranchTrainer(EMBranchManifoldTrainer):
    config_type=BranchConfig
    query_numerics='cpu_float64_projection_and_fields_v2'

    def initialize_extra(self):
        super().initialize_extra()
        self._stable_query_model=None

    def predict(self,S,genes,cell_ids):
        """S-only projection with a fixed CPU float64 decoder for reproducibility.

        The trained weights, objective and training-cell assignments are intact.
        Direct per-query differences avoid cancellation in squared distances.
        """
        _,positions=self.transform.apply(S,genes)
        ids=names(cell_ids,len(positions),'query cell IDs')
        if set(ids)&set(self.cell_ids):raise ValueError('Query and reference IDs overlap')
        count=min(self.config.neighbors,len(self.cell_ids))
        distances,index=NearestNeighbors(n_neighbors=count,n_jobs=1).fit(self.positions).kneighbors(positions)
        weights=np.exp(-distances/np.maximum(distances[:,-1:],1e-8));weights/=weights.sum(1,keepdims=True)
        t0=(self.charts.initial_time[index]*weights).sum(1)
        membership=(self.charts.membership[index]*weights[:,:,None]).sum(1)
        if self._stable_query_model is None:
            self._stable_query_model=copy.deepcopy(self.model).to(device='cpu',dtype=torch.float64).eval()
        grid=np.linspace(0,1,self.config.projection_points,dtype=np.float64)
        order={str(g):i for i,g in enumerate(genes)}
        observed=np.asarray(S,dtype=np.float64)[:,[order[str(g)] for g in self.genes]]
        times=np.zeros((len(S),self.model.branches),np.float32)
        with torch.no_grad():
            for b in range(self.model.branches):
                scale=self.state_scale(b)[self.model.genes:].detach().cpu().numpy().astype(float)
                curve=self._stable_query_model.decode(torch.from_numpy(grid),b)[:,self.model.genes:].numpy()/scale
                for i,row in enumerate(observed/scale):
                    cost=np.mean((curve-row)**2,axis=1)+self.config.anchor_weight*(grid-float(t0[i]))**2
                    times[i,b]=grid[int(np.argmin(cost))]
        model=self._stable_query_model;g=model.genes
        weight=membership.astype(np.float64)*self.charts.membership.sum(0)[None,:]
        weight/=np.maximum(weight.sum(1,keepdims=True),1e-12)
        state=np.zeros((len(S),2*g),np.float64);velocity=np.zeros_like(state);derivative=np.zeros_like(state)
        for b in range(model.branches):
            t=torch.from_numpy(times[:,b].astype(np.float64))
            with torch.no_grad():
                curve=model.decode(t,b);rhs=model.rhs(t,curve,b)
                state+=weight[:,b,None]*curve.numpy();velocity+=weight[:,b,None]*rhs.numpy()
            _,tangent=model.curve(t,b)
            derivative+=weight[:,b,None]*tangent.detach().numpy()
        per_chart=times.copy();per_chart[membership==0]=np.nan
        return dict(cell_ids=ids,genes=self.genes.copy(),pseudotime=np.sum(times*weight,axis=1).astype(np.float32),
            initial_graph_time=t0,chart_times=per_chart,chart_membership=membership,
            S_reconstructed=state[:,g:].astype(np.float32),U_predicted=state[:,:g].astype(np.float32),
            velocity_S_kinetic=velocity[:,g:].astype(np.float32),velocity_U_kinetic=velocity[:,:g].astype(np.float32),
            velocity_S_curve=derivative[:,g:].astype(np.float32),velocity_U_curve=derivative[:,:g].astype(np.float32))

    def state_scale(self,branch):
        scale=super().state_scale(branch)
        if self.config.u_weight==1:return scale
        weight=torch.ones_like(scale);weight[:self.model.genes]=np.sqrt(self.config.u_weight)
        return scale/weight.clamp_min(1e-4)

    def objective(self,batch,grid,joint):
        total,terms=super().objective(batch,grid,joint)
        direction=total*0
        if joint and self.config.direction_weight:
            active=self.model.active.bool()
            for b in range(self.model.branches):
                y,dy=self.model.curve(grid,b,create_graph=True)
                rhs=self.model.rhs(grid,y,b)
                scale=self.state_scale(b)[active]
                tangent=dy[:,active]/scale;velocity=rhs[:,active]/scale
                cosine=torch.nn.functional.cosine_similarity(tangent,velocity,dim=-1,eps=1e-6)
                direction=direction+(1-cosine).mean()/self.model.branches
            total=total+self.config.direction_weight*direction
        terms['direction']=direction
        shared_velocity=total*0
        if joint and self.config.shared_velocity_weight:
            # Only compare charts for the SAME observed shared-trunk cells.
            # Distinct daughter lineages are never forced to point together.
            clocks=self.training_clock_values(batch)
            active=self.model.active.bool();pairs=0
            for left in range(self.model.branches):
                for right in range(left+1,self.model.branches):
                    shared=torch.where(self.membership[batch,left]&self.membership[batch,right])[0][:256]
                    if len(shared)==0:continue
                    _,dl=self.model.curve(clocks[shared,left],left,create_graph=True)
                    _,dr=self.model.curve(clocks[shared,right],right,create_graph=True)
                    scale=self.model.scale[active]
                    shared_velocity=shared_velocity+(1-torch.nn.functional.cosine_similarity(
                        dl[:,active]/scale,dr[:,active]/scale,dim=-1,eps=1e-6)).mean()
                    pairs+=1
            shared_velocity=shared_velocity/max(pairs,1)
            total=total+self.config.shared_velocity_weight*shared_velocity
        terms['shared_velocity']=shared_velocity
        return total,terms

    def project_training_times(self,step):
        if self.config.shared_time_weight==0 and self.config.maximum_projection_step>=1:
            return super().project_training_times(step)
        cfg,model=self.config,self.model
        grid=self.tensor(np.linspace(0,1,cfg.projection_points));active=model.active.bool()
        with torch.no_grad():
            count=self.membership.sum(1).clamp_min(1)
            shared=(self.assignments*self.membership).sum(1)/count
            before=[];after=[];moved=[]
            for b in range(model.branches):
                ids=torch.where(self.membership[:,b])[0];scale=self.state_scale(b)[active]
                curve=model.decode(grid,b)[:,active]/scale
                sq=curve.square().mean(-1)
                for block in ids.split(512):
                    y=self.y[block][:,active]/scale;old=self.assignments[block,b].clone()
                    costs=(y.square().mean(-1,keepdim=True)+sq[None]-2*(y@curve.T)/curve.shape[1]).clamp_min(0)
                    costs+=cfg.anchor_weight*(grid[None]-self.t0[block,None]).square()
                    shared_weight=(count[block]>1).float()*cfg.shared_time_weight
                    costs+=shared_weight[:,None]*(grid[None]-shared[block,None]).square()
                    costs=costs.masked_fill((grid[None]-old[:,None]).abs()>cfg.maximum_projection_step,float('inf'))
                    best,index=costs.min(-1)
                    old_curve=model.decode(old,b)[:,active]/scale
                    old_cost=(y-old_curve).square().mean(-1)+cfg.anchor_weight*(old-self.t0[block]).square()
                    old_cost+=shared_weight*(old-shared[block]).square()
                    improve=(best<old_cost)&(block!=self.charts.root_index)
                    new=torch.where(improve,grid[index],old)
                    self.assignments[block,b]=new
                    before.append(old_cost.mean());after.append(torch.where(improve,best,old_cost).mean());moved.append((new-old).abs().mean())
            self.assignments[self.charts.root_index]=0
        row=dict(step=step,before=float(torch.stack(before).mean()),after=float(torch.stack(after).mean()),
                 mean_time_update=float(torch.stack(moved).mean()))
        assert row['after']<=row['before']+1e-6
        self.projection_history.append(row)

    def protocol_details(self):
        info=super().protocol_details()
        info.update(development_extension='Optional normalized velocity-direction loss, shared-cell time agreement, bounded E-step movement and shared-trunk tangent agreement',
                    training_labels_used=False)
        return info
