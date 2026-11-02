"""Compiled classical RK4 with an exact discrete reverse-mode derivative.

The learned model has the existing ECMUDE architecture (2x32 GELU MLP,
softplus physical parameters, Chebyshev OCV, eta=1+.1*tanh(raw)).
Only the implementation of integration/autograd is accelerated. The
discrete derivative is tested against a transparent PyTorch RK4 reference.
Driving inputs are held at sample k throughout [t_k,t_{k+1}); there is no
look-ahead input and no floating-point index rounding at solver stages.
"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

try:
    import numba
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / ".unified_deps"))
    import numba
from numba import njit
from ude import build_ecm_ude

@njit(cache=True)
def gelu(x):
    return .5*x*(1+math.erf(x/math.sqrt(2.)))

@njit(cache=True)
def dgelu(x):
    return .5*(1+math.erf(x/math.sqrt(2.))) + x*math.exp(-.5*x*x)/math.sqrt(2*math.pi)

@njit(cache=True)
def network(p, v, i, z, t):
    x = np.array([v,i,z,t])
    a = np.empty(32); b = np.empty(32)
    for j in range(32):
        s=p[138+j]
        for k in range(4): s+=p[10+j*4+k]*x[k]
        a[j]=gelu(s)
    for j in range(32):
        s=p[1194+j]
        for k in range(32): s+=p[170+j*32+k]*a[k]
        b[j]=gelu(s)
    val=p[1258]
    for j in range(32):val+=p[1226+j]*b[j]
    return p[1259]*val

@njit(cache=True)
def network_vjp(p,v,i,z,t,g,dp):
    x=np.array([v,i,z,t]);a=np.empty(32);b=np.empty(32)
    u=np.empty(32);w=np.empty(32)
    for j in range(32):
        s=p[138+j]
        for k in range(4):s+=p[10+j*4+k]*x[k]
        u[j]=s;a[j]=gelu(s)
    val=p[1258]
    for j in range(32):
        s=p[1194+j]
        for k in range(32):s+=p[170+j*32+k]*a[k]
        w[j]=s;b[j]=gelu(s);val+=p[1226+j]*b[j]
    dp[1259]+=g*val
    gn=g*p[1259]; dp[1258]+=gn
    da=np.zeros(32)
    for j in range(32):
        dp[1226+j]+=gn*b[j]
        gw=gn*p[1226+j]*dgelu(w[j]);dp[1194+j]+=gw
        for k in range(32):
            dp[170+j*32+k]+=gw*a[k];da[k]+=gw*p[170+j*32+k]
    dx=np.zeros(4)
    for j in range(32):
        gu=da[j]*dgelu(u[j]);dp[138+j]+=gu
        for k in range(4):
            dp[10+j*4+k]+=gu*x[k];dx[k]+=gu*p[10+j*4+k]
    return dx[0],dx[2]

@njit(cache=True)
def physical(p):
    return (np.logaddexp(0.,p[6])+1e-4,
            np.logaddexp(0.,p[7])+1e-4,
            np.logaddexp(0.,p[8])+1.,1.+.1*math.tanh(p[9]))

@njit(cache=True)
def ocv_basis(z):
    x=2.*min(1.,max(0.,z))-1.;b=np.zeros(6);d=np.zeros(6)
    b[0]=1.;b[1]=x;d[1]=2.
    for k in range(2,6):
        b[k]=2*x*b[k-1]-b[k-2]
        d[k]=4*b[k-1]+2*x*d[k-1]-d[k-2]
    if z<0. or z>1.:d[:]=0.
    return b,d

@njit(cache=True)
def rhs(p,v,z,i,t,mode,r,c,eta,q):
    dv=-v/(r*c)+i/c
    if mode==2:dv+=network(p,v,i,z,t)
    return dv,-eta*i/q

@njit(cache=True)
def rhs_vjp(p,v,z,i,t,mode,r,c,eta,q,gv,gz,dp):
    dp[7]+=gv*v/(r*r*c)
    dp[8]+=gv*(v/(r*c*c)-i/(c*c))
    dp[9]+=-gz*i/q
    dv=-gv/(r*c);dz=0.
    if mode==2:
        av,az=network_vjp(p,v,i,z,t,gv,dp);dv+=av;dz+=az
    return dv,dz

@njit(cache=True)
def integrate(p,I,T,Z,dt,q,mode):
    B,L=I.shape; traj=np.empty((B,L,2));out=np.empty((B,L))
    r0,r,c,eta=physical(p)
    for b in range(B):
        v=0.;z=Z[b,0]
        for k in range(L):
            i=I[b,k];t=T[b,k];traj[b,k,0]=v;traj[b,k,1]=z
            basis,_=ocv_basis(z);value=-r0*i-v
            for j in range(6):value+=p[j]*basis[j]
            if mode==1:value+=network(p,v,i,min(1.,max(0.,z)),t)
            out[b,k]=value
            if k+1==L:continue
            a,bz=rhs(p,v,z,i,t,mode,r,c,eta,q)
            c1,d=rhs(p,v+dt*a/2,z+dt*bz/2,i,t,mode,r,c,eta,q)
            e,f=rhs(p,v+dt*c1/2,z+dt*d/2,i,t,mode,r,c,eta,q)
            g,h=rhs(p,v+dt*e,z+dt*f,i,t,mode,r,c,eta,q)
            v+=dt*(a+2*c1+2*e+g)/6;z+=dt*(bz+2*d+2*f+h)/6
    return out,traj

@njit(cache=True)
def reverse(p,I,T,traj,dt,q,mode,grad):
    dp=np.zeros(len(p));r0,r,c,eta=physical(p);B,L=I.shape
    for b in range(B):
        lv=0.;lz=0.
        for k in range(L-1,-1,-1):
            v=traj[b,k,0];z=traj[b,k,1];i=I[b,k];t=T[b,k]
            if k<L-1:
                a,az=rhs(p,v,z,i,t,mode,r,c,eta,q)
                bv,bz=v+dt*a/2,z+dt*az/2
                c1,d=rhs(p,bv,bz,i,t,mode,r,c,eta,q)
                ev,ez=v+dt*c1/2,z+dt*d/2
                e,f=rhs(p,ev,ez,i,t,mode,r,c,eta,q)
                gv,gz=v+dt*e,z+dt*f
                la,la_z=lv*dt/6,lz*dt/6
                lc,lc_z=lv*dt/3,lz*dt/3
                le,le_z=lv*dt/3,lz*dt/3
                lg,lg_z=lv*dt/6,lz*dt/6
                av,az1=rhs_vjp(p,gv,gz,i,t,mode,r,c,eta,q,lg,lg_z,dp)
                lv+=av;lz+=az1;le+=dt*av;le_z+=dt*az1
                av,az1=rhs_vjp(p,ev,ez,i,t,mode,r,c,eta,q,le,le_z,dp)
                lv+=av;lz+=az1;lc+=dt*av/2;lc_z+=dt*az1/2
                av,az1=rhs_vjp(p,bv,bz,i,t,mode,r,c,eta,q,lc,lc_z,dp)
                lv+=av;lz+=az1;la+=dt*av/2;la_z+=dt*az1/2
                av,az1=rhs_vjp(p,v,z,i,t,mode,r,c,eta,q,la,la_z,dp)
                lv+=av;lz+=az1
            loss=grad[b,k];basis,deriv=ocv_basis(z)
            for j in range(6):dp[j]+=loss*basis[j];lz+=loss*p[j]*deriv[j]
            dp[6]+=-loss*i;lv-=loss
            if mode==1:
                av,az1=network_vjp(p,v,i,min(1.,max(0.,z)),t,loss,dp)
                lv+=av
                if 0<=z<=1:lz+=az1
    for j in (6,7,8):dp[j]*=1./(1.+math.exp(-p[j]))
    dp[9]*=.1*(1.-math.tanh(p[9])**2)
    return dp

class RK4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,x,stats,dt,q,mode):
        X=x.detach().cpu().numpy().astype(np.float64)
        I=-(X[...,0]*stats['I_std']+stats['I_mean'])
        T=X[...,1]*stats['T_std']+stats['T_mean']
        Z=np.clip(X[...,2]*stats['SoC_std']+stats['SoC_mean'],0,1)
        P=p.detach().cpu().numpy().astype(np.float64)
        y,traj=integrate(P,I,T,Z,dt,q,mode)
        ctx.args=P,I,T,traj,dt,q,mode,stats['V_std']
        ctx.dtype=p.dtype
        return torch.from_numpy((y-stats['V_mean'])/stats['V_std']).to(x.dtype)

    @staticmethod
    def backward(ctx,g):
        p,I,T,traj,dt,q,mode,std=ctx.args
        dp=reverse(p,I,T,traj,dt,q,mode,g.numpy().astype(np.float64)/std)
        return torch.from_numpy(dp).to(ctx.dtype),None,None,None,None,None

class UnifiedECMUDE(nn.Module):
    def __init__(self,stats,params,cfg,mode):
        super().__init__()
        base=build_ecm_ude(stats=stats,ecm_params=params)
        f=base.ode_func;n=f.nn_correction
        pieces=[base.ocv.coeffs,f._R0_raw.reshape(1),f._R1_raw.reshape(1),
                f._C1_raw.reshape(1),f._eta_raw.reshape(1)]
        if mode:
            pieces.extend([n.net[0].weight.flatten(),n.net[0].bias,
                           n.net[2].weight.flatten(),n.net[2].bias,
                           n.net[4].weight.flatten(),n.net[4].bias,n.gain.reshape(1)])
        self.p=nn.Parameter(torch.cat(pieces).detach().clone())
        self.stats=stats;self.dt=cfg.dt_seconds
        self.q=cfg.nominal_capacity_Ah*3600.;self.mode=mode

    def forward(self,x):
        return RK4Function.apply(self.p,x,self.stats,self.dt,self.q,self.mode)

    def physical_params_dict(self):
        p=self.p.detach().numpy().astype(np.float64);r0,r,c,eta=physical(p)
        return dict(R0=r0,R1=r,C1=c,eta=eta,ocv_coeffs=p[:6].tolist())
