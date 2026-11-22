"""Numerical and leakage checks required before starting the full benchmark."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import protocol as P
from unified_dynamics import UnifiedECMUDE, physical
from exp_unified import rk4_rc, paired, model_comparison
from ecm import ECMParams

def reference(p,x,stats,dt,q,mode):
    I=-(x[...,0]*stats['I_std']+stats['I_mean'])
    T=x[...,1]*stats['T_std']+stats['T_mean']
    Z=(x[...,2]*stats['SoC_std']+stats['SoC_mean']).clamp(0,1)
    r0=F.softplus(p[6])+1e-4;r=F.softplus(p[7])+1e-4;c=F.softplus(p[8])+1
    eta=1+.1*torch.tanh(p[9])
    def net(v,z,i,t):
        xx=torch.stack([v,i,z,t],-1)
        h=F.gelu(F.linear(xx,p[10:138].reshape(32,4),p[138:170]))
        h=F.gelu(F.linear(h,p[170:1194].reshape(32,32),p[1194:1226]))
        return p[1259]*F.linear(h,p[1226:1258][None],p[1258:1259]).squeeze(-1)
    def rhs(v,z,i,t):
        return -v/(r*c)+i/c+(net(v,z,i,t) if mode==2 else 0),-eta*i/q
    v=torch.zeros(len(x),dtype=x.dtype);z=Z[:,0];pred=[]
    for k in range(x.shape[1]):
        i=I[:,k];t=T[:,k];xx=2*z.clamp(0,1)-1;bs=[torch.ones_like(xx),xx]
        for j in range(2,6):bs.append(2*xx*bs[-1]-bs[-2])
        y=sum(p[j]*bs[j] for j in range(6))-r0*i-v
        if mode==1:y=y+net(v,z.clamp(0,1),i,t)
        pred.append((y-stats['V_mean'])/stats['V_std'])
        a,b=rhs(v,z,i,t);cc,d=rhs(v+dt*a/2,z+dt*b/2,i,t)
        e,f=rhs(v+dt*cc/2,z+dt*d/2,i,t);g,h=rhs(v+dt*e,z+dt*f,i,t)
        v=v+dt*(a+2*cc+2*e+g)/6;z=z+dt*(b+2*d+2*f+h)/6
    return torch.stack(pred,1)

class ProtocolTests(unittest.TestCase):
    def test_causal_soc_and_sign(self):
        z=P.causal_soc([0,.1,.05],qnom_Ah=2.,z0=.8)
        np.testing.assert_allclose(z,[.8,.75,.775])
        np.testing.assert_equal(z[:2],P.causal_soc([0,.1,9.],qnom_Ah=2.,z0=.8)[:2])

    def test_guard_and_full_coverage(self):
        cfg=P.DATASETS['panasonic'];starts,tr,va=P.temporal_split(100031,cfg)
        self.assertLessEqual(starts[tr[-1]]+cfg.window_length,starts[va[0]])
        self.assertEqual(starts[-1]+cfg.window_length,100031)

    def test_forward_and_discrete_gradients(self):
        stats=dict(I_mean=0.,I_std=1.,T_mean=0.,T_std=25.,SoC_mean=.5,SoC_std=.3,V_mean=3.6,V_std=.2)
        params=ECMParams(ocv_coeffs=np.array([3.7,.4,.01,.002,.001,.001]),R0=.03,R1=.02,C1=100.)
        cfg=P.DATASETS['panasonic'];torch.manual_seed(9)
        x=torch.randn(2,7,3,dtype=torch.float64)*.1;x[...,1]=1.
        for mode in (0,1,2):
            model=UnifiedECMUDE(stats,params,cfg,mode).double()
            # Deliberately exercise a nontrivial correction and eta derivative.
            if mode:
                with torch.no_grad():model.p[10:1259]*=2;model.p[1259]=.07
            p=model.p.detach().clone().requires_grad_();weight=torch.randn(2,7,dtype=torch.float64)
            y=model(x);yr=reference(p,x,stats,model.dt,model.q,mode)
            np.testing.assert_allclose(y.detach(),yr.detach(),rtol=1e-10,atol=1e-10)
            (y*weight).sum().backward();(yr*weight).sum().backward()
            np.testing.assert_allclose(model.p.grad,p.grad,rtol=2e-8,atol=2e-9)
            if mode==0:
                r0,r,c,_=physical(p.detach().numpy())
                np.testing.assert_allclose(rk4_rc(-x[0,:,0].numpy(),r,c,model.dt),
                    # Solve the polarization recurrence directly.
                    polarization(-x[0,:,0].numpy(),r,c,model.dt),rtol=1e-10,atol=1e-12)

    def test_exact_wilcoxon_five_seeds(self):
        self.assertEqual(paired(np.arange(5.),np.arange(5.)+1)['p_raw'],.0625)

    def test_no_pseudoreplication_for_two_nls_fits(self):
        result=model_comparison(np.zeros(5),np.ones(5),'ecm_1rc_nls','ecm_2rc_nls')
        self.assertIsNone(result['p_raw'])
        self.assertIsNone(result['significant_005'])
        self.assertEqual(result['n'],0)

def polarization(i,r,c,dt):
    v=0.;out=[]
    for current in i:
        out.append(v)
        def f(x):return -x/(r*c)+current/c
        a=f(v);b=f(v+dt*a/2);cc=f(v+dt*b/2);d=f(v+dt*cc)
        v+=dt*(a+2*b+2*cc+d)/6
    return out

if __name__=='__main__':unittest.main()
