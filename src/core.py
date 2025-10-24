
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

def bxor(a: int, b: int) -> int: return (a ^ b) & 1
def parity(bits) -> int:
    s=0
    for v in bits: s ^= (v & 1)
    return s
def rule110(l: int, c: int, r: int) -> int:
    idx=(l<<2)|(c<<1)|r
    return (0b01101110 >> idx) & 1

@dataclass
class Cell:
    x_prev:int; x_cur:int; a:int; p:int
    m:Dict[str,int]; h:Tuple[int,int,int,int,int]
    y_prev:Tuple[int,...]; y_cur:Tuple[int,...]
    s:int; theta:int; b:int

@dataclass
class LawTrack:
    chi:int=3; h_L_mode:str="id"; hard_consensus:bool=False; lex_mode:str="GOLD"
    def sigma(self, y_cur:Tuple[int,...])->int: return parity(y_cur)
    def theta(self, L:int,C:int,R:int,s:int)->int: return parity([L,C,R,s])
    def G(self, phi:Tuple[int,...], g:int, theta:int)->int:
        s=0
        for v in phi: s ^= (v&1)
        s ^= (g&1); s ^= (theta&1)
        return s & 1
    def Guard_allowed(self, pL:int,pC:int,pR:int,B:int,theta:int)->Tuple[int,int]:
        if self.lex_mode=="FWO":
            allowed1 = ((B | (pL|pC|pR)) & 1)
        else:
            allowed1 = (B & ((pL|pC|pR|theta)&1)) & 1
        return (1, allowed1)
    def U_free(self, mem_bits:Tuple[int,...], h:Tuple[int,...])->Tuple[Tuple[int,...],Tuple[int,...]]: return mem_bits, h
    def ports(self, i:int, cells:List["Cell"])->Tuple[int,int,int,int]:
        N=len(cells); L=(i-1)%N; R=(i+1)%N
        bigL=1 if (cells[L].m.get("g",0)==1) else 0
        bigR=1 if (cells[R].m.get("g",0)==1) else 0
        M_lat=1 if (bigL or bigR) else 0
        D_lat=cells[L].a if bigL else (cells[R].a if bigR else 0)
        M_up=cells[i].b; D_up=D_lat
        return M_up, D_up, M_lat, D_lat

@dataclass
class Universe:
    N:int; mu:int; L:LawTrack; cells:List[Cell]
    def left(self,i:int)->int: return (i-1)%self.N
    def right(self,i:int)->int: return (i+1)%self.N
    def recompute_sigma_theta(self)->None:
        for i in range(self.N): self.cells[i].s=self.L.sigma(self.cells[i].y_cur)
        for i in range(self.N):
            Lc=self.cells[self.left(i)].x_cur; Cc=self.cells[i].x_cur; Rc=self.cells[self.right(i)].x_cur
            self.cells[i].theta=self.L.theta(Lc,Cc,Rc,self.cells[i].s)
    def features_phi(self,i:int)->Tuple[int,...]:
        Lc=self.cells[self.left(i)]; Cc=self.cells[i]; Rc=self.cells[self.right(i)]
        return (Lc.x_cur, Cc.x_cur, Rc.x_cur, Cc.s, Cc.m["B"], bxor(Cc.x_prev, Cc.x_cur))
    def p_hat(self,i:int)->int:
        Cc=self.cells[i]; return self.L.G(self.features_phi(i), Cc.m.get("g",0), Cc.theta)&1
    def propose(self)->List[int]:
        pHat=[self.p_hat(i) for i in range(self.N)]
        for i in range(self.N): self.cells[i].p = bxor(self.cells[i].p, pHat[i])
        return pHat
    def commit_phase(self,pHat:List[int],phase:int)->None:
        # non-intervention: compute ports but don't persist
        for i in range(self.N):
            if (i%self.L.chi)!=phase: continue
            c=self.cells[i]; mem_bits=(c.m["g"],c.m["B"],c.m.get("M_up",0),c.m.get("D_up",0),c.m.get("M_lat",0),c.m.get("D_lat",0))
            _,_ = self.L.U_free(mem_bits, c.h)
        for i in range(self.N):
            if (i%self.L.chi)!=phase: continue
            c=self.cells[i]; iL=self.left(i); iR=self.right(i)
            a0,a1=self.L.Guard_allowed(pHat[iL],pHat[i],pHat[iR],c.m["B"],c.theta)
            eta=(a1 & pHat[i]) & 1; k=(pHat[iL]^pHat[iR])&1; rho=0
            u=1 if (self.L.hard_consensus and (a1==0)) else 0; dB=0
            if u==1:
                j=self.right(i); self.cells[i].b=bxor(self.cells[i].b,1); self.cells[j].b=bxor(self.cells[j].b,1)
            if dB==1: c.m["B"]=bxor(c.m["B"],1)
            c.a=bxor(c.a, eta); c.h=(eta,k,rho,u,dB)
    def phys_micro(self)->None:
        new_x=[0]*self.N
        for i in range(self.N):
            Lc=self.cells[self.left(i)].x_cur; Cc=self.cells[i].x_cur; Rc=self.cells[self.right(i)].x_cur
            new_x[i]=bxor(self.cells[i].x_prev, rule110(Lc,Cc,Rc))
        for i in range(self.N): self.cells[i].x_prev, self.cells[i].x_cur = self.cells[i].x_cur, new_x[i]
        for i in range(self.N): self.cells[i].y_prev, self.cells[i].y_cur = self.cells[i].y_cur, self.cells[i].y_prev
    def step(self,t_phase:int)->None:
        self.recompute_sigma_theta(); pHat=self.propose()
        if self.L.h_L_mode=="id":
            for rph in range(self.L.chi): self.commit_phase(pHat,(t_phase+rph)%self.L.chi)
            self.phys_micro()
        else:
            self.phys_micro()
            for rph in range(self.L.chi): self.commit_phase(pHat,(t_phase+rph)%self.L.chi)
        self.recompute_sigma_theta()
    def inv_phys_micro(self)->None:
        old_x_prev=[0]*self.N
        for i in range(self.N):
            Lc=self.cells[self.left(i)].x_prev; Cc=self.cells[i].x_prev; Rc=self.cells[self.right(i)].x_prev
            old_x_prev[i]=bxor(self.cells[i].x_cur, rule110(Lc,Cc,Rc))
        for i in range(self.N): self.cells[i].x_cur, self.cells[i].x_prev = self.cells[i].x_prev, old_x_prev[i]
        for i in range(self.N): self.cells[i].y_prev, self.cells[i].y_cur = self.cells[i].y_cur, self.cells[i].y_prev
    def inv_commit_phase(self,phase:int)->None:
        for i in range(self.N):
            if (i%self.L.chi)!=phase: continue
            c=self.cells[i]; eta,k,rho,u,dB=(c.h+(0,0,0,0,0))[:5]
            if u==1:
                j=self.right(i); self.cells[i].b=bxor(self.cells[i].b,1); self.cells[j].b=bxor(self.cells[j].b,1)
            if dB==1: c.m["B"]=bxor(c.m["B"],1)
            c.a=bxor(c.a, eta); c.h=(0,0,0,0,0)
    def inv_propose(self)->None:
        self.recompute_sigma_theta(); pHat=[self.p_hat(i) for i in range(self.N)]
        for i in range(self.N): self.cells[i].p = bxor(self.cells[i].p, pHat[i])
    def step_inverse(self,t_phase:int)->None:
        if self.L.h_L_mode=="id":
            self.inv_phys_micro()
            for rph in reversed(range(self.L.chi)): self.inv_commit_phase((t_phase+rph)%self.L.chi)
            self.inv_propose()
        else:
            for rph in reversed(range(self.L.chi)): self.inv_commit_phase((t_phase+rph)%self.L.chi)
            self.inv_phys_micro(); self.inv_propose()
        self.recompute_sigma_theta()
    def inverse_check(self,t_phase:int)->bool:
        import copy
        snap=copy.deepcopy(self.cells); self.step(t_phase); self.step_inverse(t_phase)
        for a,b in zip(snap,self.cells):
            if not (a.x_prev==b.x_prev and a.x_cur==b.x_cur and a.a==b.a and a.p==b.p and
                    a.m==b.m and a.y_prev==b.y_prev and a.y_cur==b.y_cur and a.b==b.b and a.h==b.h):
                return False
        return True
    def freedom_ratio(self)->float:
        self.recompute_sigma_theta(); pHat=[self.p_hat(i) for i in range(self.N)]
        cnt=0
        for i in range(self.N):
            a0,a1=self.L.Guard_allowed(pHat[self.left(i)],pHat[i],pHat[self.right(i)],self.cells[i].m["B"],self.cells[i].theta)
            if a1==1: cnt+=1
        return cnt/float(self.N)
    def a_equal3_ratio(self)->float:
        a=[c.a&1 for c in self.cells]; eq=0
        for i in range(self.N):
            if a[self.left(i)]==a[i]==a[self.right(i)]: eq+=1
        return eq/float(self.N)

def make_lawtrack_default(chi:int=3, h_L_mode:str="id", hard_consensus:bool=False, lex_mode:str="GOLD")->LawTrack:
    return LawTrack(chi=chi, h_L_mode=h_L_mode, hard_consensus=hard_consensus, lex_mode=lex_mode)

def make_universe(N:int=64, mu:int=5, L:Optional[LawTrack]=None)->Universe:
    if L is None:
        chi = 2 if (N%2==0) else 3
        L = make_lawtrack_default(chi=chi, h_L_mode="id", hard_consensus=False)
    cells=[]
    for i in range(N):
        x_prev=(i>>0)&1; x_cur=(i>>1)&1; a=(i>>2)&1; p=(i>>3)&1
        m={"g":(i>>4)&1, "B":(i>>5)&1, "M_up":0, "D_up":0, "M_lat":0, "D_lat":0}
        y_prev=tuple(((i+j)&1) for j in range(mu))
        y_cur =tuple((((i+1)+j)&1) for j in range(mu))
        s=0; theta=0; b=(i&1); h=(0,0,0,0,0)
        cells.append(Cell(x_prev,x_cur,a,p,m,h,y_prev,y_cur,s,theta,b))
    U=Universe(N=N, mu=mu, L=L, cells=cells); U.recompute_sigma_theta(); return U
