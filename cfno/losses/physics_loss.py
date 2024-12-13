import torch
import numpy as np

from qmat.lagrange import LagrangeApproximation

### TODOs:
### 1. needs to be adapted to deal with (mini-)batches -> einsum, compare to Fourier layer and adapt
### 2. for training requires autodiff of the qmat stuff? needs to be tested
### 3. implementation is for 2d, needs to be generalized



class LpOmegaNorm(object):
    """
    Compute Lp norm of a single function by integrating in space
    """
    
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 d:int=2, # space dimension == len(grids), we don't really need this
                 p:int=2  # order of the norm
                 ):
        super().__init__()
        # p has to be >= 1 for the norm to make sense
        assert p >= 1
        # Dimension is postive
        assert d > 0
        assert len(grids) == d
        self.p = p
        self.dx = L[0] / (grids[0].size-1.0)
        self.dy = 1 # in case of 2d this stays at 1 to not influence the result
        if d > 2:
            self.dy = L[1] / (grids[1].size-1.0)
        approx = LagrangeApproximation(grids[-1])
        self.I = approx.getIntegrationMatrix([(0, L[-1])])  # spectral integration along z
        
    def integrate(self,f):
        """
        integrates the grid function abs(f)**p in space
        """
        fp = np.abs(f)**self.p
        # integrate in z with qmat
        intZ = self.I @ fp.T
        # in x and potentially y direction we have equidistant grids with spacing dx, dy        
        return intZ.sum() * self.dx * self.dy   # sum everything (along x- and in 3d y-axis) 
                                                # and scale with grid widths to have an actual discrete Lp norm 
 
    def __call__(self,f):
        return self.integrate(f)**(1/self.p)
    
    
    
class PhysicsLoss(object): # todo: generalize to 3d
    """
    Base class for the physics based losses. 
    Mainly implements calculating the required derivatives.
    """
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  u0,      # initial value of time step
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  ):
         super().__init__()
         # Dimension is postive
         assert d > 0
         assert len(grids) == d
         self.grids = grids
         self.u0 = u0
         self.dt = dt
         self.nx = grids[0].size
         self.Lx = L[0]
         self.dx = self.Lx/(self.nx-1)
         self.approx = LagrangeApproximation(self.grids[-1])
         self.D = self.approx.getDerivationMatrix() # spectral differentiation along z
         self.varChoices = ["vx", "vz", "b", "p"]
        
    def calculateTimeDerivative(self, u, varName):
        fInit = self.u0[self.varChoices.index(varName)].T.copy()
        f = u[self.varChoices.index(varName)].T.copy()
        f_t = (f-fInit)/self.dt
        
        return f_t
         
    def calculateFirstSpatialDerivatives(self, u, varName):
        f = u[self.varChoices.index(varName)].T.copy()

        # spectral differentiation in z -- this works, but can torch autodiff this?
        f_z = self.D@f
        
        # FDM ([nz,nx]) -- this works
        f_x = np.zeros_like(f)
        f = torch.from_numpy(f)  
        # the switching between torch and numpy is stupid; ideally all should be torch tensors
        f_x = (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1))/(2.0*self.dx)
        # fix boundary
        f_x[:,0] = (f[:,1] - f[:,0])/self.dx
        f_x[:,-1] = (f[:,-1] - f[:,-2])/self.dx
        f_x = f_x.numpy()
        
        return f_x, f_z

    def calculateSecondSpatialDerivatives(self, u, varName):
        f = u[self.varChoices.index(varName)].T.copy()

        f_zz = self.D@(self.D@f)
        f_xx = np.zeros_like(f)
        f_xx = ( torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1) ) / self.dx**2
        # fix boundary
        f_xx[:, 0] = (f[:, 2] - 2 * f[:, 1] + f[:, 0]) / self.dx**2
        f_xx[:, -1] = (f[:, -1] - 2 * f[:, -2] + f[:, -3]) / self.dx**2
        f_xx = f_xx.numpy()
        
        return f_xx, f_zz
        
    def calculateDerivatives(self, u, varName): # convenience...
        u_t = self.calculateTimeDerivative(u, varName)
        u_x, u_z = self.calculateFirstSpatialDerivatives(u, varName)
        u_xx, u_zz = self.calculateSecondSpatialDerivatives(u, varName)
        
        return u_t, u_x, u_z, u_xx, u_zz
    


class BuoyancyEquationLoss2D(PhysicsLoss):    # todo: generalize to 3d
    """
    Compute residual for the buoyancy equation
    """
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  u0,      # initial value of time step
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  kappa:float=1.0,
                  ):
         super().__init__(grids, L, u0, dt, d)

         self.kappa = kappa
         self.lpnorm = LpOmegaNorm(grids, L, d, 2)
         
    def computeResidual(self, u):
         bt, bx, bz, bxx, bzz = self.calculateDerivatives(u, "b")

         ## residual
         vx = u[self.varChoices.index("vx")].T.copy()
         vz = u[self.varChoices.index("vz")].T.copy()

         return bt-self.kappa*(bxx+bzz) + vx*bx + vz*bz
     
    def __call__(self, u):
         return self.lpnorm(self.computeResidual(u).T)
         


class BuoyancyUpdateEquationLoss2D(PhysicsLoss):    # todo: generalize to 3d
    """
    Compute equation residual for the update of buoyancy 
    """
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  u0,      # initial value of time step
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  kappa:float=1.0,
                  ):
         super().__init__(grids, L, u0, dt, d)

         self.kappa = kappa
         self.lpnorm = LpOmegaNorm(grids, L, d, 2)
         
    def computeResidual(self, du):
         _, db_x, db_z, db_xx, db_zz = self.calculateDerivatives(du, "b")
         
         # db_t (for the update) is calculated wrong in calculateDerivatives, 
         # so re-calculate it here using that the update at t0 is zero
         db_t = du[self.varChoices.index("b")].T.copy() / self.dt
         
         # additionally we need b_x, b_z
         b_x, b_z  = self.calculateFirstSpatialDerivatives(self.u0, "b")

         ## residual
         dvx = du[self.varChoices.index("vx")].T.copy()
         dvz = du[self.varChoices.index("vz")].T.copy()
         vx = self.u0[self.varChoices.index("vx")].T.copy()
         vz = self.u0[self.varChoices.index("vz")].T.copy()
         

         return db_t-self.kappa*(db_xx+db_zz) + (vx+dvx)*db_x + (vz+dvz)*db_z + dvx*b_x + dvz*b_z
     
    def __call__(self, u):
         return self.lpnorm(self.computeResidual(u).T)




class DivergenceLoss(PhysicsLoss):
    """
    Measures how much the L2-norm of the divergence of a field differs from zero.
    """
    
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 u0,      # initial value of time step
                 dt:float=0.1, # time step size
                 d:int=2, # space dimension == len(grids), we don't really need this
                 varNames:list = ["vx", "vz"]
                 ):
        super().__init__(grids, L, u0, dt, d)
        self.varNames = varNames
        self.lpnorm = LpOmegaNorm(grids, L, d, 2)


    def __call__(self, u):
        vx_x, _ = self.calculateFirstSpatialDerivatives(u, self.varNames[0])
        _, vz_z = self.calculateFirstSpatialDerivatives(u, self.varNames[1])
        return self.lpnorm((vx_x + vz_z).T)
      



class IntegralLoss(PhysicsLoss):
    """
    Measures how much the spatial integral of a function (specified by varName) deviates from a given value
    """
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 u0,      # initial value of time step
                 dt:float=0.1, # time step size
                 d:int=2, # space dimension == len(grids), we don't really need this
                 varName:str = "p",
                 value:float=0.0,
                 ):
         super().__init__(grids, L, u0, dt, d)
         self.dx = L[0] / (grids[0].size-1.0)
         self.dy = 1 # in case of 2d this stays at 1 to not influence the result
         if d > 2:
             self.dy = L[1] / (grids[1].size-1.0)
         approx = LagrangeApproximation(grids[-1])
         self.I = approx.getIntegrationMatrix([(0, L[-1])])  # spectral integration along z       
         self.varName = varName
         self.value = value
            
    def integrate(self, u):
         # cannot directly use the LpOmegaNorm class, as there for p=1 abs(f) is integrated, not f
         f = u[self.varChoices.index(self.varName)].T.copy()
         intZ = self.I @ f
         # in x and potentially y direction we have equidistant grids with spacing dx, dy        
         return intZ.sum() * self.dx * self.dy   # sum everything (along x- and in 3d y-axis) 
                                                 # and scale with grid widths
  
        
    def __call__(self, u): 
         return np.abs(self.integrate(u) - self.value)
     
            