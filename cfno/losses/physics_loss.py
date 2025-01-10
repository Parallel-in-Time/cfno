import torch
from qmat.lagrange import LagrangeApproximation

### TODOs:
### 1. training requires autodiff of the qmat stuff? needs to be tested -> seems to be fine
### 2. implementation is for 2d, needs to be generalized to handle 3d as well

# Store all loss classes in a dictionary using the register decorator 
LOSSES_CLASSES = {}

def register(cls):
    assert hasattr(cls, '__call__') and callable(cls), "loss class must implement the __call__ method"
    LOSSES_CLASSES[cls.__name__] = cls
    return cls


@register
class LpOmegaNorm(object):
    """
    Compute Lp norm of a minibatch of grid functions by integrating in space
    """
    
    def __init__(self,
                 grids,      # tuple of grids (x,y,z) or (x,z) depending on d
                 L,          # tuple of domain sizes fitting to grids
                 d:int=2,    # space dimension == len(grids), we don't really need this
                 p:int=2,    # order of the norm
                 device=None # device where the tensors live (to move the integration matrix to the correct device)
                 ):
        super().__init__()
        # p has to be >= 1 for the norm to make sense
        assert p >= 1
        # Dimension is positive
        assert d > 0
        assert len(grids) == d
        self.p = p
        self.dx = L[0] / (grids[0].size-1.0)
        self.dy = 1 # in case of 2d this stays at 1 to not influence the result
        if d > 2:
            self.dy = L[1] / (grids[1].size-1.0)
        approx = LagrangeApproximation(grids[-1])
        self.I = approx.getIntegrationMatrix([(0, L[-1])])  # spectral integration along z
        self.I = torch.from_numpy(self.I).type(torch.float).to(device)
        self.device=device
        
    def integrate(self,f):
        """
        integrates the grid functions abs(f)**p in space
        shape of f is [nbatch, nx, ny, nz] in 3d. [nbatch, nx, nz] in 2d
        """
        fp = torch.abs(f)**self.p

        
        intZ = torch.einsum('ij,bkj->bki', self.I, fp)
        # in x and potentially y direction we have equidistant grids with spacing dx, dy        
        return torch.sum(intZ, dim=-2) * self.dx * self.dy   # sum everything (along x- and in 3d y-axis) 
                                                # and scale with grid widths


    def __call__(self,f):
        return torch.mean(self.integrate(f)**(1/self.p))




class PhysicsLoss(object): # todo: generalize to 3d
    """
    Base class for the physics based losses. 
    Mainly implements calculating the required derivatives.
    """
    # u0 and u have shape [nbatches, 4, nx, nz]
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  device=None # device where the tensors live (to move the differentiation matrix to the correct device)
                  ):
         super().__init__()
         # Dimension is positive
         assert d > 0
         assert len(grids) == d
         self.grids = grids
         self.dt = dt
         self.nx = grids[0].size
         self.Lx = L[0]
         self.dx = self.Lx/(self.nx-1)
         self.approx = LagrangeApproximation(self.grids[-1])
         self.D = self.approx.getDerivationMatrix() # spectral differentiation along z
         self.D = torch.from_numpy(self.D).type(torch.float).to(device)
         self.varChoices = ["vx", "vz", "b", "p"]
         self.device=device
        
    def calculateTimeDerivative(self, u, u0, varName):        
        return ((u0[:,self.varChoices.index(varName)] - u[:,self.varChoices.index(varName)]) / self.dt).to(self.device)
        
         
    def calculateFirstSpatialDerivatives(self, u, varName):
        f = u[:,self.varChoices.index(varName)].to(self.device)
        
        # spectral differentiation in z -- this works, but can torch autodiff this?
        f_z = torch.einsum('ij,bkj->bki', self.D, f)
        
        # FDM ([nbatch,nx, nz]) -- this works, but is rather inaccurate
        f_x = torch.zeros_like(f)
        f_x = (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2))/(2.0*self.dx)

        # fix boundary
        f_x[...,0,:] = (f[...,1,:] - f[...,0,:])/self.dx    
        f_x[...,-1,:] = (f[...,-1,:] - f[...,-2,:])/self.dx
        
        return f_x, f_z

    def calculateSecondSpatialDerivatives(self, u, varName):
        f = u[:,self.varChoices.index(varName)].to(self.device)

        f_z = torch.einsum('ij,bkj->bki', self.D, f)
        f_zz = torch.einsum('ij,bkj->bki', self.D, f_z)

        f_xx = torch.zeros_like(f)
        f_xx = ( torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2) ) / self.dx**2
        # fix boundary
        f_xx[..., 0,:] = (f[..., 2,:] - 2 * f[..., 1,:] + f[..., 0,:]) / self.dx**2
        f_xx[..., -1,:] = (f[..., -1,:] - 2 * f[..., -2,:] + f[..., -3,:]) / self.dx**2
        
        return f_xx, f_zz
        
    def calculateDerivatives(self, u, u0, varName): # convenience...
        u_t = self.calculateTimeDerivative(u, u0, varName)
        u_x, u_z = self.calculateFirstSpatialDerivatives(u, varName)
        u_xx, u_zz = self.calculateSecondSpatialDerivatives(u, varName)
        
        return u_t, u_x, u_z, u_xx, u_zz
    

@register
class BuoyancyEquationLoss2D(PhysicsLoss):    # todo: generalize to 3d
    """
    Compute residual for the buoyancy equation
    """
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  kappa:float=1.0,                      
                  device=None
                  ):
         super().__init__(grids, L, dt, d, device)

         self.kappa = kappa
         self.lpnorm = LpOmegaNorm(grids, L, d, 2, device)
         
    def computeResidual(self, u, u0):
         bt, bx, bz, bxx, bzz = self.calculateDerivatives(u, u0, "b")

         ## residual
         vx = u[:,self.varChoices.index("vx")].to(self.device)
         vz = u[:,self.varChoices.index("vz")].to(self.device)
         
         return bt-self.kappa*(bxx+bzz) + vx*bx + vz*bz
     
    def __call__(self, pred, ref, inp):
         return torch.mean(self.lpnorm(self.computeResidual(pred, inp)))
         

@register
class BuoyancyUpdateEquationLoss2D(PhysicsLoss):    # todo: generalize to 3d
    """
    Compute equation residual for the update of buoyancy 
    """
    def __init__(self,
                  grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                  L,       # tuple of domain sizes fitting to grids
                  dt:float=0.1, # time step size
                  d:int=2, # space dimension == len(grids), we don't really need this
                  kappa:float=1.0,
                  device=None
                  ):
         super().__init__(grids, L, dt, d, device)

         self.kappa = kappa
         self.lpnorm = LpOmegaNorm(grids, L, d, 2)
         
    def computeResidual(self, du, u0):
         db_x, db_z = self.calculateFirstSpatialDerivatives(du, "b")
         db_xx, db_zz = self.calculateSecondSpatialDerivatives(du, "b")
         
         # db_t (for the update) is calculated wrong in calculateDerivatives, 
         # so re-calculate it here using that the update at t0 is zero
         db_t = du[:,self.varChoices.index("b")] / self.dt
         
         # additionally we need b_x, b_z
         b_x, b_z  = self.calculateFirstSpatialDerivatives(u0, "b")

         ## residual
         dvx = du[:,self.varChoices.index("vx")]
         dvz = du[:,self.varChoices.index("vz")]
         vx = u0[:,self.varChoices.index("vx")]
         vz = u0[:,self.varChoices.index("vz")]
         

         return db_t-self.kappa*(db_xx+db_zz) + (vx+dvx)*db_x + (vz+dvz)*db_z + dvx*b_x + dvz*b_z
     
    def __call__(self, pred, ref, inp):
         return torch.mean(self.lpnorm(self.computeResidual(pred, inp)))


@register
class DivergenceLoss(PhysicsLoss):
    """
    Measures how much the L2-norm of the divergence of a field differs from zero.
    """
    
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 dt:float=0.1, # time step size
                 d:int=2, # space dimension == len(grids), we don't really need this
                 varNames:list = ["vx", "vz"],
                 device=None
                 ):
        super().__init__(grids, L, dt, d, device)
        self.varNames = varNames
        self.lpnorm = LpOmegaNorm(grids, L, d, 2)


    def __call__(self, pred, ref=None, inp=None):
        vx_x, _ = self.calculateFirstSpatialDerivatives(pred, self.varNames[0])
        _, vz_z = self.calculateFirstSpatialDerivatives(pred, self.varNames[1])
        return torch.mean(self.lpnorm((vx_x + vz_z)))
      

@register
class IntegralLoss(PhysicsLoss):
    """
    Measures how much the spatial integral of a function (specified by varName) deviates from a given value
    """
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 dt:float=0.1, # time step size
                 d:int=2, # space dimension == len(grids), we don't really need this
                 varName:str="p",
                 value:float=0.0,
                 device=None
                 ):
         super().__init__(grids, L, dt, d, device)
         self.dx = L[0] / (grids[0].size-1.0)
         self.dy = 1 # in case of 2d this stays at 1 to not influence the result
         if d > 2:
             self.dy = L[1] / (grids[1].size-1.0)
         approx = LagrangeApproximation(grids[-1])
         self.I = approx.getIntegrationMatrix([(0, L[-1])])  # spectral integration along z    
         self.I = torch.from_numpy(self.I).type(torch.float).to(device)
         self.varName = varName
         self.value = value
            
    def integrate(self, u):
         # cannot directly use the LpOmegaNorm class, as there for p=1 abs(f) is integrated, not f
         f = u[:,self.varChoices.index(self.varName)]
         
         intZ = torch.einsum('ij,bkj->bki', self.I, f)
         # in x and potentially y direction we have equidistant grids with spacing dx, dy        
         return torch.sum(intZ, dim=-2) * self.dx * self.dy   # sum everything (along x- and in 3d y-axis) 
                                                 # and scale with grid widths

    def __call__(self, pred, ref, inp): 
         return torch.mean(torch.abs(self.integrate(pred) - self.value))


@register
class RBEqLoss(object):
    """
    Combines all the above defined physics losses (weighted) into one total loss for the Rayleigh Benard Equations
    """
    def __init__(self,
                 grids,   # tuple of grids (x,y,z) or (x,z) depending on d
                 L,       # tuple of domain sizes fitting to grids
                 dt:float=0.1, # time step size
                 d:int=2, # space dimension == len(grids), we don't really need this
                 kappa:float=1.0,
                 device=None,
                 weights=[1, 1]
                 ):
        self.weights=weights
        self.device=device
        self.losses = [BuoyancyEquationLoss2D(grids, L, dt, d, kappa, device), IntegralLoss(grids,L,dt,d,device=device)]
        
    def __call__(self, pred, ref, inp):
        sum = 0.0
        for i, loss in enumerate(self.losses):
            sum = sum + self.weights[i] * loss(pred, ref, inp)
        return sum
    
    
    
