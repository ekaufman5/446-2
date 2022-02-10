
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, domain, u):
        self.dtype=u.dtype
        print(self.dtype)
        self.u = u
        self.dudx = spectral.Field(domain, dtype=u.dtype)
        self.ududx = spectral.Field(domain, dtype=u.dtype)
        N = len(u.data)
        print(N)
        x_basis = x_basis = spectral.Fourier(N)        
        self.kx = x_basis.wavenumbers(u.dtype)        

    def evolve(self, timestepper, dt, num_steps):
        u = self.u
        dudx = self.dudx
        ududx = self.ududx        
        kx = self.kx
        for i in range(num_steps):        
            u.require_coeff_space()
            dudx.require_coeff_space()
            dudx.data = 1j*kx*u.data
            u.require_grid_space()
            dudx.require_grid_space()
            ududx.require_grid_space()
            ududx.data = 6 * u.data * dudx.data
            #ududx.require_coeff_space()
    
            #diag = 1/dt + kx**3
            #LHS = sparse.diags(diag)
            #u.require_coeff_space()
            #RHS.data += u.data/dt
            timestepper.step(dt)            
            #u.data = spla.spsolve(LHS, RHS.data)


class SHEquation:

    def __init__(self, domain, u):
        pass

    def evolve(self, timestepper, dt, num_steps):
        pass


