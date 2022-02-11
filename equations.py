
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, domain, u):
        self.dtype=u.dtype
        N = len(u.data)         
        self.u = u
        self.domain=domain
        self.dudx = spectral.Field(domain, dtype=u.dtype)
        self.ududx = spectral.Field(domain, dtype=u.dtype)
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.ududx], dtype=u.dtype)
        x_basis = spectral.Fourier(N)        
        self.kx = x_basis.wavenumbers(u.dtype)
        p = self.problem.pencils[0]
        p.M = sparse.eye(x_basis.N)  
        if self.dtype == np.complex128:
            p.L = sparse.diags(-1j*self.kx**3)
        else: 
            updiag = np.zeros(x_basis.N-1)
            updiag[::2] = -self.kx[::2]
            lowdiag = -updiag
            self.D = sparse.diags([updiag, lowdiag], offsets=(1,-1))
            p.L = -self.D.multiply(self.D.multiply(self.D))       

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)        
        u = self.u
        dudx = self.dudx
     
        ududx = self.ududx        
        kx = self.kx
        for i in range(num_steps):        
            u.require_coeff_space()
            dudx.require_coeff_space()
            if self.dtype == np.complex128:
                dudx.data = 1j*kx*u.data
            else:
                dudx.data = self.D @ u.data 
            print(dudx.dtype)            
            u.require_grid_space(scales=3/2)

            dudx.require_grid_space(scales=3/2)
            ududx.require_grid_space(scales=3/2)
            ududx.data = 6 * u.data * dudx.data
            #ududx.require_coeff_space()
    
            #diag = 1/dt + kx**3
            #LHS = sparse.diags(diag)
            #u.require_coeff_space()
            #RHS.data += u.data/dt
            ts.step(dt)            
            #u.data = spla.spsolve(LHS, RHS.data)


class SHEquation:

    def __init__(self, domain, u):
        pass

    def evolve(self, timestepper, dt, num_steps):
        pass


