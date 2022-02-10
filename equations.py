
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, domain, u):
        #dtype=self.dtype
        u = spectral.Field(domain)
        dudx = spectral.Field(domain)
        ududx = spectral.Field(domain)
        N = x_basis.N
        kx = x_basis.wavenumbers()

    def evolve(self, timestepper, dt, num_steps):
        u.require_coeff_space()
        dudx.require_coeff_space()
        dudx.data = 1j*kx*u.data
        u.require_grid_space()
        dudx.require_grid_space()
        ududx.require_grid_space()
        ududx.data = 6 * u.data * dudx.data
        ududx.require_coeff_space()

        diag = 1/dt + kx**3
        LHS = sparse.diags(diag)
        u.require_coeff_space()
        RHS.data += u.data/dt
        u.data = spla.spsolve(LHS, RHS.data)

class SHEquation:

    def __init__(self, domain, u):
        pass

    def evolve(self, timestepper, dt, num_steps):
        pass


