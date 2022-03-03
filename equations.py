
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


class SoundWaves:

    def __init__(self, domain, u, p, p0):
        dtype = self.dtype = u.dtype
        self.domain = domain
        self.u = u
        self.p = p
        self.px = spectral.Field(domain, dtype=dtype)
        self.ux = spectral.Field(domain, dtype=dtype)
        self.p0 = p0
        self.RHS = spectral.Field(domain, dtype=dtype)
        self.RHS0 = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u, p], [self.RHS0,self.RHS], num_BCs=2, dtype = dtype)
        x_basis = domain.bases[0]
        self.N = N = x_basis.N


        diag = np.arange(N-1) + 1
        self.D = D = sparse.diags(diag, offsets=1) * 2/3

        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0, 2))

        Z = sparse.csr_matrix((N, N))
        pen = self.problem.pencils[0]
        n = pen.wavenumbers
            
        L = sparse.bmat([[ Z,    D],
                             [ D,    Z]])
        
        M = sparse.csr_matrix((2*N+2, 2*N+2))
        M[:N,:N] = C
        M[N:2*N, N:2*N] = C
  
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :1*N] = (-1)**i
        BC_rows[1, :1*N] = (+1)**i
     
        cols = np.zeros((2*N, 2))
        cols[  N-1, 0] = 1 
        cols[2*N-1, 1] = 1
            
        corner = np.zeros((2,2))
        pen.M = M
        pen.L = sparse.bmat([[L, cols],
                           [BC_rows, corner]])
        pen.L.eliminate_zeros()
        pen.M.eliminate_zeros()
        self.t = 0

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        ux = self.ux
        p0 = self.p0
        RHS = self.RHS
        p0.require_coeff_space()
        p0.require_grid_space(scales=3/2)

        for i in range(num_steps):
            # take a timestep
            u.require_coeff_space()
            ux.require_coeff_space()
            ux.data = spla.spsolve(self.C ,self.D @ u.data)            
            RHS.require_coeff_space()
            ux.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = (1 - p0.data) * ux.data 
            RHS.require_coeff_space()
            RHS.data = self.C @ RHS.data
        
            ts.step(dt, [0,0])
            self.t += dt

class CGLEquation:

    def __init__(self, domain, u):
        dtype = self.dtype = u.dtype
        self.domain = domain
        self.u = u
        self.ux = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.duxdx = spectral.Field(domain, dtype=dtype)
        
        self.RHS = spectral.Field(domain, dtype=dtype)
        self.RHS0 = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u, ux], [self.RHS0,self.RHS], num_BCs=2, dtype = dtype)
        x_basis = domain.bases[0]
        self.N = N = x_basis.N


        diag = np.arange(N-1) + 1
        self.D = D = sparse.diags(diag, offsets=1) * 2/3

        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0, 2))

        Z = sparse.csr_matrix((N, N))
        pen = self.problem.pencils[0]
        n = pen.wavenumbers
            
        L = sparse.bmat([[ Z,    Z],
                             [ D,    -C]])
        
        M = sparse.csr_matrix((2*N+2, 2*N+2))
        M[:N,:N] = C
  
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :1*N] = (-1)**i
        BC_rows[1, :1*N] = (+1)**i
     
        cols = np.zeros((2*N, 2))
        cols[  N-1, 0] = 1 
        cols[2*N-1, 1] = 1
            
        corner = np.zeros((2,2))
        pen.M = M
        pen.L = sparse.bmat([[L, cols],
                           [BC_rows, corner]])
        pen.L.eliminate_zeros()
        pen.M.eliminate_zeros()
        self.t = 0

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
              
        u = self.u
        ux = self.ux
        dudx = self.dudx
        duxdx = self.duxdx
        RHS = self.RHS
 
        b = 0.5
        c = -1.76
        #for i in range(num_steps):
            u.require_coeff_space()
            dudx.require_coeff_space()
            dudx.data = spla.spsolve(self.C ,self.D @ u.data)            
            RHS.require_coeff_space()
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = u + (1+1j*b)*duxdx.data - (1+1j*c)*np.abs(u.data)**2 * u.data 
            RHS.require_coeff_space()
            RHS.data = self.C @ RHS.data
        
            ts.step(dt, [0,0])
            self.t += dt


class BurgersEquation:
    
    def __init__(self, domain, u, nu):
        dtype = u.dtype
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = -nu*D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space()
            dudx.require_grid_space()
            u_RHS.require_grid_space()
            u_RHS.data = -u.data*dudx.data
            ts.step(dt)


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)



