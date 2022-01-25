import numpy as np
import numbers
class Q: 
    def __init__(self, data):
        self.data = np.array(data)
    def __repr__(self):
        return "%s + %s i + %s j + %s k" %(self.data[0], self.data[1], self.data[2], self.data[3])
    
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    
    def __add__(self, other):
        return Q( self.data + other.data )
    
    def __neg__(self):
        return Q( -self.data )
    
    def __sub__(self, other):
        return self + (-other)
    
    def mul(self, other):
        if isinstance(other, Q):
            return Q( (self.data[0]*other.data[0] - self.data[1]*other.data[1] - self.data[2]*other.data[2] - self.data[3]*other.data[3],
                       self.data[1]*other.data[0] + self.data[0]*other.data[1] + self.data[2]*other.data[3] - self.data[3]*other.data[2],
                      self.data[0]*other.data[2] - self.data[1]*other.data[3] + self.data[2]*other.data[0] + self.data[3]*other.data[1],
                      self.data[0]*other.data[3] + self.data[1]*other.data[2] - self.data[2]*other.data[1] + self.data[3]*other.data[0],) )
        elif isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")


class O: 
    def __init__(self, data):
        self.data = np.array(data)
    def __repr__(self):
        return "%s + %s i + %s j + %s k + %s l + %s m + %s n + %s o" %(self.data[0], self.data[1], self.data[2], self.data[3], self.data[4], self.data[5], self.data[6], self.data[7])
    
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    
    def __add__(self, other):
        return O( self.data + other.data )
    
    def __neg__(self):
        return O( -self.data )
    
    def __sub__(self, other):
        return self + (-other)
    def __mul__(self, other):
        if isinstance(other, O):
            a = self.data[0]
            b = self.data[1]
            c = self.data[2]
            d = self.data[3]
            e = self.data[4]
            f = self.data[5]
            g = self.data[6]
            h = self.data[7]
            
            i = other.data[0]
            j = other.data[1]
            k = other.data[2]
            l = other.data[3]
            m = other.data[4]
            n = other.data[5]
            o = other.data[6]
            p = other.data[7]
            return O( (a*i-b*j-c*k-d*l-m*e-n*f-o*g-p*h,
                       a*j+b*i+c*l-d*k-m*f+n*e+o*h-p*g,
                       a*k-b*l+c*i+d*j-m*g-n*h+o*e+p*f,
                       a*l+b*k-c*j+d*i-m*h+n*g-o*f+p*e ,
                       m*a-n*b-o*c-p*d+e*i+f*j+g*k+h*l,
                       m*b+n*a+o*d-p*c-e*j+f*i-g*l+h*k,
                       m*c-n*d+o*a+p*b-e*k+f*l+g*i-h*j,
                       m*d+n*c-o*b+p*a-e*l-f*k+g*j+h*i) )
        elif isinstance(other, numbers.Number):
            return O( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return O( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
