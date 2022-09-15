from HyperDuals import *
import torch

###########################################################################################
# Dual Interval Tensors (uses interval operations in HyperDuals as primitives)
###########################################################################################
class DualIntervalTensor:
    def __init__(self, real_l: torch.Tensor, real_u: torch.Tensor, e1_l: torch.Tensor = None, e1_u: torch.Tensor = None):
        if e1_l is None:
            e1_l = torch.zeros(real_l.shape)
        if e1_u is None:
            e1_u = torch.zeros(real_l.shape)

        assert (real_l.shape == e1_l.shape) and (real_l.device == e1_l.device)
        assert torch.all(torch.le(real_l, real_u)) and torch.all(torch.le(e1_l, e1_u))

        self.real_l = real_l
        self.real_u = real_u
        self.e1_l = e1_l
        self.e1_u = e1_u
        self.device = real_l.device
        self.dtype = real_l.dtype

    def __repr__(self):
        return f"[{self.real_l}\n{self.real_u}] + \n[{self.e1_l}\n{self.e1_u}]e1"

    def __getitem__(self, key):
        return DualIntervalTensor(self.real_l[key], self.real_u[key], self.e1_l[key], self.e1_u[key])

    def __neg__(self):
        return DualIntervalTensor(-self.real_u, -self.real_l, -self.e1_u, -self.e1_l)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            rl = self.real_l + other.real_l
            ru = self.real_u + other.real_u
            e1_l = self.e1_l + other.e1_l
            e1_u = self.e1_u + other.e1_u
            return DualIntervalTensor(rl, ru, e1_l, e1_u)
        elif isinstance(other, (int, float, tuple)):
            return DualIntervalTensor(self.real_l + other, self.real_u + other, self.e1_l, self.e1_u)
        else:
            raise TypeError(f"Unsupported operand type(s) for +/-: '{self.__class__}' and '{type(other)}'")

    __radd__ = __add__

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    # hadamard product between two hyper-dual interval tensors
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            # [a, b] * [c, d]
            a, b, c, d = self.real_l, self.real_u, other.real_l, other.real_u
            rl, ru = imul(a, b, c, d)
            l1, u1 = imul(self.real_l, self.real_u, other.e1_l, other.e1_u)
            l2, u2 = imul(self.e1_l, self.e1_u, other.real_l, other.real_u)
            e1_l, e1_u = iadd(l1, u1, l2, u2)
            return DualIntervalTensor(rl, ru, e1_l, e1_u)
        elif isinstance(other, (int, float, torch.Tensor)):
            zero = torch.tensor(0.0, device=self.device)

            prl, pru, pe1_l, pe1_u = self.real_l * other, self.real_u * other, self.e1_l * other, self.e1_u * other
            prl = prl * (torch.le(zero, other))
            pru = pru * (torch.le(zero, other))
            pe1_l = pe1_l * (torch.le(zero, other))
            pe1_u = pe1_u * (torch.le(zero, other))

            nrl, nru, ne1_l, ne1_u = self.real_l * other, self.real_u * other, self.e1_l * other, self.e1_u * other
            nrl = nrl * (torch.gt(zero, other))
            nru = nru * (torch.gt(zero, other))
            ne1_l = ne1_l * (torch.gt(zero, other))
            ne1_u = ne1_u * (torch.gt(zero, other))

            rl = prl + nrl
            ru = pru + nru
            e1_l = pe1_l + ne1_l
            e1_u = pe1_u + ne1_u
            return DualIntervalTensor(rl, ru, e1_l, e1_u)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            one = torch.tensor(1.0)
            inv_other_rl, inv_other_ru = idiv(one, one, other.real_l, other.real_u)
            squared_real_l, squared_real_u = isquare(other.real_l, other.real_u)
            inv_other_e1_l, inv_other_e1_u = idiv(other.e1_l, other.e1_u, squared_real_l, squared_real_u)
            inv_other_e1_l, inv_other_e1_u = ineg(inv_other_e1_l, inv_other_e1_u)
            inv = DualIntervalTensor(inv_other_rl, inv_other_ru, inv_other_e1_l, inv_other_e1_u)
            return self * inv
        elif isinstance(other, (int, float)):  # division by constant is trivial (just scale each term)
            if other < 0:
                ru = torch.div(self.real_l, other)
                rl = torch.div(self.real_u, other)
                e1_u = torch.div(self.e1_l, other)
                e1_l = torch.div(self.e1_u, other)
                return DualIntervalTensor(ru, rl, e1_u, e1_l)
            elif other > 0:
                rl = torch.div(self.real_l, other)
                ru = torch.div(self.real_u, other)
                e1_l = torch.div(self.e1_l, other)
                e1_u = torch.div(self.e1_u, other)
                return DualIntervalTensor(rl, ru, e1_l, e1_u)
            else:
                raise Exception
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")

    # works for 1xmxn times 1xnxp and batch matrix multiplication!
    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            a, b, c, d = self.real_l, self.real_u, other.real_l, other.real_u
            rl, ru = imatmul(a, b, c, d)
            l1, u1 = imatmul(self.real_l, self.real_u, other.e1_l, other.e1_u)
            l2, u2 = imatmul(self.e1_l, self.e1_u, other.real_l, other.real_u)
            e1_l, e1_u = iadd(l1, u1, l2, u2)
            return DualIntervalTensor(rl, ru, e1_l, e1_u)
        else:
            raise TypeError(f"Unsupported operand type(s) for @: '{self.__class__}' and '{type(other)}'")

    def clear_e1(self):
        self.e1_l = self.e1_l - self.e1_l
        self.e1_u = self.e1_u - self.e1_u

    def get_real(self):
        return self.real_l, self.real_u

    def get_e1(self):
        return self.e1_l, self.e1_u

    def matmul(self, other):
        return self @ other

    def flatten(self, start_dim=0, end_dim=-1):
        rl = self.real_l.flatten(start_dim, end_dim)
        ru = self.real_u.flatten(start_dim, end_dim)
        e1_l = self.e1_l.flatten(start_dim, end_dim)
        e1_u = self.e1_l.flatten(start_dim, end_dim)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def unsqueeze(self, dim):
        rl = torch.unsqueeze(self.real_l, dim)
        ru = torch.unsqueeze(self.real_u, dim)
        e1_l = torch.unsqueeze(self.e1_l, dim)
        e1_u = torch.unsqueeze(self.e1_u, dim)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def unfold(self, dimension, size, step):
        rl = self.real_l.unfold(dimension, size, step)
        ru = self.real_u.unfold(dimension, size, step)
        e1_l = self.e1_l.unfold(dimension, size, step)
        e1_u = self.e1_u.unfold(dimension, size, step)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def repeat(self, *dim):
        rl = self.real_l.repeat(dim)
        ru = self.real_u.repeat(dim)
        e1_l = self.e1_l.repeat(dim)
        e1_u = self.e1_u.repeat(dim)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def size(self, dim=None):
        return self.real_l.size(dim)

    def shape(self):
        return self.real_l.shape

    def transpose(self, dim0, dim1):
        rl = self.real_l.transpose(dim0, dim1)
        ru = self.real_u.transpose(dim0, dim1)
        e1_l = self.e1_l.transpose(dim0, dim1)
        e1_u = self.e1_u.transpose(dim0, dim1)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def t(self):
        return self.transpose(0, 1)

    def view(self, *shape):
        rl = self.real_l.view(shape)
        ru = self.real_u.view(shape)
        e1_l = self.e1_l.view(shape)
        e1_u = self.e1_u.view(shape)
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def clone(self):
        rl = self.real_l.detach().clone()
        ru = self.real_u.detach().clone()
        e1_l = self.e1_l.detach().clone()
        e1_u = self.e1_u.detach().clone()
        return DualIntervalTensor(rl, ru, e1_l, e1_u)

    def debug_scalarize(self):
        self.real_l = self.real_u
        self.e1_l = self.e1_u


def SmoothRelu_di(di: DualIntervalTensor, Beta=1):
    zero = torch.zeros(di.real_l.shape, device=di.device, dtype=di.dtype)
    one = torch.ones(di.real_l.shape, device=di.device, dtype=di.dtype)
    rl, ru = ismoothrelu(di.real_l, di.real_u, Beta)
    d1_l, d1_u = isigmoid(di.real_l, di.real_u, Beta)
    d1_l, d1_u = iintersect(d1_l, d1_u, zero, one)
    e1_l, e1_u = imul(d1_l, d1_u, di.e1_l, di.e1_u)
    return DualIntervalTensor(rl, ru, e1_l, e1_u)


def Exp_di(di: DualIntervalTensor):
    rl = torch.exp(di.real_l)
    ru = torch.exp(di.real_u)
    # rl and ru are also the values of the 1st derivative
    e1_l, e1_u = imul(rl, ru, di.e1_l, di.e1_u)
    return DualIntervalTensor(rl, ru, e1_l, e1_u)


def Log_di(di: DualIntervalTensor):
    one = torch.ones(di.real_l.shape, device=di.device, dtype=di.dtype)
    rl = torch.log(di.real_l)
    ru = torch.log(di.real_u)
    d1_l, d1_u = idiv(one, one, di.real_l, di.real_u)
    e1_l, e1_u = imul(d1_l, d1_u, di.e1_l, di.e1_u)
    return DualIntervalTensor(rl, ru, e1_l, e1_u)


def Pow_di(a: DualIntervalTensor, di: DualIntervalTensor):
    return Exp_di(Log_di(di) * a)


def Tanh_di(di: DualIntervalTensor):
    one = torch.ones(di.real_l.shape, device=di.device, dtype=di.dtype)
    rl = torch.tanh(di.real_l)
    ru = torch.tanh(di.real_u)
    i1, i2 = isquare(rl, ru)
    d1_l, d1_u = isub(one, one, i1, i2)
    e1_l, e1_u = imul(d1_l, d1_u, di.e1_l, di.e1_u)
    return DualIntervalTensor(rl, ru, e1_l, e1_u)


# abstracts a scalar tensor to a DualIntervalTensor
def abstract_di(t: torch.Tensor, q=0):
    rl = t - q
    ru = t + q
    e1_l = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e1_u = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    return DualIntervalTensor(rl, ru, e1_l, e1_u)


def Affine_di(DI, layer):
    assert type(layer) in [torch.tensor, torch.Tensor, torch.FloatTensor]
    return DI @ abstract_di(layer)
