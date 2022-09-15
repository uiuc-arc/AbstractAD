import numpy as np
import torch
from torch.nn.functional import softplus

torch.set_default_dtype(torch.float64)
device = "cpu"

###########################################################################################
# interval domain operations
###########################################################################################

# hadamard (element-wise) multiplication
def imul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    ac, ad, bc, bd = a * c, a * d, b * c, b * d
    real_l = torch.minimum(torch.minimum(ac, ad), torch.minimum(bc, bd))
    real_u = torch.maximum(torch.maximum(ac, ad), torch.maximum(bc, bd))
    return (real_l, real_u)


def imul_scalar(scalar, a: torch.Tensor, b: torch.Tensor):
    if scalar > 0:
        return (scalar * a, scalar * b)
    else:
        return (scalar * b, scalar * a)


# https://hal.inria.fr/inria-00469472/document says its not as simple as just doing ac,ad,bc,bd
# https://www.tuhh.de/ti3/paper/rump/Ru11a.pdf gives a better algorithm (Algorithm 4.5)
def imatmul(a1: torch.Tensor, a2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor):
    mA = (a1 + a2) * 0.5  # using Rump's algorithm 4.8 is much more precise than algo 4.5
    rA = mA - a1
    mB = (b1 + b2) * 0.5
    rB = mB - b1
    sA = torch.sign(mA)
    sB = torch.sign(mB)
    absMA = torch.abs(mA)
    absMB = torch.abs(mB)
    rhoA = sA * torch.minimum(absMA, rA)
    rhoB = sB * torch.minimum(absMB, rB)
    rC = torch.matmul(absMA, rB) + torch.matmul(rA, (absMB + rB)) + torch.matmul(-torch.abs(rhoA), torch.abs(rhoB))
    C2 = torch.matmul(mA, mB) + torch.matmul(rhoA, rhoB) + rC
    C1 = torch.matmul(mA, mB) + torch.matmul(rhoA, rhoB) - rC
    return (C1, C2)


# we don't need full interval matrix multiplication if one of the matrices is not an interval matrix
def imatmul_with_scalar(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor):
    mb = (b1 + b2) * 0.5
    rb = mb - b1
    absa = torch.abs(a)
    rc = torch.matmul(absa, rb)
    l = torch.matmul(a, mb) - rc
    u = torch.matmul(a, mb) + rc
    return (l, u)


# adds the intervals [a,b]+[c,d] - performs this elementwise to each element in the tensors
def iadd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    return (a + c, b + d)


def isub(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    return (a - d, b - c)


# computes hull([a,b],[c,d])=[min(a,c),max(b,d)] - performs this elementwise to each element in the tensors
def ihull(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    return (torch.minimum(a, c), torch.maximum(b, d))


def contains_empty_interval(a: torch.Tensor, b: torch.Tensor):
    return ~torch.all(torch.le(a, b))


# semantically returns intervals of the form 1,-1 for empty intervals
def iintersect(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    l, u = torch.maximum(a, c), torch.minimum(b, d)
    mask = torch.le(l, u)
    neg_mask = (~mask).double()
    l = (l * mask) + (neg_mask)
    u = (u * mask) - (neg_mask)
    return l, u


# elementwise division
def idiv(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    one = torch.ones(a.shape)
    guard = torch.le(c, 0) * torch.ge(d, 0)  # true if the division is undefined (contains zero in the interval)
    dguarded = ((~guard) * d) + (guard * one)
    cguarded = ((~guard) * c) + (guard * one)  # just puts a 1 in any place there the division wouldn't be defined

    minus_inf_mask = (torch.le(c, 0) * torch.ge(d, 0)).type(a.dtype)
    minus_inf_mask[torch.gt(minus_inf_mask, 0)] = torch.tensor(-np.inf)
    plus_inf_mask = (torch.le(c, 0) * torch.ge(d, 0)).type(a.dtype)
    plus_inf_mask[torch.gt(plus_inf_mask, 0)] = torch.tensor(np.inf)

    dinv = torch.div(one, dguarded) + minus_inf_mask
    cinv = torch.div(one, cguarded) + plus_inf_mask

    return imul(a, b, dinv, cinv)


# squares each tensor entry elementwise
def isquare(a: torch.Tensor, b: torch.Tensor):
    asq = a * a
    bsq = b * b
    maxsq = torch.maximum(asq, bsq)
    minsq = torch.minimum(asq, bsq)
    bools = (~(torch.le(a, 0) * torch.ge(b, 0))) * (maxsq)  # bools is 0 if a<0 and b>0
    return (torch.minimum(bools, minsq), maxsq)


def ineg(a: torch.Tensor, b: torch.Tensor):
    return (-b, -a)


def itranspose(a: torch.Tensor, b: torch.Tensor):
    return (torch.transpose(a, 0, 1), torch.transpose(b, 0, 1))


def iexp(a: torch.Tensor, b: torch.Tensor):
    return (torch.exp(a), torch.exp(b))


def ismoothrelu(a: torch.Tensor, b: torch.Tensor, Beta=1):
    l = softplus(a, Beta, threshold=100)
    u = softplus(b, Beta, threshold=100)
    return (l, u)


def isigmoid(a: torch.Tensor, b: torch.Tensor, Beta=1):
    l = torch.sigmoid(a * Beta)
    u = torch.sigmoid(b * Beta)
    return (l, u)


def ifstderivsigmoid(a: torch.Tensor, b: torch.Tensor, Beta=1):
    i1, i2 = isigmoid(a, b, Beta)
    one = torch.ones(a.shape, device=a.device)
    i3, i4 = isub(one, one, i1, i2)
    l, u = imul(i1, i2, i3, i4)
    return (l, u)


def icontained(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    return torch.all(torch.ge(a, c)) * torch.all(torch.le(b, d))


###########################################################################################
# HyperDual Interval Tensors (uses above interval operations as primitives)
###########################################################################################
class HyperDualIntervalTensor:
    def __init__(
        self,
        real_l: torch.Tensor,
        real_u: torch.Tensor,
        e1_l: torch.Tensor = None,
        e1_u: torch.Tensor = None,
        e2_l: torch.Tensor = None,
        e2_u: torch.Tensor = None,
        e1e2_l: torch.Tensor = None,
        e1e2_u: torch.Tensor = None,
    ):
        if e1_l is None:
            e1_l = torch.zeros(real_l.shape)
        if e1_u is None:
            e1_u = torch.zeros(real_l.shape)
        if e2_l is None:
            e2_l = torch.zeros(real_l.shape)
        if e2_u is None:
            e2_u = torch.zeros(real_l.shape)
        if e1e2_l is None:
            e1e2_l = torch.zeros(real_l.shape)
        if e1e2_u is None:
            e1e2_u = torch.zeros(real_l.shape)

        assert (real_l.shape == e1_l.shape) and (e1_l.shape == e2_l.shape) and (real_l.device == e1_l.device)
        assert torch.all(torch.le(real_l, real_u)) and torch.all(torch.le(e1_l, e1_u)) and torch.all(torch.le(e2_l, e2_u)) and torch.all(torch.le(e1e2_l, e1e2_u))

        self.real_l = real_l
        self.real_u = real_u
        self.e1_l = e1_l
        self.e1_u = e1_u
        self.e2_l = e2_l
        self.e2_u = e2_u
        self.e1e2_l = e1e2_l
        self.e1e2_u = e1e2_u
        self.device = real_l.device
        self.dtype = real_l.dtype

    def __repr__(self):
        return f"[{self.real_l}\n{self.real_u}] + \n[{self.e1_l}\n{self.e1_u}]e1 + \n[{self.e2_l}\n{self.e2_u}]e2 + \n[{self.e1e2_l}\n{self.e1e2_u}]e1e2"

    def __getitem__(self, key):
        return HyperDualIntervalTensor(self.real_l[key], self.real_u[key], self.e1_l[key], self.e1_u[key], self.e2_l[key], self.e2_u[key], self.e1e2_l[key], self.e1e2_u[key])

    def __neg__(self):
        return HyperDualIntervalTensor(-self.real_u, -self.real_l, -self.e1_u, -self.e1_l, -self.e2_u, -self.e2_l, -self.e1e2_u, -self.e1e2_l)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            rl = self.real_l + other.real_l
            ru = self.real_u + other.real_u
            e1_l = self.e1_l + other.e1_l
            e1_u = self.e1_u + other.e1_u
            e2_l = self.e2_l + other.e2_l
            e2_u = self.e2_u + other.e2_u
            e1e2_l = self.e1e2_l + other.e1e2_l
            e1e2_u = self.e1e2_u + other.e1e2_u
            return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)
        elif isinstance(other, (int, float, tuple)):
            return HyperDualIntervalTensor(self.real_l + other, self.real_u + other, self.e1_l, self.e1_u, self.e2_l, self.e2_u, self.e1e2_l, self.e1e2_u)
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

            l3, u3 = imul(self.real_l, self.real_u, other.e2_l, other.e2_u)
            l4, u4 = imul(self.e2_l, self.e2_u, other.real_l, other.real_u)
            e2_l, e2_u = iadd(l3, u3, l4, u4)

            t1l, t1u = imul(self.real_l, self.real_u, other.e1e2_l, other.e1e2_u)
            t2l, t2u = imul(self.e1_l, self.e1_u, other.e2_l, other.e2_u)
            t3l, t3u = imul(other.e1_l, other.e1_u, self.e2_l, self.e2_u)
            t4l, t4u = imul(other.real_l, other.real_u, self.e1e2_l, self.e1e2_u)
            t5l, t5u = iadd(t1l, t1u, t2l, t2u)
            t6l, t6u = iadd(t3l, t3u, t4l, t4u)
            e1e2_l, e1e2_u = iadd(t5l, t5u, t6l, t6u)
            
            return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)
        elif isinstance(other, (int, float, torch.Tensor)):
            prl, pru, pe1_l, pe1_u = self.real_l * other, self.real_u * other, self.e1_l * other, self.e1_u * other
            pe2_l, pe2_u, pe1e2_l, pe1e2_u = self.e2_l * other, self.e2_u * other, self.e1e2_l * other, self.e1e2_u * other

            zero = torch.tensor(0.0, device=self.device)
            prl = prl * (torch.le(zero, other))
            pru = pru * (torch.le(zero, other))
            pe1_l = pe1_l * (torch.le(zero, other))
            pe1_u = pe1_u * (torch.le(zero, other))
            pe2_l = pe2_l * (torch.le(zero, other))
            pe2_u = pe2_u * (torch.le(zero, other))
            pe1e2_l = pe1e2_l * (torch.le(zero, other))
            pe1e2_u = pe1e2_u * (torch.le(zero, other))

            nrl, nru, ne1_l, ne1_u = self.real_l * other, self.real_u * other, self.e1_l * other, self.e1_u * other
            ne2_l, ne2_u, ne1e2_l, ne1e2_u = self.e2_l * other, self.e2_u * other, self.e1e2_l * other, self.e1e2_u * other

            nrl = nrl * (torch.gt(zero, other))
            nru = nru * (torch.gt(zero, other))
            ne1_l = ne1_l * (torch.gt(zero, other))
            ne1_u = ne1_u * (torch.gt(zero, other))
            ne2_l = ne2_l * (torch.gt(zero, other))
            ne2_u = ne2_u * (torch.gt(zero, other))
            ne1e2_l = ne1e2_l * (torch.gt(zero, other))
            ne1e2_u = ne1e2_u * (torch.gt(zero, other))

            rl = prl + nrl
            ru = pru + nru
            e1_l = pe1_l + ne1_l
            e1_u = pe1_u + ne1_u
            e2_l = pe2_l + ne2_l
            e2_u = pe2_u + ne2_u
            e1e2_l = pe1e2_l + ne1e2_l
            e1e2_u = pe1e2_u + ne1e2_u
            return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            one = torch.tensor(1.0)
            inv_other_rl, inv_other_ru = idiv(one, one, other.real_l, other.real_u)
            squared_real_l, squared_real_u = isquare(other.real_l, other.real_u)
            cubed_real_l, cubed_real_u = imul(other.real_l, other.real_u, squared_real_l, squared_real_u)

            inv_other_e1_l, inv_other_e1_u = idiv(other.e1_l, other.e1_u, squared_real_l, squared_real_u)
            inv_other_e1_l, inv_other_e1_u = ineg(inv_other_e1_l, inv_other_e1_u)
            inv_other_e2_l, inv_other_e2_u = idiv(other.e2_l, other.e2_u, squared_real_l, squared_real_u)
            inv_other_e2_l, inv_other_e2_u = ineg(inv_other_e2_l, inv_other_e2_u)

            fst_l, fst_u = idiv(other.e1e2_l, other.e1e2_u, squared_real_l, squared_real_u)
            fst_l, fst_u = ineg(fst_l, fst_u)

            snd_l, snd_u = imul(other.e1_l, other.e1_u, other.e2_l, other.e2_u)
            snd_l, snd_u = imul(2.0, 2.0, snd_l, snd_u)
            snd_l, snd_u = idiv(snd_l, snd_u, cubed_real_l, cubed_real_u)

            inv_other_e1e2_l, inv_other_e1e2_u = iadd(fst_l, fst_u, snd_l, snd_u)
            inv = HyperDualIntervalTensor(inv_other_rl, inv_other_ru, inv_other_e1_l, inv_other_e1_u, inv_other_e2_l, inv_other_e2_u, inv_other_e1e2_l, inv_other_e1e2_u)
            
            return self * inv
        elif isinstance(other, (int, float)):  # division by constant is trivial (just scale each term)
            if other < 0:
                ru = torch.div(self.real_l, other)
                rl = torch.div(self.real_u, other)
                e1_u = torch.div(self.e1_l, other)
                e1_l = torch.div(self.e1_u, other)
                e2_u = torch.div(self.e2_l, other)
                e2_l = torch.div(self.e2_u, other)
                e1e2_u = torch.div(self.e1e2_l, other)
                e1e2_l = torch.div(self.e1e2_u, other)
                return HyperDualIntervalTensor(ru, rl, e1_u, e1_l, e2_u, e2_l, e1e2_u, e1e2_l)
            elif other > 0:
                rl = torch.div(self.real_l, other)
                ru = torch.div(self.real_u, other)
                e1_l = torch.div(self.e1_l, other)
                e1_u = torch.div(self.e1_u, other)
                e2_l = torch.div(self.e2_l, other)
                e2_u = torch.div(self.e2_u, other)
                e1e2_l = torch.div(self.e1e2_l, other)
                e1e2_u = torch.div(self.e1e2_u, other)
                return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)
            else:
                raise Exception
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")

    # works for 1xmxn times 1xnxp and batch matrix multiplication
    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            a, b, c, d = self.real_l, self.real_u, other.real_l, other.real_u
            rl, ru = imatmul(a, b, c, d)

            l1, u1 = imatmul(self.real_l, self.real_u, other.e1_l, other.e1_u)
            l2, u2 = imatmul(self.e1_l, self.e1_u, other.real_l, other.real_u)
            e1_l, e1_u = iadd(l1, u1, l2, u2)

            l3, u3 = imatmul(self.real_l, self.real_u, other.e2_l, other.e2_u)
            l4, u4 = imatmul(self.e2_l, self.e2_u, other.real_l, other.real_u)
            e2_l, e2_u = iadd(l3, u3, l4, u4)

            t1l, t1u = imatmul(self.real_l, self.real_u, other.e1e2_l, other.e1e2_u)
            t2l, t2u = imatmul(self.e1_l, self.e1_u, other.e2_l, other.e2_u)
            t3l, t3u = imatmul(self.e2_l, self.e2_u, other.e1_l, other.e1_u)
            t4l, t4u = imatmul(self.e1e2_l, self.e1e2_u, other.real_l, other.real_u)
            t5l, t5u = iadd(t1l, t1u, t2l, t2u)
            t6l, t6u = iadd(t3l, t3u, t4l, t4u)
            e1e2_l, e1e2_u = iadd(t5l, t5u, t6l, t6u)

            return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)
        else:
            raise TypeError(f"Unsupported operand type(s) for @: '{self.__class__}' and '{type(other)}'")

    def clear_e1(self):
        self.e1_l = self.e1_l - self.e1_l
        self.e1_u = self.e1_u - self.e1_u

    def clear_e2(self):
        self.e2_l = self.e2_l - self.e2_l
        self.e2_u = self.e2_u - self.e2_u

    def clear_e1e2(self):
        self.e1e2_l = self.e1e2_l - self.e1e2_l
        self.e1e2_u = self.e1e2_u - self.e1e2_u

    def get_real(self):
        return self.real_l, self.real_u

    def get_e1(self):
        return self.e1_l, self.e1_u

    def get_e2(self):
        return self.e2_l, self.e2_u

    def get_e1e2(self):
        return self.e1e2_l, self.e1e2_u

    def matmul(self, other):
        return self @ other

    def flatten(self, start_dim=0, end_dim=-1):
        rl = self.real_l.flatten(start_dim, end_dim)
        ru = self.real_u.flatten(start_dim, end_dim)
        e1_l = self.e1_l.flatten(start_dim, end_dim)
        e1_u = self.e1_l.flatten(start_dim, end_dim)
        e2_l = self.e2_l.flatten(start_dim, end_dim)
        e2_u = self.e2_u.flatten(start_dim, end_dim)
        e1e2_l = self.e1e2_l.flatten(start_dim, end_dim)
        e1e2_u = self.e1e2_u.flatten(start_dim, end_dim)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def unsqueeze(self, dim):
        rl = torch.unsqueeze(self.real_l, dim)
        ru = torch.unsqueeze(self.real_u, dim)
        e1_l = torch.unsqueeze(self.e1_l, dim)
        e1_u = torch.unsqueeze(self.e1_u, dim)
        e2_l = torch.unsqueeze(self.e2_l, dim)
        e2_u = torch.unsqueeze(self.e2_u, dim)
        e1e2_l = torch.unsqueeze(self.e1e2_l, dim)
        e1e2_u = torch.unsqueeze(self.e1e2_u, dim)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def unfold(self, dimension, size, step):
        rl = self.real_l.unfold(dimension, size, step)
        ru = self.real_u.unfold(dimension, size, step)
        e1_l = self.e1_l.unfold(dimension, size, step)
        e1_u = self.e1_u.unfold(dimension, size, step)
        e2_l = self.e2_l.unfold(dimension, size, step)
        e2_u = self.e2_u.unfold(dimension, size, step)
        e1e2_l = self.e1e2_l.unfold(dimension, size, step)
        e1e2_u = self.e1e2_u.unfold(dimension, size, step)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def repeat(self, *dim):
        rl = self.real_l.repeat(dim)
        ru = self.real_u.repeat(dim)
        e1_l = self.e1_l.repeat(dim)
        e1_u = self.e1_u.repeat(dim)
        e2_l = self.e2_l.repeat(dim)
        e2_u = self.e2_u.repeat(dim)
        e1e2_l = self.e1e2_l.repeat(dim)
        e1e2_u = self.e1e2_u.repeat(dim)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def size(self, dim=None):
        return self.real_l.size(dim)

    def shape(self):
        return self.real_l.shape

    def transpose(self, dim0, dim1):
        rl = self.real_l.transpose(dim0, dim1)
        ru = self.real_u.transpose(dim0, dim1)
        e1_l = self.e1_l.transpose(dim0, dim1)
        e1_u = self.e1_u.transpose(dim0, dim1)
        e2_l = self.e2_l.transpose(dim0, dim1)
        e2_u = self.e2_u.transpose(dim0, dim1)
        e1e2_l = self.e1e2_l.transpose(dim0, dim1)
        e1e2_u = self.e1e2_u.transpose(dim0, dim1)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def t(self):
        return self.transpose(0, 1)

    def view(self, *shape):
        rl = self.real_l.view(shape)
        ru = self.real_u.view(shape)
        e1_l = self.e1_l.view(shape)
        e1_u = self.e1_u.view(shape)
        e2_l = self.e2_l.view(shape)
        e2_u = self.e2_u.view(shape)
        e1e2_l = self.e1e2_l.view(shape)
        e1e2_u = self.e1e2_u.view(shape)
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def clone(self):
        rl = self.real_l.detach().clone()
        ru = self.real_u.detach().clone()
        e1_l = self.e1_l.detach().clone()
        e1_u = self.e1_u.detach().clone()
        e2_l = self.e2_l.detach().clone()
        e2_u = self.e2_u.detach().clone()
        e1e2_l = self.e1e2_l.detach().clone()
        e1e2_u = self.e1e2_u.detach().clone()
        return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)

    def debug_scalarize(self):
        self.real_l = self.real_u
        self.e1_l = self.e1_u
        self.e2_l = self.e2_u
        self.e1e2_l = self.e1e2_u


def e1e2(hdi: HyperDualIntervalTensor, deriv1_l: torch.Tensor, deriv1_u: torch.Tensor, deriv2_l: torch.Tensor, deriv2_u: torch.Tensor):
    i1, i2 = imul(hdi.e1e2_l, hdi.e1e2_u, deriv1_l, deriv1_u)
    i3, i4 = imul(hdi.e1_l, hdi.e1_u, hdi.e2_l, hdi.e2_u)
    i5, i6 = imul(i3, i4, deriv2_l, deriv2_u)
    return iadd(i1, i2, i5, i6)


def SmoothRelu_hdi(hdi: HyperDualIntervalTensor, Beta=1):
    zero = torch.zeros(hdi.real_l.shape, device=hdi.device, dtype=hdi.dtype)
    one = torch.ones(hdi.real_l.shape, device=hdi.device, dtype=hdi.dtype)
    pi_by_4 = torch.tensor(np.pi / 4.0, device=hdi.device, dtype=hdi.dtype) * one
    rl, ru = ismoothrelu(hdi.real_l, hdi.real_u, Beta)

    d1_l, d1_u = isigmoid(hdi.real_l, hdi.real_u, Beta)
    d1_l, d1_u = iintersect(d1_l, d1_u, zero, one)
    e1_l, e1_u = imul(d1_l, d1_u, hdi.e1_l, hdi.e1_u)
    e2_l, e2_u = imul(d1_l, d1_u, hdi.e2_l, hdi.e2_u)

    # snd deriv of smoothrelu is fst deriv of sigmoid
    d2_l, d2_u = ifstderivsigmoid(hdi.real_l, hdi.real_u, Beta)
    d2_l, d2_u = iintersect(d2_l, d2_u, zero, pi_by_4)
    e1e2_l, e1e2_u = e1e2(hdi, d1_l, d1_u, d2_l, d2_u)

    return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)


def Exp_hdi(hdi: HyperDualIntervalTensor):
    rl = torch.exp(hdi.real_l)
    ru = torch.exp(hdi.real_u)

    # rl and ru are also the values of the 1st derivative
    e1_l, e1_u = imul(rl, ru, hdi.e1_l, hdi.e1_u)
    e2_l, e2_u = imul(rl, ru, hdi.e2_l, hdi.e2_u)

    # rl and ru are also the values of the 2nd derivative
    e1e2_l, e1e2_u = e1e2(hdi, rl, ru, rl, ru)

    return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)


def Log_hdi(hdi: HyperDualIntervalTensor):
    one = torch.ones(hdi.real_l.shape, device=hdi.device, dtype=hdi.dtype)
    rl = torch.log(hdi.real_l)
    ru = torch.log(hdi.real_u)

    d1_l, d1_u = idiv(one, one, hdi.real_l, hdi.real_u)
    e1_l, e1_u = imul(d1_l, d1_u, hdi.e1_l, hdi.e1_u)
    e2_l, e2_u = imul(d1_l, d1_u, hdi.e2_l, hdi.e2_u)

    i1, i2 = isquare(hdi.real_l, hdi.real_u)
    i3, i4 = idiv(one, one, i1, i2)
    d2_l, d2_u = ineg(i3, i4)
    e1e2_l, e1e2_u = e1e2(hdi, d1_l, d1_u, d2_l, d2_u)

    return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)


def Pow_hdi(a: HyperDualIntervalTensor, hdi: HyperDualIntervalTensor):
    return Exp_hdi(Log_hdi(hdi) * a)


def Tanh_hdi(hdi: HyperDualIntervalTensor):
    one = torch.ones(hdi.real_l.shape, device=hdi.device, dtype=hdi.dtype)
    rl = torch.tanh(hdi.real_l)
    ru = torch.tanh(hdi.real_u)

    i1, i2 = isquare(rl, ru)
    d1_l, d1_u = isub(one, one, i1, i2)
    e1_l, e1_u = imul(d1_l, d1_u, hdi.e1_l, hdi.e1_u)
    e2_l, e2_u = imul(d1_l, d1_u, hdi.e2_l, hdi.e2_u)

    i3, i4 = imul(rl, ru, d1_l, d1_u)
    d2_l, d2_u = imul_scalar(-2, i3, i4)
    e1e2_l, e1e2_u = e1e2(hdi, d1_l, d1_u, d2_l, d2_u)

    return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)


# abstracts a scalar tensor to a HyperDualIntervalTensor
def abstract_hdi(t: torch.Tensor, q=0):
    rl = t - q
    ru = t + q
    e1_l = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e1_u = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e2_l = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e2_u = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e1e2_l = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    e1e2_u = torch.zeros(t.shape, device=t.device, dtype=t.dtype)
    return HyperDualIntervalTensor(rl, ru, e1_l, e1_u, e2_l, e2_u, e1e2_l, e1e2_u)


def Affine_hdi(HDI, layer):
    assert type(layer) in [torch.tensor, torch.Tensor, torch.FloatTensor]
    return HDI @ abstract_hdi(layer)
