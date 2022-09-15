import itertools
import numpy as np
import torch
from torch.nn.functional import softplus


def copy_centers(centers, num):
    cols = np.prod(list(centers.shape))
    return centers.expand((num, cols))


# converts row vectors like [a,b,c] to [[a,0,0],[0,b,0],[0,0,c]]
def traceify(rowvec):
    leng = torch.numel(rowvec)
    copied = copy_centers(rowvec, leng)
    identity = torch.eye(leng)
    return identity * copied


def IntervalsToZonotope(lower, upper):
    assert lower.shape == upper.shape
    assert torch.all(lower <= upper)
    midpoint = (upper + lower) / 2.0
    radii = (upper - lower) / 2.0
    generators = traceify(radii)
    return Zonotope(midpoint, generators)


def ZonotopeToInterval(zonotope):
    return (zonotope.get_lb(), zonotope.get_ub())


class Zonotope:
    def __init__(self, centers, coefs):
        assert np.prod(list(centers.shape)) == coefs.shape[1]
        self.centers = centers   # will be a 1 x M (where M is number of vars) row vector
        self.generators = coefs  # will be a N x M (where N is number of noise erms) matrix

    def get_num_vars(self):
        return np.prod(list(self.centers.shape))

    def get_num_noise_symbs(self):
        return self.generators.shape[0]

    def get_coeff_abs(self):
        return torch.sum(torch.abs(self.generators), dim=0)  # sum along rows

    def get_lb(self):
        cof_abs = self.get_coeff_abs()
        lb = self.centers - cof_abs
        return lb

    def get_ub(self):
        cof_abs = self.get_coeff_abs()
        ub = self.centers + cof_abs
        return ub

    def expand(self, n):
        if n == 0:
            return
        if n > 0:
            cols = self.generators.shape[1]
            self.generators = torch.cat([self.generators, torch.zeros((n, cols))], dim=0)
        else:
            raise Exception

    def __add__(self, other):
        if type(other) in [float, int]:
            centers = self.centers + other
            return Zonotope(centers, self.generators)
        elif type(other) is Zonotope:
            # make sure same dimensions
            self_noise_syms = self.get_num_noise_symbs()
            other_noise_syms = other.get_num_noise_symbs()

            if self_noise_syms > other_noise_syms:
                other.expand(self_noise_syms - other_noise_syms)
            elif self_noise_syms < other_noise_syms:
                self.expand(other_noise_syms - self_noise_syms)

            centers = self.centers + other.centers
            generators = self.generators + other.generators
            return Zonotope(centers, generators)
        elif type(other) in [torch.tensor, torch.Tensor, torch.FloatTensor]:  # instead of adding the same constant to all centers, adds a different constant to each center
            if other.shape == self.centers.shape:
                centers = self.centers + other
                return Zonotope(centers, self.generators)
            else:
                raise Exception
        else:
            raise Exception

    __radd__ = __add__

    def __str__(self):
        return "Centers: \n" + self.centers.__str__() + "\n" + "Generators: \n" + self.generators.__str__() + "\n"

    def __neg__(self):
        return Zonotope(-self.centers, self.generators)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if type(other) in [float, int]:  # scalar multiplication can be done exactly
            centers = other * self.centers
            generators = other * self.generators
            return Zonotope(centers, generators)
        elif type(other) is Zonotope:
            # make sure same dimensions
            self_noise_syms = self.get_num_noise_symbs()
            other_noise_syms = other.get_num_noise_symbs()

            if self_noise_syms > other_noise_syms:
                other.expand(self_noise_syms - other_noise_syms)
            elif self_noise_syms < other_noise_syms:
                self.expand(other_noise_syms - self_noise_syms)

            centers = self.centers * other.centers
            a0_copy = copy_centers(self.centers, self.get_num_noise_symbs())
            b0_copy = copy_centers(other.centers, other.get_num_noise_symbs())

            a0bi = a0_copy * other.generators
            b0ai = b0_copy * self.generators
            before_adding_new_terms = a0bi + b0ai

            abs_ai = self.get_coeff_abs()
            abs_bi = other.get_coeff_abs()
            new_noise_magnitudes = abs_ai * abs_bi

            # need to convert new noise magnitudes from [a,b,c] -> [[a,0,0],[0,b,0],[0,0,c]]
            traceified = traceify(new_noise_magnitudes)
            generators = torch.cat([before_adding_new_terms, traceified], dim=0)
            return Zonotope(centers, generators)
        else:
            raise Exception

    __rmul__ = __mul__

    def get_stacked(self):
        return torch.cat([self.centers.unsqueeze(0), self.generators], dim=0)


# for affine layers, no need to add new noise symbols
def AffineZonotope(Zono, layer, bias=None):
    centers = Zono.centers
    generators = Zono.generators
    new_centers = centers @ layer
    if bias is not None:
        new_centers = new_centers + bias
    new_generators = generators @ layer
    return Zonotope(new_centers, new_generators)


def TanhZonotope(Zono):
    lb = Zono.get_lb()
    ub = Zono.get_ub()

    centers = Zono.centers
    generators = Zono.generators

    if torch.all(lb == ub):
        centers = torch.tanh(lb)
        generators = torch.zeros(centers.shape)
        return Zonotope(centers, generators)

    lambda_opt = torch.min(-torch.square(torch.tanh(lb)) + 1, -torch.square(torch.tanh(ub)) + 1)
    mu1 = 0.5 * (torch.tanh(ub) + torch.tanh(lb) - lambda_opt * (ub + lb))
    mu2 = 0.5 * (torch.tanh(ub) - torch.tanh(lb) - lambda_opt * (ub - lb))
    new_center = (lambda_opt * centers) + mu1
    new_generators_before = lambda_opt * generators
    traceified_mu2 = traceify(mu2)
    new_generators = torch.cat((new_generators_before, traceified_mu2), 0)

    return Zonotope(new_center, new_generators)


def SoftPlus(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return softplus(x, beta, threshold=100)


def Sigmoid(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return torch.sigmoid(x, beta)


# inverse of a sigmoid function
def InverseSigmoid(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return torch.div((torch.log(x) - torch.log(1 - x)), beta)


# uses this form: Fast reliable interrogation of procedurally defined implicit surfaces using extended revised affine arithmetic
# https://www.researchgate.net/publication/220251211_Fast_reliable_interrogation_of_procedurally_defined_implicit_surfaces_using_extended_revised_affine_arithmetic
def SoftPlusZonoChebyshev(Zono, beta=1):
    centers = Zono.centers
    generators = Zono.generators

    a = Zono.get_lb()
    b = Zono.get_ub()
    fa = SoftPlus(a, beta)
    fb = SoftPlus(b, beta)
    alpha = (fb - fa) / (b - a)  # lambda_opt in DeepZ
    alpha[alpha == 1] -= torch.finfo(torch.float64).eps  # resolves numerical underflow
    intercept = fa - (alpha * a)
    u = InverseSigmoid(alpha, beta)
    r = lambda x: alpha * x + intercept
    zeta = 0.5 * (SoftPlus(u, beta) + r(u)) - (alpha * u)  # mu1 in DeepZ
    delta = 0.5 * abs(SoftPlus(u, beta) - r(u))  # mu2 in DeepZ

    new_center = (alpha * centers) + zeta
    new_generators_before = alpha * generators

    traceified_delta = traceify(delta)
    new_generators = torch.cat((new_generators_before, traceified_delta), 0)
    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)
    return res


def ExpZonoChebyshev(Zono):
    centers = Zono.centers
    generators = Zono.generators

    a = Zono.get_lb()
    b = Zono.get_ub()
    fa = torch.exp(a)
    fb = torch.exp(b)
    alpha = (fb - fa) / (b - a)  # lambda_opt in DeepZ
    intercept = fa - (alpha * a)
    u = torch.log(alpha)
    r = lambda x: alpha * x + intercept
    zeta = 0.5 * (torch.exp(u) + r(u)) - (alpha * u)  # mu1 in DeepZ
    delta = 0.5 * abs(torch.exp(u) - r(u))  # mu2 in DeepZ

    new_center = (alpha * centers) + zeta
    new_generators_before = alpha * generators

    traceified_delta = traceify(delta)
    new_generators = torch.cat((new_generators_before, traceified_delta), 0)
    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)
    return res


def SigmoidZonotope(Zono):
    lb = Zono.get_lb()
    ub = Zono.get_ub()

    centers = Zono.centers
    generators = Zono.generators

    if torch.all(lb == ub):
        centers = torch.sigmoid(lb)
        generators = torch.zeros(centers.shape)
        return Zonotope(centers, generators)

    lambda_opt = torch.min(torch.sigmoid(lb) * (1 - torch.sigmoid(lb)), torch.sigmoid(ub) * (1 - torch.sigmoid(ub)))
    mu1 = 0.5 * (torch.sigmoid(ub) + torch.sigmoid(lb) - lambda_opt * (ub + lb))
    mu2 = 0.5 * (torch.sigmoid(ub) - torch.sigmoid(lb) - lambda_opt * (ub - lb))
    new_center = (lambda_opt * centers) + mu1
    new_generators_before = lambda_opt * generators
    traceified_mu2 = traceify(mu2)
    new_generators = torch.cat((new_generators_before, traceified_mu2), 0)

    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)
    return res


def HyperDualIntervalToDualZonotope(hdi):
    real = IntervalsToZonotope(hdi.real_l.flatten(), hdi.real_u.flatten())
    dual = IntervalsToZonotope(hdi.e1_l.flatten(), hdi.e1_u.flatten())
    dual.generators = torch.cat([torch.zeros((real.generators.shape[0], real.generators.shape[1])), dual.generators])

    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = real.get_num_noise_symbs()
    real.expand(dual_num_noise_terms - real_num_noise_terms)
    return filter_dual_zono(DualZonotope(real.centers, real.generators, dual.centers, dual.generators))


class DualZonotope:
    def __init__(self, real_centers, real_coefs, dual_centers, dual_coefs):
        assert real_coefs.shape == dual_coefs.shape
        self.real = Zonotope(real_centers, real_coefs)
        self.dual = Zonotope(dual_centers, dual_coefs)

    def __str__(self):
        return "Real: \n" + self.real.__str__() + "\nDual: \n" + self.dual.__str__() + "\n"

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return DualZonotope(self.real.centers + other, self.real.generators, self.dual.centers, self.dual.generators)
        elif isinstance(other, self.__class__):
            r = self.real + other.real
            d = self.dual + other.dual
            return DualZonotope(r.centers, r.generators, d.centers, d.generators)
        else:
            raise Exception

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return DualZonotope(-self.real.centers, self.real.generators, -self.dual.centers, self.dual.generators)

    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            return DualZonotope(self.real.centers * other, self.real.generators * other, self.dual.centers * other, self.dual.generators * other)
        else:
            raise Exception


# assumes the layer and bias are torch tensors
def AffineDualZonotope(DZ, layer, bias=None):
    real = AffineZonotope(DZ.real, layer)
    if bias is not None:
        real = real + bias
    dual = AffineZonotope(DZ.dual, layer)
    return DualZonotope(real.centers, real.generators, dual.centers, dual.generators)


def SmoothReluDualZonotope(DualZono):
    smoothrelu_real = SoftPlusZonoChebyshev(DualZono.real)
    smoothrelu_deriv = SigmoidZonotope(DualZono.real)
    dual = smoothrelu_deriv * DualZono.dual

    # go back and expand real part's number of noise symbols to match
    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = smoothrelu_real.get_num_noise_symbs()
    smoothrelu_real.expand(dual_num_noise_terms - real_num_noise_terms)

    return DualZonotope(smoothrelu_real.centers, smoothrelu_real.generators, dual.centers, dual.generators)


def TanhDualZonotope(DualZono):
    tanh_real = TanhZonotope(DualZono.real)
    tanh_deriv = -(tanh_real * tanh_real) + 1
    dual = tanh_deriv * DualZono.dual

    # go back and expand real part's number of noise symbols to match
    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = tanh_real.get_num_noise_symbs()
    tanh_real.expand(dual_num_noise_terms - real_num_noise_terms)

    return DualZonotope(tanh_real.centers, tanh_real.generators, dual.centers, dual.generators)


def format_str(st):
    l = list(st)
    return torch.tensor([float(-1) if x == "0" else float(x) for x in l])


def add_leading_one(mat):
    rows = mat.shape[0]
    new = torch.zeros((rows + 1, rows + 1))
    new[0, 0] = 1.0
    new[1:, 1:] = mat
    return new


# because the norm is a convex function and the feasible set is closed and convex, the maximum will be acheived at the boundaries
# that is to say when the noise symbols are either +/- 1
def get_identities(n):
    # generate all bitstrings
    all_bin_str = ["".join(seq) for seq in itertools.product("01", repeat=n)]
    formatted = [format_str(x) for x in all_bin_str]
    traceified = [traceify(x) for x in formatted]
    final_tensor = torch.stack([add_leading_one(x) for x in traceified])
    return final_tensor


# checks if a certain row is always zero for all matrices in the list
def check_zero_rows(lst_of_mats):
    S = torch.sum(torch.sum(torch.abs(torch.stack(lst_of_mats)), dim=0), dim=1)
    res = S != 0
    return res


# https://discuss.pytorch.org/t/filter-out-undesired-rows/28933
def filter_zeros(a, ind):
    return a[ind, :]


def filter_lst(lst_of_mats):
    inds = check_zero_rows(lst_of_mats)
    return [filter_zeros(x, inds) for x in lst_of_mats]


def filter_dual_zono(dz):
    lst = [dz.real.generators, dz.dual.generators]
    inds = check_zero_rows(lst)
    real_generators = filter_zeros(dz.real.generators, inds)
    dual_generators = filter_zeros(dz.dual.generators, inds)
    return DualZonotope(dz.real.centers, real_generators, dz.dual.centers, dual_generators)
