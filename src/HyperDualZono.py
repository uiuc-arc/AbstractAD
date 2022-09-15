from SimpleZono import *


class HyperDualZonotope:
    def __init__(self, real_centers, real_coefs, e1_centers, e1_coefs, e2_centers, e2_coefs, e1e2_centers, e1e2_coefs):
        assert real_coefs.shape == e1_coefs.shape
        assert e1e2_coefs.shape == e2_coefs.shape
        self.real = Zonotope(real_centers, real_coefs)
        self.e1 = Zonotope(e1_centers, e1_coefs)
        self.e2 = Zonotope(e2_centers, e2_coefs)
        self.e1e2 = Zonotope(e1e2_centers, e1e2_coefs)

    def __str__(self):
        return "Real: \n" + self.real.__str__() + "\ne1: \n" + self.e1.__str__() + "\ne2: \n" + self.e2.__str__() + "\ne1e2: \n" + self.e1e2.__str__() + "\n"

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return HyperDualZonotope(
                self.real.centers + other, self.real.generators, self.e1.centers, self.e1.generators, self.e2.centers, self.e2.generators, self.e1e2.centers, self.e1e2.generators
            )


def filter_hyperdual_zono(hdz):
    lst = [hdz.real.generators, hdz.e1.generators, hdz.e2.generators, hdz.e1e2.generators]
    inds = check_zero_rows(lst)
    real_generators = filter_zeros(hdz.real.generators, inds)
    e1_generators = filter_zeros(hdz.e1.generators, inds)
    e2_generators = filter_zeros(hdz.e2.generators, inds)
    e1e2_generators = filter_zeros(hdz.e1e2.generators, inds)
    return HyperDualZonotope(hdz.real.centers, real_generators, hdz.e1.centers, e1_generators, hdz.e2.centers, e2_generators, hdz.e1e2.centers, e1e2_generators)


def HyperDualIntervalToHyperDualZonotope(hdi):
    real = IntervalsToZonotope(hdi.real_l.flatten(), hdi.real_u.flatten())
    e1 = IntervalsToZonotope(hdi.e1_l.flatten(), hdi.e1_u.flatten())
    e1.generators = torch.cat([torch.zeros((real.generators.shape[0], real.generators.shape[1])), e1.generators])
    e2 = IntervalsToZonotope(hdi.e2_l.flatten(), hdi.e2_u.flatten())
    e2.generators = torch.cat([torch.zeros((e1.generators.shape[0], e1.generators.shape[1])), e2.generators])
    e1e2 = IntervalsToZonotope(hdi.e1e2_l.flatten(), hdi.e1e2_u.flatten())
    e1e2.generators = torch.cat([torch.zeros((e2.generators.shape[0], e2.generators.shape[1])), e1e2.generators])

    e1e2_num_noise_terms = e1e2.get_num_noise_symbs()
    real.expand(e1e2_num_noise_terms - real.get_num_noise_symbs())
    e1.expand(e1e2_num_noise_terms - e1.get_num_noise_symbs())
    e2.expand(e1e2_num_noise_terms - e2.get_num_noise_symbs())

    return filter_hyperdual_zono(HyperDualZonotope(real.centers, real.generators, e1.centers, e1.generators, e2.centers, e2.generators, e1e2.centers, e1e2.generators))


# assumes the layer and bias are torch tensors
def AffineHyperDualZonotope(HDZ, layer, bias=None):
    HDZ = filter_hyperdual_zono(HDZ)
    real = AffineZonotope(HDZ.real, layer)
    if bias is not None:
        real = real + bias
    e1 = AffineZonotope(HDZ.e1, layer)
    e2 = AffineZonotope(HDZ.e2, layer)
    e1e2 = AffineZonotope(HDZ.e1e2, layer)
    return HyperDualZonotope(real.centers, real.generators, e1.centers, e1.generators, e2.centers, e2.generators, e1e2.centers, e1e2.generators)


def SmoothReluHyperDualZonotope(HyperDualZono):
    HyperDualZono = filter_hyperdual_zono(HyperDualZono)
    smoothrelu_real = SoftPlusZonoChebyshev(HyperDualZono.real)
    smoothrelu_deriv = SigmoidZonotope(HyperDualZono.real)

    e1 = smoothrelu_deriv * HyperDualZono.e1
    e1_num_noise_terms = e1.get_num_noise_symbs()

    # go back and expand real part's number of noise symbols to match
    smoothrelu_real.expand(e1_num_noise_terms - smoothrelu_real.get_num_noise_symbs())
    HyperDualZono.real.expand(e1_num_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(e1_num_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    smoothrelu_deriv.expand(e1_num_noise_terms - smoothrelu_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(e1_num_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(e1_num_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    e2 = smoothrelu_deriv * HyperDualZono.e2
    e2_num_noise_terms = e2.get_num_noise_symbs()

    smoothrelu_real.expand(e2_num_noise_terms - smoothrelu_real.get_num_noise_symbs())
    HyperDualZono.real.expand(e2_num_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(e2_num_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    smoothrelu_deriv.expand(e2_num_noise_terms - smoothrelu_deriv.get_num_noise_symbs())
    e1.expand(e2_num_noise_terms - e1.get_num_noise_symbs())
    HyperDualZono.e2.expand(e2_num_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(e2_num_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    snd_deriv = smoothrelu_deriv * (-smoothrelu_deriv + 1)
    snd_deriv_noise_terms = snd_deriv.get_num_noise_symbs()

    smoothrelu_real.expand(snd_deriv_noise_terms - smoothrelu_real.get_num_noise_symbs())
    HyperDualZono.real.expand(snd_deriv_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(snd_deriv_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(snd_deriv_noise_terms - e1.get_num_noise_symbs())
    smoothrelu_deriv.expand(snd_deriv_noise_terms - smoothrelu_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(snd_deriv_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(snd_deriv_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(snd_deriv_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    term1 = (snd_deriv * HyperDualZono.e1) * HyperDualZono.e2  #
    term1_noise_terms = term1.get_num_noise_symbs()

    smoothrelu_real.expand(term1_noise_terms - smoothrelu_real.get_num_noise_symbs())
    HyperDualZono.real.expand(term1_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(term1_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(term1_noise_terms - e1.get_num_noise_symbs())
    smoothrelu_deriv.expand(term1_noise_terms - smoothrelu_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(term1_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(term1_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(term1_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    term2 = smoothrelu_deriv * HyperDualZono.e1e2
    term2_noise_terms = term2.get_num_noise_symbs()

    smoothrelu_real.expand(term2_noise_terms - smoothrelu_real.get_num_noise_symbs())
    HyperDualZono.real.expand(term2_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(term2_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(term2_noise_terms - e1.get_num_noise_symbs())
    smoothrelu_deriv.expand(term2_noise_terms - smoothrelu_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(term2_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(term2_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(term2_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    e1e2 = term1 + term2
    return HyperDualZonotope(smoothrelu_real.centers, smoothrelu_real.generators, e1.centers, e1.generators, e2.centers, e2.generators, e1e2.centers, e1e2.generators)


def ExpHyperDualZonotope(HyperDualZono):
    exp_real = ExpZonoChebyshev(HyperDualZono.real)
    exp_deriv = exp_real

    e1 = exp_deriv * HyperDualZono.e1
    e1_num_noise_terms = e1.get_num_noise_symbs()

    exp_real.expand(e1_num_noise_terms - exp_real.get_num_noise_symbs())
    HyperDualZono.real.expand(e1_num_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(e1_num_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    exp_deriv.expand(e1_num_noise_terms - exp_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(e1_num_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(e1_num_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    e2 = exp_deriv * HyperDualZono.e2
    e2_num_noise_terms = e2.get_num_noise_symbs()

    exp_real.expand(e2_num_noise_terms - exp_real.get_num_noise_symbs())
    HyperDualZono.real.expand(e2_num_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(e2_num_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    exp_deriv.expand(e2_num_noise_terms - exp_deriv.get_num_noise_symbs())
    e1.expand(e2_num_noise_terms - e1.get_num_noise_symbs())
    HyperDualZono.e2.expand(e2_num_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(e2_num_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    snd_deriv = exp_deriv
    snd_deriv_noise_terms = snd_deriv.get_num_noise_symbs()

    exp_real.expand(snd_deriv_noise_terms - exp_real.get_num_noise_symbs())
    HyperDualZono.real.expand(snd_deriv_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(snd_deriv_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(snd_deriv_noise_terms - e1.get_num_noise_symbs())
    exp_deriv.expand(snd_deriv_noise_terms - exp_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(snd_deriv_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(snd_deriv_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(snd_deriv_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    term1 = (snd_deriv * HyperDualZono.e1) * HyperDualZono.e2
    term1_noise_terms = term1.get_num_noise_symbs()

    exp_real.expand(term1_noise_terms - exp_real.get_num_noise_symbs())
    HyperDualZono.real.expand(term1_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(term1_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(term1_noise_terms - e1.get_num_noise_symbs())
    exp_deriv.expand(term1_noise_terms - exp_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(term1_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(term1_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(term1_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    term2 = exp_deriv * HyperDualZono.e1e2
    term2_noise_terms = term2.get_num_noise_symbs()

    exp_real.expand(term2_noise_terms - exp_real.get_num_noise_symbs())
    HyperDualZono.real.expand(term2_noise_terms - HyperDualZono.real.get_num_noise_symbs())
    HyperDualZono.e1.expand(term2_noise_terms - HyperDualZono.e1.get_num_noise_symbs())
    e1.expand(term2_noise_terms - e1.get_num_noise_symbs())
    exp_deriv.expand(term2_noise_terms - exp_deriv.get_num_noise_symbs())
    HyperDualZono.e2.expand(term2_noise_terms - HyperDualZono.e2.get_num_noise_symbs())
    e2.expand(term2_noise_terms - e2.get_num_noise_symbs())
    HyperDualZono.e1e2.expand(term2_noise_terms - HyperDualZono.e1e2.get_num_noise_symbs())

    e1e2 = term1 + term2
    return HyperDualZonotope(exp_real.centers, exp_real.generators, e1.centers, e1.generators, e2.centers, e2.generators, e1e2.centers, e1e2.generators)
