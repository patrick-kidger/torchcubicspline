import math
import torch

from . import misc


def _validate_input(t, X):
    if not t.is_floating_point():
        raise ValueError("t must both be floating point.")
    if not X.is_floating_point():
        raise ValueError("X must both be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")

    if X.ndimension() < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels. It instead has "
                         "shape {}.".format(tuple(X.shape)))

    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t. X has shape {} and t has shape {}, "
                         "corresponding to time dimensions of {} and {} respectively."
                         .format(tuple(X.shape), tuple(t.shape), X.size(-2), t.size(0)))

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2. It instead has shape {}, corresponding to a "
                         "time dimension of size {}.".format(tuple(t.shape), t.size(0)))


def _natural_cubic_spline_coeffs_without_missing_values(t, x):
    # x should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = x.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[..., 1:] - t[..., :1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    else:
        # Set up some intermediate values
        time_diffs = t[1:] - t[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=x.dtype, device=x.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = misc.tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
                                                  time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


def _natural_cubic_spline_coeffs_with_missing_values(t, x):
    if x.ndimension() == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x)
    else:
        a_pieces = []
        b_pieces = []
        two_c_pieces = []
        three_d_pieces = []
        for p in x.unbind(dim=0):  # TODO: parallelise over this
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p)
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (misc.cheap_stack(a_pieces, dim=0),
                misc.cheap_stack(b_pieces, dim=0),
                misc.cheap_stack(two_c_pieces, dim=0),
                misc.cheap_stack(three_d_pieces, dim=0))


def _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x):
    # t and x both have shape (length,)

    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        # Note that we may assume that X.size(0) >= 2 by the checks in __init__ so "X.size(0) - 1" is a valid
        # thing to do.
        return (torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device))
    # else we have at least one non-NaN entry, in which case we're going to impute at least one more entry (as
    # the path is of length at least 2 so the start and the end aren't the same), so we will then have at least two
    # non-Nan entries. In particular we can call _compute_coeffs safely later.

    # How to deal with missing values at the start or end of the time series? We're creating some splines, so one
    # option is just to extend the first piece backwards, and the final piece forwards. But polynomials tend to
    # behave badly when extended beyond the interval they were constructed on, so the results can easily end up
    # being awful.
    # Instead we impute an observation at the very start equal to the first actual observation made, and impute an
    # observation at the very end equal to the last actual observation made, and then proceed with splines as
    # normal.
    need_new_not_nan = False
    if torch.isnan(x[0]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[0] = path_no_nan[0]
    if torch.isnan(x[-1]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[-1] = path_no_nan[-1]
    if need_new_not_nan:
        not_nan = ~torch.isnan(x)
        path_no_nan = x.masked_select(not_nan)
    times_no_nan = t.masked_select(not_nan)

    # Find the coefficients on the pieces we do understand
    # These all have shape (len - 1,)
    (a_pieces_no_nan,
     b_pieces_no_nan,
     two_c_pieces_no_nan,
     three_d_pieces_no_nan) = _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)

    # Now we're going to normalise them to give coefficients on every interval
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []

    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan))
    next_time_no_nan = next(iter_times_no_nan)
    for time in t[:-1]:
        # will always trigger on the first iteration because of how we've imputed missing values at the start and
        # end of the time series.
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (misc.cheap_stack(a_pieces, dim=0),
            misc.cheap_stack(b_pieces, dim=0),
            misc.cheap_stack(two_c_pieces, dim=0),
            misc.cheap_stack(three_d_pieces, dim=0))


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def natural_cubic_spline_coeffs(t, x):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tuple of five tensors, which should in turn be passed to `torchcubicspline.NaturalCubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `torchcubicspline.NaturalCubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call NaturalCubicSpline as the first part of your model.

        See also the accompanying example.py.
    """
    _validate_input(t, x)

    if torch.isnan(x).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, x.transpose(-1, -2))
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, x.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    # The code so far has created twice the c value and three times the d value because it was written with a preference
    # for computing the derivative of the natural cubic spline, which need those values instead. I'm not going to try
    # and change that for this standalone torchcubicspline project, and instead this is a simple fix.
    c = two_c.transpose(-1, -2) / 2
    d = three_d.transpose(-1, -2) / 3
    return t, a, b, c, d


class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        t = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(t, x)

        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...

        spline = NaturalCubicSpline(coeffs)

        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)

        point = torch.tensor([0.4, 0.5])
        # will be a tensor of shape (2, 1, 2, 3), corresponding to batch, time and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, **kwargs):
        """
        Arguments:
            coeffs: As returned by `torchcubicspline.natural_cubic_spline_coeffs`.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        t, a, b, c, d = coeffs

        self._t = t
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        index = torch.bucketize(t.detach(), self._t) - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 2 * self._c[..., index, :] + 3 * self._d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv
