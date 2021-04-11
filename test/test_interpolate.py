import torch
import torchcubicspline


# Represents a random natural cubic spline with a single knot in the middle
class _Cubic:
    def __init__(self, batch_dims, num_channels, start, end):
        self.a = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.b = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.c = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.d1 = -self.c / (3 * start)
        self.d2 = -self.c / (3 * end)

    def _normalise_dims(self, t):
        a = self.a
        b = self.b
        c = self.c
        d1 = self.d1
        d2 = self.d2
        for _ in t.shape:
            a = a.unsqueeze(-2)
            b = b.unsqueeze(-2)
            c = c.unsqueeze(-2)
            d1 = d1.unsqueeze(-2)
            d2 = d2.unsqueeze(-2)
        t = t.unsqueeze(-1)
        d = torch.where(t > 0, d2, d1)
        return a, b, c, d, t

    def evaluate(self, t):
        a, b, c, d, t = self._normalise_dims(t)
        t_sq = t ** 2
        t_cu = t_sq * t
        return a + b * t + c * t_sq + d * t_cu

    def derivative(self, t, order=1):
        a, b, c, d, t = self._normalise_dims(t)
        t_sq = t ** 2
        if order == 1:
            return b + 2 * c * t + 3 * d * t_sq
        elif order == 2:
            return 2 * c + 6 * d * t


def test_interp():
    for _ in range(10):
        for drop in (False, True):
            num_points = torch.randint(low=5, high=100, size=(1,)).item()
            times1 = torch.rand(num_points // 2, dtype=torch.float64) - 1
            times2 = torch.rand(num_points // 2, dtype=torch.float64)
            times = torch.cat([times1, times2, torch.tensor([0.], dtype=torch.float64)]).sort().values
            num_channels = torch.randint(low=1, high=3, size=(1,)).item()
            num_batch_dims = torch.randint(low=0, high=3, size=(1,)).item()
            batch_dims = []
            for _ in range(num_batch_dims):
                batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
            cubic = _Cubic(batch_dims, num_channels, start=times[0], end=times[-1])
            values = cubic.evaluate(times)
            if drop:
                for values_slice in values.unbind(dim=-1):
                    num_drop = int(num_points * torch.randint(low=1, high=4, size=(1,)).item() / 10)
                    num_drop = min(num_drop, num_points - 4)
                    to_drop = torch.randperm(num_points - 2)[:num_drop] + 1  # don't drop first or last
                    values_slice[..., to_drop] = float('nan')
            coeffs = torchcubicspline.natural_cubic_spline_coeffs(times, values)
            spline = torchcubicspline.NaturalCubicSpline(coeffs)
            _test_equal(batch_dims, num_channels, cubic, spline)

# TODO: test edge cases


def _test_equal(batch_dims, num_channels, obj1, obj2):
    for dimension in (0, 1, 2):
        sizes = []
        for _ in range(dimension):
            sizes.append(torch.randint(low=1, high=4, size=(1,)).item())
        expected_size = tuple(batch_dims) + tuple(sizes) + (num_channels,)
        eval_times = torch.rand(sizes, dtype=torch.float64) * 3 - 1.5
        obj1_evaluate = obj1.evaluate(eval_times)
        obj2_evaluate = obj2.evaluate(eval_times)
        obj1_derivative = obj1.derivative(eval_times, order=1)
        obj2_derivative = obj2.derivative(eval_times, order=1)
        obj1_second_derivative = obj1.derivative(eval_times, order=2)
        obj2_second_derivative = obj2.derivative(eval_times, order=2)
        assert obj1_evaluate.shape == expected_size
        assert obj2_evaluate.shape == expected_size
        assert obj1_derivative.shape == expected_size
        assert obj2_derivative.shape == expected_size
        assert obj1_second_derivative.shape == expected_size
        assert obj2_second_derivative.shape == expected_size
        assert obj1_evaluate.allclose(obj2_evaluate)
        assert obj1_derivative.allclose(obj2_derivative)
        assert obj1_second_derivative.allclose(obj2_second_derivative, atol=1e-4, rtol=1e-3)


def test_example1():
    import torch
    from torchcubicspline import (natural_cubic_spline_coeffs,
                                  NaturalCubicSpline)

    length, channels = 7, 3
    t = torch.linspace(0, 1, length)
    x = torch.rand(length, channels)
    coeffs = natural_cubic_spline_coeffs(t, x)
    spline = NaturalCubicSpline(coeffs)
    point = torch.tensor(0.4)
    out = spline.evaluate(point)
    assert out.shape == (3,)


def test_example2():
    import torch
    from torchcubicspline import (natural_cubic_spline_coeffs,
                                  NaturalCubicSpline)

    t = torch.linspace(0, 1, 7)
    # (2, 1) are batch dimensions. 7 is the time dimension
    # (of the same length as t). 3 is the channel dimension.
    x = torch.rand(2, 1, 7, 3)
    coeffs = natural_cubic_spline_coeffs(t, x)
    # coeffs is a tuple of tensors

    # ...at this point you can save the coeffs, put them
    # through PyTorch's Datasets and DataLoaders, etc...

    spline = NaturalCubicSpline(coeffs)

    point = torch.tensor(0.4)
    # will be a tensor of shape (2, 1, 3), corresponding to
    # batch, batch, and channel dimensions
    out = spline.derivative(point)
    assert out.shape == (2, 1, 3)

    point = torch.tensor([[0.4, 0.5]])
    # will be a tensor of shape (2, 1, 1, 2, 3), corresponding to
    # batch, batch, time, time and channel dimensions
    out = spline.derivative(point)
    assert out.shape == (2, 1, 1, 2, 3)


def test_specification():
    for _ in range(10):
        for num_batch_dims in (0, 1, 2, 3):
            batch_dims = []
            for _ in range(num_batch_dims):
                batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
            length = torch.randint(low=5, high=10, size=(1,)).item()
            channels = torch.randint(low=1, high=5, size=(1,)).item()
            t = torch.linspace(0, 1, length, dtype=torch.float64)
            x = torch.rand(*batch_dims, length, channels, dtype=torch.float64)
            coeffs = torchcubicspline.natural_cubic_spline_coeffs(t, x)
            spline = torchcubicspline.NaturalCubicSpline(coeffs)
            for i, point in enumerate(t):
                out = spline.evaluate(point)
                xi = x[..., i, :]
                assert out.allclose(xi)
