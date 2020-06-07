# torchcubicspline
Interpolating natural cubic splines using PyTorch. Includes support for:
- Batching
- GPU support and backpropagation via PyTorch
- Support for missing values (represent them as NaN)
- Evaluating the first derivative of the spline

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchcubicspline.git
```

## Example

Simple example:
```python
import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

length, channels = 7, 3
t = torch.linspace(0, 1, length)
x = torch.rand(length, channels)
coeffs = natural_cubic_spline_coeffs(t, x)
spline = NaturalCubicSpline(coeffs)
point = torch.tensor(0.4)
out = spline.evaluate(point)
```

With multiple batch and evaluation dimensions:
```python
import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
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

point = torch.tensor([[0.4, 0.5]])
# will be a tensor of shape (2, 1, 1, 2, 3), corresponding to
# batch, batch, time, time and channel dimensions
out = spline.derivative(point)
```

## Functionality

Functionality is provided via the `natural_cubic_spline_coeffs` function and `NaturalCubicSpline` class.

`natural_cubic_spline_coeffs` takes an increasing sequence of times represented by a tensor `t` of shape `(length,)` and some corresponding observations `x` of shape `(..., length, channels)`, where `...` are batch dimensions, and each `(length, channels)` slice represents a sequence of `length` points, each point with `channels` many values.

Then calling
```python
coeffs = natural_cubic_spline_coeffs(t, x)
spline = NaturalCubicSpline(coeffs)
```
produces an instance `spline` such that
```
spline.evaluate(t[i]) == x[..., i, :]
```
for all `i`.

#### Why is there a function and a class?

The slow bit is done during `natural_cubic_spline_coeffs`. The fast bit is `NaturalCubicSpline`. The returned `coeffs` are a tuple of PyTorch tensors, so you can take this opportunity to save or load them, push them through `torch.utils.data.Dataset` or `torch.utils.data.DataLoader`, etc.

#### Derivatives

The derivative of the spline at a point may be calculated via `spline.derivative`. (Not be confused with backpropagation, which is also supported through both `spline.evaluate` and `spline.derivative`.)

#### Missing values

Support for missing values is done by setting that element of `x` to `NaN`. In particular this allows for batching elements with different observation times: take `times` to be the observation times of all elements in the batch, and just set each element to have a missing observation `NaN` at the times of the observations of the other batch elements.

## Limitations

If possible, you should cache the coefficients returned by `natural_cubic_spline_coeffs`. In particular if there are missing values then the computation can be quite slow.

## Any issues?

Any issues or questions - open an issue to let me know. :)
