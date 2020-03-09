# torchcubicspline
Interpolating natural cubic splines using PyTorch. Includes support for:
- Batching
- GPU support and backpropagation via PyTorch
- Support for missing values (represent them as NaN)
- Evaluating the first derivative of the spline

## Functionality

Functionality is provided via the `NaturalCubicSpline` class. It takes an increasing sequence of times represented by a tensor `times` of shape `(length,)` and some corresponding observations `X` of shape `(..., length, channels)`, where `...` are batch dimensions, and each `(length, channels)` slice represents `length` points, each in Euclidean space of dimension `channels`.

Then calling `spline = NaturalCubicSpline(times, X)` produces an instance `spline` such that
```
spline.evaluate(times[i]) == X[..., i, :]
```
for all `i`.

Furthermore the derivative of the spline at a point may be calculated via `spline.derivative`. (Not be confused with backpropagation, which is also supported through both `spline.evaluate` and `spline.derivative`.)

Finally, support for missing values is done by passing them as `NaN`. In particular this allows for batching elements with different observation times: take `times` to be the observation times of all elements in the batch, and just set each element to have a missing observation `NaN` at the times of the observations of the other batch elements.

## Limitations

If possible, you should cache the instance `spline` created by the call `spline = NaturalCubicSpline(times, X)`. This is because whilst `spline.evaluate` and `spline.derivative` should be quick to evaluate, some of the initial computations during `__init__` aren't parallelised as much as perhaps they could be. In general slow down is expected when:
- `length` is large and the batch sizes are small (due to non-parallelism within `misc.tridiagonal_solve`).
- or if the batch sizes are large and you have missing data (due to non-parallelism when working over every batch element to figure out where the missing values are).

## Any issues?

Any issues - let me know!
