# torchcubicspline
Interpolating natural cubic splines using PyTorch. Includes for:
- Batching
- GPU support and backpropagation via PyTorch
- Support for missing values (represent them as NaN)
- Evaluating the first derivative of the spline

Functionality is provided via the `NaturalCubicSpline` class. It takes an increasing sequence of times represented by a tensor `times` of shape `(length,)` and some corresponding observations `X` of shape `(..., length, channels)`, where `...` are batch dimensions, and each `(length, channels)` slice represents `length` points, each in Euclidean space of dimension `channels`.

Then calling `spline = NaturalCubicSpline(times, X)` produces an instance `spline` such that
```
spline.evaluate(times[i]) == X[..., i, :]
```
for all `i`.

Furthermore the derivative of the spline at a point may be calculated via `spline.derivative`. (Not be confused with backpropagation, which is also supported.)

Any issues - let me know!
