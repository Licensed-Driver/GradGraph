# GradGraph

An open source library for exact gradients and Hessians for C# using forward-mode auto differentiation built using [MathNet.Numerics](https://numerics.mathdotnet.com/).

Allows you to express a function with normal operations and have the fully computed value, gradient, and Hessian carried through at each step without inefficient and unstable estimations.
This was originally created as a way for me to have a general framework for defining continuous differentiable non-euclidean fields in 3D space using the pullback metric. I thought using it to define weird and warped movement in a game would be cool, but after an afternoon of making the framework, it was just kinda mid. So I decided I might as well use it somehow! It was already made as a general case so that I could change the functions easily for testing out different effects, and here we are.

## Exampler
```csharp
using GradGraph;

// f(x, y) = sin(x) * y + x / y at (x, y) = (1, 2)
var p = new GradVector(1.0, 2.0);
GradVal x = p[0], y = p[1];

GradVal f = x.Sin() * y + x / y;

double value = f.Value;    // f(1, 2), accessed read only reference
var gradient = f.GetGradient(); // [df/dx, df/dy], clones
var hessian  = f.GetHessian();  // [[d2f/dx2, d2f/dxdy], [d2f/dydx, d2f/dy2]], clones

GradVector pMod = p.Append(f);  // [1, 2, f(1, 2)]

var jacobian = pMod.GetJacobian(); // [[1,0],[0,1],[df/dx,df/dy]]
```

## Install

**.NET**

```
dotnet add package GradGraph
```

**Unity**

GradGraph uses `netstandard2.1` and `C# 9`, so it can be used in Unity. Install it with [NuGetForUnity](https://github.com/GlitchEnzo/NuGetForUnity), which also pulls in MathNet.Numerics. Or you can copy `GradGraph.dll` and `MathNet.Numerics.dll` into `Assets/Plugins`.

## Usage

### Inputs

`new GradVector(params double[])` turns regular numbers into differentiable inputs. Element `i` gets a gradient of `e_i` (1 in position `i`, 0 elsewhere) and a zero Hessian, so everything computed from them is differentiated with respect to that set of inputs.

```csharp
var inputs = new GradVector(x0, y0, z0);   // 3 inputs
GradVal x = inputs[0], y = inputs[1], z = inputs[2];
```

You can also build a single value directly:

```csharp
var x = new GradVal(size: 3, position: 0, value: 1.5);  // input #0 of a 3 vector
var c = new GradVal(2.0);   // constant (zero gradient)
```


### Operations

| | Supported |
|---|---|
| `GradVal` with `GradVal` | `+` `-` `*` `/` |
| `GradVal` with `double` | `+` `-` `*` `/` (either side), unary `-` |
| Functions | `Sin()`, `Cos()` |
| Comparisons | `<` `>` `<=` `>=` `==` `!=` (compare values only) |

All operations return new values to keep mid-function Gradients and Hessians stable mid-equation; a `GradVal` is immutable, and `GetGradient()` / `GetHessian()` return copies.

### Vectors and Jacobians

`GradVector` holds `GradVal`s and gives you vector-valued functions and their Jacobians.

```csharp
// F(x, y) = (x * y, sin x, x + y)
var p = new GradVector(2.0, 3.0);
var F = new GradVector(p[0] * p[1], p[0].Sin(), p[0] + p[1]);

var J = F.GetJacobian();   // 3x2 MathNet Matrix<double>
```

| Member | Description |
|---|---|
| `v[i]` | Read element `i` as a `GradVal` |
| `v.Length` | Number of elements |
| `a + b`, `a - b` | Element-wise add / subtract |
| `v * s`, `s * v`, `v / s` | Scale by a `double` |
| `v.Append(GradVal)` | Append one value, keeping derivatives |
| `v.Extend(IEnumerable<GradVal>)` | Append several values, keeping derivatives |
| `GradVector.Concat(a, b)` | Join two vectors, keeping derivatives |
| `v.Apply(Func<GradVal, GradVal>)` | Apply a function to every element |
| `v.GetJacobian()` | Matrix of every element's gradient, one row per element |
| `GradVector.Constants(params double[])` | Vector of constants (zero gradient) |
| `GradVector.Zeros(n)` | Vector of `n` zero constants |
| `GradMath.Dot(a, b)` | Dot product as a `GradVal` |
| `GradMath.Sin(v)` | Element-wise `Sin` |

`GradVector` also implements `IEnumerable<GradVal>`, so LINQ and `foreach` can enumerate its GradVals.

### Example: pulling back a metric

I used this initially for 3D non-euclidean geometry, by embeding a space with a map `F`, take its Jacobian and pull back the metric with `g = JᵀJ`. Second derivatives of the embedding just come from `GetHessian()`.

```csharp
// Graph embedding (x, y, z) -> (x, y, z, h(x, y, z))
var p = new GradVector(0.3, -1.2, 0.8);
GradVal h = p[0].Sin() * (p[1] + p[2].Sin()).Cos();
var F = p.Append(h);

var J = F.GetJacobian();            // 4x3
var g = J.TransposeThisAndMultiply(J); // pulled-back metric, 3x3
var H = h.GetHessian();             // second derivatives of the embedding
```

## How it works

Each `GradVal` stores a value, a gradient vector and a dense Hessian matrix, all with respect to the same `n` inputs. Every operation applies the first and second-order chain rule directly, for example for a product:
```
∇(ab)  = b∇a + a∇b
∇²(ab) = b∇²a + a∇²b + ∇a∇bᵀ + ∇b∇aᵀ
```

and for any univariate function `h` (like `sin`):

```
∇h(a)  = h'(a)∇a
∇²h(a) = h''(a)∇a∇aᵀ + h'(a)∇²a
```

Because the derivatives are propagated exactly, the results are accurate to floating-point precision, and each evaluation is a single pass. Numerical differentiation needs extra function evaluations for every input (and more again for a Hessian) and has both truncation and rounding errors that it has to balance.

## Limitations

- **Functions:** only `Sin` and `Cos` at the moment. More (`Exp`, `Log`, `Sqrt`, `Pow`, …) are planned.
- **Memory:** each value stores an `n × n` Hessian, so this suits problems with a small number of inputs (tens, not thousands).
- **Default values:** a `default(GradVal)` has no gradient or Hessian. Thus null gradients and Hessians are treated as constants, and will maintain mathematical accuracy accordingly.

## Building and testing

```
dotnet build src/GradGraph
dotnet test tests/GradGraph.Tests
```

The current tests were AI-generated, which I plan to audit and rewrite by hand. The tests check the derivative rules against closed-form results and check gradients and Hessians of composite functions against central finite differences.

## License

MIT
