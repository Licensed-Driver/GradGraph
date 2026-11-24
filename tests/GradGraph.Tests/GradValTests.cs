using System;
using System.Collections.Generic;
using GradGraph;
using Xunit;
using static GradGraph.Tests.TestHelpers;

namespace GradGraph.Tests
{
    public class GradValTests
    {
        // ---------- Inputs ----------

        [Fact]
        public void InputVariable_HasUnitGradientAndZeroHessian()
        {
            var v = new Vector(new[] { 3.0, 5.0, 7.0 });

            for (int i = 0; i < 3; i++)
            {
                var expectedGrad = new double[3];
                expectedGrad[i] = 1.0;

                AssertClose(new[] { 3.0, 5.0, 7.0 }[i], v[i].GetValue());
                AssertClose(expectedGrad, v[i].GetGradient());
                AssertClose(new double[3, 3], v[i].GetHessian());
            }
        }

        [Fact]
        public void SizeConstructor_WithPosition_IsInputVariable()
        {
            var x = new GradVal(3, position: 1, value: 2.5);

            AssertClose(2.5, x.GetValue());
            AssertClose(new[] { 0.0, 1.0, 0.0 }, x.GetGradient());
            AssertClose(new double[3, 3], x.GetHessian());
            Assert.Equal(3, x.Length());
        }

        [Fact]
        public void SizeConstructor_WithoutPosition_IsConstant()
        {
            var c = new GradVal(2, value: 4.0);

            AssertClose(4.0, c.GetValue());
            AssertClose(new[] { 0.0, 0.0 }, c.GetGradient());
            AssertClose(new double[2, 2], c.GetHessian());
        }

        // ---------- Closed-form rules ----------

        [Fact]
        public void Product_MatchesClosedForm()
        {
            // f = x * y at (3, 5): grad = (y, x), Hessian = [[0, 1], [1, 0]]
            var v = new Vector(new[] { 3.0, 5.0 });
            var f = v[0] * v[1];

            AssertClose(15.0, f.GetValue());
            AssertClose(new[] { 5.0, 3.0 }, f.GetGradient());
            AssertClose(new[,] { { 0.0, 1.0 }, { 1.0, 0.0 } }, f.GetHessian());
        }

        [Fact]
        public void Square_HasHessianOfTwo()
        {
            // f = x * x at x = 4: f' = 8, f'' = 2
            var x = new Vector(new[] { 4.0 })[0];
            var f = x * x;

            AssertClose(16.0, f.GetValue());
            AssertClose(new[] { 8.0 }, f.GetGradient());
            AssertClose(new[,] { { 2.0 } }, f.GetHessian());
        }

        [Fact]
        public void Reciprocal_MatchesClosedForm()
        {
            // f = a / x at x = 2, a = 3: f' = -a/x^2, f'' = 2a/x^3
            var x = new Vector(new[] { 2.0 })[0];
            var f = 3.0 / x;

            AssertClose(1.5, f.GetValue());
            AssertClose(new[] { -0.75 }, f.GetGradient());
            AssertClose(new[,] { { 0.75 } }, f.GetHessian());
        }

        [Fact]
        public void Sin_MatchesClosedForm()
        {
            const double x0 = 0.7;
            var f = new Vector(new[] { x0 })[0].Sin();

            AssertClose(Math.Sin(x0), f.GetValue());
            AssertClose(new[] { Math.Cos(x0) }, f.GetGradient());
            AssertClose(new[,] { { -Math.Sin(x0) } }, f.GetHessian());
        }

        [Fact]
        public void Cos_MatchesClosedForm()
        {
            const double x0 = 0.7;
            var f = new Vector(new[] { x0 })[0].Cos();

            AssertClose(Math.Cos(x0), f.GetValue());
            AssertClose(new[] { -Math.Sin(x0) }, f.GetGradient());
            AssertClose(new[,] { { -Math.Cos(x0) } }, f.GetHessian());
        }

        [Fact]
        public void ScalarOperators_ShiftAndScaleCorrectly()
        {
            var v = new Vector(new[] { 2.0, 3.0 });
            var xy = v[0] * v[1]; // value 6, grad (3, 2), H = [[0,1],[1,0]]

            var shifted = 10.0 + xy - 1.0;
            AssertClose(15.0, shifted.GetValue());
            AssertClose(new[] { 3.0, 2.0 }, shifted.GetGradient());
            AssertClose(new[,] { { 0.0, 1.0 }, { 1.0, 0.0 } }, shifted.GetHessian());

            var scaled = 2.0 * xy / 4.0;
            AssertClose(3.0, scaled.GetValue());
            AssertClose(new[] { 1.5, 1.0 }, scaled.GetGradient());
            AssertClose(new[,] { { 0.0, 0.5 }, { 0.5, 0.0 } }, scaled.GetHessian());

            var negated = -xy;
            AssertClose(-6.0, negated.GetValue());
            AssertClose(new[] { -3.0, -2.0 }, negated.GetGradient());

            var flipped = 1.0 - xy;
            AssertClose(-5.0, flipped.GetValue());
            AssertClose(new[] { -3.0, -2.0 }, flipped.GetGradient());
            AssertClose(new[,] { { 0.0, -1.0 }, { -1.0, 0.0 } }, flipped.GetHessian());
        }

        // ---------- Composite functions vs finite differences ----------

        public static IEnumerable<object[]> Composites()
        {
            Func<Vector, GradVal> field1 = p => p[0].Sin() * (p[1] + p[2].Sin()).Cos();
            Func<Vector, GradVal> field2 = p => (p[0] + p[1] * p[2].Sin()).Sin();
            Func<Vector, GradVal> rational = p => p[0] * p[1] / (1.0 + p[2] * p[2]);
            Func<Vector, GradVal> mixed = p => (2.0 - p[0]) / (p[1] + 3.0) * p[2].Cos() - 0.5 * p[0];

            var points = new[]
            {
                new[] { 0.3, -1.2, 0.8 },
                new[] { 1.7, 0.4, -2.1 },
            };

            foreach (var point in points)
            {
                yield return new object[] { "sin(x)cos(y+sin z)", field1, point };
                yield return new object[] { "sin(x + y sin z)", field2, point };
                yield return new object[] { "xy/(1+z^2)", rational, point };
                yield return new object[] { "(2-x)/(y+3) cos z - x/2", mixed, point };
            }
        }

        [Theory]
        [MemberData(nameof(Composites))]
        public void Gradient_MatchesFiniteDifferences(string name, Func<Vector, GradVal> f, double[] point)
        {
            _ = name; // shown in the test runner to identify the case
            var result = f(new Vector(point));
            AssertClose(NumericGradient(f, point), result.GetGradient(), tol: 1e-7);
        }

        [Theory]
        [MemberData(nameof(Composites))]
        public void Hessian_MatchesFiniteDifferences(string name, Func<Vector, GradVal> f, double[] point)
        {
            _ = name;
            var result = f(new Vector(point));
            var hessian = result.GetHessian();

            AssertSymmetric(hessian);
            AssertClose(NumericHessian(f, point), hessian, tol: 1e-5);
        }

        // ---------- Comparison and equality ----------

        [Fact]
        public void ComparisonOperators_CompareValuesOnly()
        {
            var v = new Vector(new[] { 1.0, 2.0 });

            Assert.True(v[0] < v[1]);
            Assert.True(v[1] > 1.5);
            Assert.True(0.5 <= v[0]);
            Assert.True(v[0] == 1.0);
            Assert.True(v[0] != v[1]);
        }

        [Fact]
        public void Equals_ComparesValueAndDerivatives()
        {
            var a = new Vector(new[] { 1.0, 2.0 })[0];
            var b = new Vector(new[] { 1.0, 2.0 })[0];
            var c = new Vector(new[] { 1.0, 2.0 })[1] - 1.0; // value 1.0 but different gradient

            Assert.True(a.Equals(b));
            Assert.Equal(a.GetHashCode(), b.GetHashCode());

            Assert.True(a == c);       // == only looks at Value
            Assert.False(a.Equals(c)); // Equals also looks at the derivatives
        }

        // ---------- Immutability ----------

        [Fact]
        public void Getters_ReturnCopies()
        {
            var x = new Vector(new[] { 1.0, 2.0 })[0];

            x.GetGradient()[0] = 99.0;
            x.GetHessian()[0, 0] = 99.0;

            AssertClose(new[] { 1.0, 0.0 }, x.GetGradient());
            AssertClose(new double[2, 2], x.GetHessian());
        }
    }
}
