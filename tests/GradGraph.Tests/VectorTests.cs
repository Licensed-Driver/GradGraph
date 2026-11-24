using System;
using System.Linq;
using GradGraph;
using Xunit;
using static GradGraph.Tests.TestHelpers;

namespace GradGraph.Tests
{
    public class VectorTests
    {
        [Fact]
        public void Constructor_FromDoubles_SetsValuesAndLength()
        {
            var v = new Vector(new[] { 1.0, 2.0, 3.0 });

            Assert.Equal(3, v.Length);
            Assert.Equal(new[] { 1.0, 2.0, 3.0 }, v.ToArray());
        }

        [Fact]
        public void Enumerator_YieldsValues()
        {
            var v = new Vector(new[] { 4.0, 5.0 });
            var seen = new System.Collections.Generic.List<double>();

            foreach (double d in v)
                seen.Add(d);

            Assert.Equal(new[] { 4.0, 5.0 }, seen);
            Assert.Equal(9.0, v.Sum());
        }

        [Fact]
        public void Jacobian_OfInputs_IsIdentity()
        {
            var v = new Vector(new[] { 1.0, 2.0, 3.0 });
            var j = v.GetJacobian();

            AssertClose(new[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } }, j);
        }

        [Fact]
        public void Jacobian_OfMap_MatchesClosedForm()
        {
            // F(x, y) = (x * y, sin x, x + y) at (2, 3)
            var p = new Vector(new[] { 2.0, 3.0 });
            var F = new Vector(new[] { p[0] * p[1], p[0].Sin(), p[0] + p[1] });

            var j = F.GetJacobian();

            AssertClose(new[,]
            {
                { 3.0, 2.0 },
                { Math.Cos(2.0), 0.0 },
                { 1.0, 1.0 },
            }, j);
        }

        [Fact]
        public void Extend_AppendsAndKeepsDerivatives()
        {
            var p = new Vector(new[] { 2.0, 3.0 });
            var extended = p.Extend(new[] { p[0] * p[1] });

            Assert.Equal(3, extended.Length);
            Assert.Equal(new[] { 2.0, 3.0, 6.0 }, extended.ToArray());
            AssertClose(new[] { 3.0, 2.0 }, extended[2].GetGradient());
            AssertClose(new[] { 1.0, 0.0 }, extended[0].GetGradient());
        }

        [Fact]
        public void Concat_JoinsBothVectorsInOrder()
        {
            var p = new Vector(new[] { 1.0, 2.0 });
            var a = new Vector(new[] { p[0], p[1] });
            var b = new Vector(new[] { p[0] * p[1], p[0] + 10.0, p[1].Sin() });

            var joined = Vector.Concat(a, b);

            Assert.Equal(5, joined.Length);
            Assert.Equal(new[] { 1.0, 2.0, 2.0, 11.0, Math.Sin(2.0) }, joined.ToArray());
            AssertClose(new[] { 2.0, 1.0 }, joined[2].GetGradient());
        }

        [Fact]
        public void AddAndSubtract_AreElementwise()
        {
            var p = new Vector(new[] { 1.0, 2.0 });
            var q = new Vector(new[] { p[1], p[0] }); // swapped

            var sum = p + q;
            var diff = p - q;

            Assert.Equal(new[] { 3.0, 3.0 }, sum.ToArray());
            Assert.Equal(new[] { -1.0, 1.0 }, diff.ToArray());
            AssertClose(new[] { 1.0, 1.0 }, sum[0].GetGradient());
            AssertClose(new[] { 1.0, -1.0 }, diff[0].GetGradient());
        }

        [Fact]
        public void AddingDifferentLengths_Throws()
        {
            var a = new Vector(new[] { 1.0, 2.0 });
            var b = new Vector(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<IndexOutOfRangeException>(() => a + b);
        }

        [Fact]
        public void ScalarMultiplyAndDivide_ScaleValuesAndGradients()
        {
            var p = new Vector(new[] { 1.0, 2.0 });

            var doubled = 2.0 * p;
            var halved = p / 2.0;

            Assert.Equal(new[] { 2.0, 4.0 }, doubled.ToArray());
            Assert.Equal(new[] { 0.5, 1.0 }, halved.ToArray());
            AssertClose(new[] { 0.0, 2.0 }, doubled[1].GetGradient());
            AssertClose(new[] { 0.5, 0.0 }, halved[0].GetGradient());
        }

        [Fact]
        public void Dot_MatchesClosedForm()
        {
            // f = x*y + y*z with a = (x, y), b = (y, z) at (1, 2, 3)
            var p = new Vector(new[] { 1.0, 2.0, 3.0 });
            var a = new Vector(new[] { p[0], p[1] });
            var b = new Vector(new[] { p[1], p[2] });

            var f = GradMath.Dot(a, b);

            AssertClose(8.0, f.GetValue());
            AssertClose(new[] { 2.0, 4.0, 2.0 }, f.GetGradient());
            AssertClose(new[,]
            {
                { 0.0, 1.0, 0.0 },
                { 1.0, 0.0, 1.0 },
                { 0.0, 1.0, 0.0 },
            }, f.GetHessian());
        }

        [Fact]
        public void Dot_DifferentLengths_Throws()
        {
            var a = new Vector(new[] { 1.0, 2.0 });
            var b = new Vector(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<IndexOutOfRangeException>(() => GradMath.Dot(a, b));
        }

        [Fact]
        public void CopyConstructor_DoesNotShareStorage()
        {
            var p = new Vector(new[] { 1.0, 2.0 });
            var arr = new[] { p[0], p[1] };
            var fromArray = new Vector(arr);

            arr[0] = p[1]; // mutate the source array after construction

            Assert.Equal(new[] { 1.0, 2.0 }, fromArray.ToArray());
            Assert.Equal(new[] { 1.0, 2.0 }, new Vector(fromArray).ToArray());
        }
    }
}
