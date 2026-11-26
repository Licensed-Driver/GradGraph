
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace GradGraph
{

    /// <summary>
    /// A scalar that carries its exact gradient and Hessian through every operation (forward-mode automatic differentiation).
    /// Every operation returns a new GradVal with the value, gradient, and Hessian updated by the chain rule.
    /// GradVals with no gradient or Hessian (e.g. from a double) are treated as constants.
    /// </summary>
    public readonly struct GradVal
    {

        // The derivative of it's vector with respect to the initial value
        private readonly Vector<double> _gradient;
        // The second derivative of it's vector with respect to the initial value
        private readonly Matrix<double> _hessian;
        // Private constructor to copy over all values from another GradVal into a new one
        private GradVal(double value, Vector<double> derivative, Matrix<double> secondDeriv) => (Value, _gradient, _hessian) = (value, derivative, secondDeriv);

        /// <summary>
        /// Double value of this GradVal.
        /// </summary>
        public double Value { get; }

        /// <summary>
        /// Retrives a copy of the gradient for this GradVal instance.
        /// </summary>
        /// <returns>A clone of the gradient vector</returns>
        public Vector<double> GetGradient() => _gradient?.Clone();

        /// <summary>
        /// Retrieves a copy of the Hessian matrix for this GradVal instance.
        /// </summary>
        /// <returns>A clone of the Hessian matrix</returns>
        public Matrix<double> GetHessian() => _hessian?.Clone();

        /// <summary>
        /// The amount of inputs for this gradval.
        /// i.e. The number of variables this GradVal depends on, and the length of the gradient Vector.
        /// </summary>
        public int Length => _gradient?.Count ?? 0;

        /// <summary>
        /// Create a new GradVal that represents a constant with no gradient
        /// </summary>
        /// <param name="value">The constant</param>
        public GradVal(double value) => (Value, _gradient, _hessian) = (value, default, default);

        /// <summary>
        /// Creates a new GradVal.
        /// The value is selected as <paramref name="vals"/>[<paramref name="position"/>].
        /// The gradient and Hessian are created from <paramref name="vals"/>.
        /// </summary>
        /// <param name="vals">The vector this value is from (i.e. the values it depends on)</param>
        /// <param name="position">The position of this value in <paramref name="vals"/></param>
        public GradVal(IEnumerable<double> vals, int position) : 
            this(vals.ToArray(), position) {}

        /// <summary>
        /// Creates a new GradVal.
        /// The value is selected as <paramref name="vals"/>[<paramref name="position"/>].
        /// The gradient and Hessian are created from <paramref name="vals"/>.
        /// </summary>
        /// <param name="vals">The vector this value is from (i.e. the values it depends on)</param>
        /// <param name="position">The position of this value in <paramref name="vals"/></param>
        public GradVal(double[] vals, int position)
        {
            if (position >= vals.Length || position < 0) throw new IndexOutOfRangeException($"Position {position} is out of range for {vals.Length} values");
            Value = vals[position];
            _gradient = Vector<double>.Build.Dense(vals.Length, i => i == position ? 1.0 : 0.0);   // All start at 0
            _hessian = Matrix<double>.Build.Dense(vals.Length, vals.Length);
        }

        /// <summary>
        /// Creates a new GradVal with <paramref name="value"/> at <paramref name="position"/> in a vector of <paramref name="size"/>
        /// </summary>
        /// <param name="value">The value of this GradVal, which is dependent on all variables in a vector of <paramref name="size"/></param>
        /// <param name="position">The position of <paramref name="value"/> within its source vector</param>
        /// <param name="size">The size of this GradVals source vector</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public GradVal(double value, int position, int size)
        {
            if (position >= size || position < 0) throw new IndexOutOfRangeException($"Position {position} is out of range for {size} values");
            Value = value;
            _gradient = Vector<double>.Build.Dense(size, i => i == position ? 1.0 : 0);
            _hessian = Matrix<double>.Build.Dense(size, size);
        }

        /// <summary>
        /// Applies the chain rule for a scalar function with value f, derivative df, and second derivative d2f
        /// </summary>
        /// <param name="f">The value of the scalar function</param>
        /// <param name="df">The first derivative of the scalar function</param>
        /// <param name="d2f">The second derivative of the scalar function</param>
        /// <returns>A new GradVal with the value, gradient, and Hessian fully calculated during the forward computation.</returns>
        public GradVal Chain(double f, double df, double d2f) =>
            new GradVal(f, Scale(_gradient, df), Combine(OuterProduct(_gradient, _gradient), d2f, _hessian, df));

        /// <summary>
        /// Calculate a new GradVal holding the Sine of this GradVal.
        /// Continues the gradient and Hessian through the first and second derivative of the Sine function.
        /// </summary>
        /// <returns>A new GradVal that contains the exact value, gradient, and Hessian of this GradVal sined</returns>
        public GradVal Sin() => Chain(Math.Sin(Value), Math.Cos(Value), -Math.Sin(Value));

        /// <summary>
        /// Calculate a new GradVal holding the Cosine of this GradVal.
        /// Continues the gradient and Hessian through the first and second derivative of the Cosin function.
        /// </summary>
        /// <returns>A new GradVal that contains the exact value, gradient, and Hessian of this GradVal Cosined</returns>
        public GradVal Cos() => Chain(Math.Cos(Value), -Math.Sin(Value), -Math.Cos(Value));

        // All operations involving scalars
        /// <summary>
        /// Unary negative operator. Makes value negative, preserving gradient and Hessian
        /// </summary>
        /// <param name="a">GradVal to negate.</param>
        /// <returns></returns>
        public static GradVal operator -(GradVal a) => new GradVal(-a.Value, Scale(a._gradient, -1), Scale(a._hessian, -1));
        /// <summary>
        /// Implicit cast operator from double to GradVal constant.
        /// Resulting GradVal does not contain a Gradient or Hessian, as it is assumed to be a constant.
        /// </summary>
        /// <param name="a">double to cast</param>
        public static implicit operator GradVal(double a) => new GradVal(a);

        /// <summary>
        /// Add a constant to the GradVal.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">GradVal to add to</param>
        /// <param name="b">Constant to be added</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from Constant addition</returns>
        public static GradVal operator +(GradVal a, double b) => new GradVal(a.Value+b, a._gradient, a._hessian);
        /// <summary>
        /// Add a constant to the GradVal.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Constant to be added</param>
        /// <param name="b">GradVal to add to</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from Constant addition</returns>
        public static GradVal operator +(double a, GradVal b) => new GradVal(a + b.Value, b._gradient, b._hessian);

        /// <summary>
        /// Subtract a constant from the GradVal.
        /// Preserves accurate gradient and Hesssian.
        /// </summary>
        /// <param name="a">GradVal to be subtracted from</param>
        /// <param name="b">Constant to subtract</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from Constant subtraction</returns>
        public static GradVal operator -(GradVal a, double b) => new GradVal(a.Value-b, a._gradient, a._hessian);
        /// <summary>
        /// Subtract a GradVal from a constant.
        /// Preserves accurate gradient and Hesssian.
        /// </summary>
        /// <param name="a">Constant to be subtracted from</param>
        /// <param name="b">GradVal to subtract</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from Constant subtraction</returns>
        public static GradVal operator -(double a, GradVal b) => new GradVal(a - b.Value, Scale(b._gradient, -1), Scale(b._hessian, -1));

        /// <summary>
        /// Multiply a GradVal by a constant.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">GradVal to be scalar multiplied</param>
        /// <param name="b">Constant to multiply by</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from scalar multiplication</returns>
        public static GradVal operator *(GradVal a, double b) => new GradVal(a.Value * b, Scale(a._gradient, b), Scale(a._hessian, b));
        /// <summary>
        /// Multiply a GradVal by a constant.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Constant to multiply by</param>
        /// <param name="b">GradVal to be scalar multiplied</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from scalar multiplication</returns>
        public static GradVal operator *(double a, GradVal b) => new GradVal(a * b.Value, Scale(b._gradient, a), Scale(b._hessian, a));

        /// <summary>
        /// Divide a GradVal by a constant.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">GradVal to be scalar divided</param>
        /// <param name="b">Constant to divide by</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from scalar division</returns>
        public static GradVal operator /(GradVal a, double b) => new GradVal(a.Value / b, Scale(a._gradient, 1 / b), Scale(a._hessian, 1 / b));
        /// <summary>
        /// Divide a GradVal by a constant.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Constant to get divided</param>
        /// <param name="b">GradVal to divide by</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from scalar division</returns>
        public static GradVal operator /(double a, GradVal b) => b.Chain(a / b.Value, -a / (b.Value * b.Value), 2 * a / Math.Pow(b.Value, 3));

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is greater than <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if value of <paramref name="a"/> is greater than <paramref name="b"/></returns>
        public static bool operator >(GradVal a, double b) => a.Value > b;
        /// <summary>
        /// Compare if <paramref name="a"/> is greater than the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is greater than value of <paramref name="b"/></returns>
        public static bool operator >(double a, GradVal b) => a > b.Value;

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is less than <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if the value of <paramref name="a"/> is less than <paramref name="b"/></returns>
        public static bool operator <(GradVal a, double b) => a.Value < b;
        /// <summary>
        /// Compare if <paramref name="a"/> is less than the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is less than the value of <paramref name="b"/></returns>
        public static bool operator <(double a, GradVal b) => a < b.Value;

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is equal to <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if the value of <paramref name="a"/> is equal to <paramref name="b"/></returns>
        public static bool operator ==(GradVal a, double b) => a.Value == b;
        /// <summary>
        /// Compare if <paramref name="a"/> is equal to the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is equal to the value of <paramref name="b"/></returns>
        public static bool operator ==(double a, GradVal b) => a == b.Value;

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is not equal to <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if the value of <paramref name="a"/> is not equal to <paramref name="b"/></returns>
        public static bool operator !=(GradVal a, double b) => a.Value != b;
        /// <summary>
        /// Compare if <paramref name="a"/> is not equal to the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is not equal to the value of <paramref name="b"/></returns>
        public static bool operator !=(double a, GradVal b) => a != b.Value;

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is less than or equal to <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if the value of <paramref name="a"/> is less than or equal to <paramref name="b"/></returns>
        public static bool operator <=(GradVal a, double b) => a.Value <= b;
        /// <summary>
        /// Compare if <paramref name="a"/> is less than or equal to the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is less than or equal to the value of <paramref name="b"/></returns>
        public static bool operator <=(double a, GradVal b) => a <= b.Value;

        /// <summary>
        /// Compare if the value of <paramref name="a"/> is greater than or equal to <paramref name="b"/>.
        /// </summary>
        /// <param name="a">GradVal to compare the value of</param>
        /// <param name="b">double to compare against</param>
        /// <returns>True if the value of <paramref name="a"/> is greater than or equal to <paramref name="b"/></returns>
        public static bool operator >=(GradVal a, double b) => a.Value >= b;
        /// <summary>
        /// Compare if <paramref name="a"/> is greater than or equal to the value of <paramref name="b"/>.
        /// </summary>
        /// <param name="a">double to compare</param>
        /// <param name="b">GradVal to compare its value against</param>
        /// <returns>True if <paramref name="a"/> is greater than or equal to the value of <paramref name="b"/></returns>
        public static bool operator >=(double a, GradVal b) => a >= b.Value;

        // All operations involving two GradVals
        /// <summary>
        /// Add two GradVals.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to add</param>
        /// <param name="b">Right GradVal to add</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from addition</returns>
        public static GradVal operator +(GradVal a, GradVal b) => new GradVal(a.Value + b.Value, Combine(a._gradient, 1, b._gradient, 1), Combine(a._hessian, 1, b._hessian, 1));

        /// <summary>
        /// Subtract two GradVals.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">GradVal to get subtracted from</param>
        /// <param name="b">GradVal to subtract</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from subtraction</returns>
        public static GradVal operator -(GradVal a, GradVal b) => new GradVal(a.Value - b.Value, Combine(a._gradient, 1, b._gradient, -1), Combine(a._hessian, 1, b._hessian, -1));

        /// <summary>
        /// Multiply two GradVals.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to multiply</param>
        /// <param name="b">Right GradVal to multiply</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from multiplication</returns>
        public static GradVal operator *(GradVal a, GradVal b) => new GradVal(a.Value * b.Value, Combine(a._gradient, b.Value, b._gradient, a.Value), Combine(Combine(a._hessian, b.Value, b._hessian, a.Value), 1, Combine(OuterProduct(a._gradient, b._gradient), 1, OuterProduct(b._gradient, a._gradient), 1), 1));

        /// <summary>
        /// Divide a GradVal by another GradVal.
        /// Preserves accurate gradient and Hessian.
        /// </summary>
        /// <param name="a">Numerator GradVal</param>
        /// <param name="b">Denominator GradVal</param>
        /// <returns>New GradVal with the value, gradient, and Hessian resulting from division</returns>
        public static GradVal operator /(GradVal a, GradVal b) => a * (1/b);    // All steps copy so we chillin

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is greater than the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is greater than the value of <paramref name="b"/></returns>
        public static bool operator >(GradVal a, GradVal b) => a.Value > b.Value;

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is less than the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is less than the value of <paramref name="b"/></returns>
        public static bool operator <(GradVal a, GradVal b) => a.Value < b.Value;

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is equal to the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is equal to the value of <paramref name="b"/></returns>
        public static bool operator ==(GradVal a, GradVal b) => a.Value == b.Value;

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is not equal to the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is not equal to the value of <paramref name="b"/></returns>
        public static bool operator !=(GradVal a, GradVal b) => a.Value != b.Value;

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is less than or equal to the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is less than or equal to the value of <paramref name="b"/></returns>
        public static bool operator <=(GradVal a, GradVal b) => a.Value <= b.Value;

        /// <summary>
        /// Compare whether the value of <paramref name="a"/> is greater than or equal to the value of <paramref name="b"/>.
        /// Does not compare gradient or Hessian.
        /// </summary>
        /// <param name="a">Left GradVal to compare</param>
        /// <param name="b">Right GradVal to compare</param>
        /// <returns>True if value of <paramref name="a"/> is greater than or equal to the value of <paramref name="b"/></returns>
        public static bool operator >=(GradVal a, GradVal b) => a.Value >= b.Value;

        /// <summary>
        /// Evaluate if given object is equivalent to this GradVal in all aspects.
        /// Is not equivalent to == as this compares value.
        /// </summary>
        /// <param name="obj">The object to compare against.</param>
        /// <returns>True if obj is a GradVal and has the same value, gradient, and hessian. False otherwise.</returns>
        public override bool Equals(object obj)
        {
            // Obviously they aren't equal if it isn't even a gradval
            if (obj is not GradVal other)
                return false;
            // Only true equal if everything is the same
            return Value.Equals(other.Value);
        }

        /// <summary>
        /// Get the hash for this GradVal.
        /// Hash preserves equivalence in value,
        /// i.e. GradVal1.Equals(GradVal2) = True => GradVal1.GetHashCode() == GradVal2.GetHashCode() = True
        /// </summary>
        /// <returns>Hash for this GradVal.</returns>
        public override int GetHashCode() =>
            Value.GetHashCode();

        /*
            Linearly combines the two given derivative arrays using the given coefficients.
        */
        private static Vector<double> Combine(Vector<double> vec1, double coef1, Vector<double> vec2, double coef2)
        {
            // null derivatives mean all zero, so just scale the other one (null if both are null)
            if (vec1 == null) return Scale(vec2, coef2);
            if (vec2 == null) return Scale(vec1, coef1);
            if (vec1.Count != vec2.Count) throw new IndexOutOfRangeException($"Vector of length {vec1.Count} cannot be combined with vector of length {vec2.Count}");
            return coef1 * vec1 + coef2 * vec2;
        }

        /*
            Linearly combines the two given hessians or other matrices with the given coefficients.
        */
        private static Matrix<double> Combine(Matrix<double> mat1, double coef1, Matrix<double> mat2, double coef2)
        {
            // null derivatives mean all zero, so just scale the other one (null if both are null)
            if (mat1 == null) return Scale(mat2, coef2);
            if (mat2 == null) return Scale(mat1, coef1);
            if (mat1.RowCount != mat2.RowCount || mat1.ColumnCount != mat2.ColumnCount) throw new IndexOutOfRangeException($"Matrix of shape [{mat1.RowCount}x{mat1.ColumnCount}] is not able to be linearly combined with matrix of shape [{mat2.RowCount}x{mat2.ColumnCount}]");
            return coef1 * mat1 + coef2 * mat2;
        }

        /*
            Scales the given derivative array by the given coefficient.
        */
        private static Vector<double> Scale(Vector<double> deriv, double coef)
        {
            // Since new GradVal[n] doesn't create the Derivative reference
            if (deriv == null) return null;
            return deriv.Clone() * coef;
        }

        /*
            Scales the given matrix by the given coefficient.
        */
        private static Matrix<double> Scale(Matrix<double> hessian, double coef)
        {
            if (hessian == null) return null;
            return hessian.Clone() * coef;
        }

        /*
            Safely calculates the outer product of two vectors.
        */
        private static Matrix<double> OuterProduct(Vector<double> vec1, Vector<double> vec2)
        {
            if (vec1 == null && vec2 == null) return null;
            if (vec1 == null) return Matrix<double>.Build.Dense(vec2.Count, vec2.Count);
            if (vec2 == null) return Matrix<double>.Build.Dense(vec1.Count, vec1.Count);

            return vec1.OuterProduct(vec2);
        }
    }

    /// <summary>
    /// An immutable vector of GradVals, used for the inputs and outputs of multivariate functions.
    /// Tracks how many inputs its elements depend on, and gives the Jacobian of the elements with respect to those inputs.
    /// </summary>
    public readonly struct GradVector : IEnumerable<GradVal>
    {
        private readonly GradVal[] Values;
        private readonly int _length;
        private readonly int _inputLength;
        /// <summary>
        /// Length of the GradVector.
        /// </summary>
        public readonly int Length {get => _length;}

        /// <summary>
        /// Creates a new GradVector with new GradVals created from <paramref name="values"/>.
        /// All elements in <paramref name="values"/> are assumed to depend on each other and therefore have a gradient length of <paramref name="values"/>.Count().
        /// </summary>
        /// <param name="values">double values to hold</param>
        public GradVector(IEnumerable<double> values)
        {
            var arr = values.ToArray();
            Values = new GradVal[arr.Length];
            for (int i = 0; i < Values.Length; i++)
                Values[i] = new GradVal(arr, i);    // Passed as array to reduce amount of lazy selects
            _length = Values.Length;
            _inputLength = Values.Length;
        }
        
        /// <summary>
        /// Create a new Vector with the values of the given IEnumerable.
        /// Values are cloned, not referenced, to maintain gradient stability mid-function.
        /// </summary>
        /// <param name="values">The value Enumerable to be copied.</param>
        public GradVector(IEnumerable<GradVal> values)
        {
            var arr = values.ToArray();
            Values = arr;
            _length = Values.Length;
            _inputLength = GradMath.InputSize(arr);
        }

        /// <summary>
        /// Create a new vector with the given values.
        /// </summary>
        /// <param name="values">A collection of doubles to be held in the Vector</param>
        public GradVector(params double[] values)
        {
            Values = new GradVal[values.Length];
            _length = values.Length;
            _inputLength = values.Length;
            for (int i = 0; i < Values.Length; i++)
                Values[i] = new GradVal(values, i);
        }

        /// <summary>
        /// Creates a new GradVector that creates a copy of each element in <paramref name="values"/>.
        /// <b>Clones each value, and does not hold a reference.</b>
        /// </summary>
        /// <param name="values">Values to hold</param>
        public GradVector(params GradVal[] values)
        {
            Values = (GradVal[])values.Clone();
            _length = values.Length;
            _inputLength = GradMath.InputSize(values);
        }

        /// <summary>
        /// Creates a new GradVector that creates a copy of each element in <paramref name="toCopy"/>.
        /// <b>Clones the GradVector, and does not hold a reference.</b>
        /// </summary>
        /// <param name="toCopy">GradVector to copy</param>
        public GradVector(GradVector toCopy)
        {
            Values = (GradVal[])toCopy.Values.Clone();
            _length = Values.Length;
            _inputLength = GradMath.InputSize(toCopy);
        }

        /// <summary>
        /// Private constructor for static methods that can compute their own values to avoid an extra clone.
        /// </summary>
        /// <param name="values">Value array reference to be used in values.</param>
        /// <param name="length">Length of <paramref name="values"/></param>
        /// <param name="inputLength">Number of inputs each value depends on.</param>
        private GradVector(GradVal[] values, int length, int inputLength)
        {
            Values = values;
            _length = length;
            _inputLength = inputLength;
        }

        /// <summary>
        /// Allow array indexing of GradVectors.
        /// <b>Can only be used for getting, not setting.</b>
        /// </summary>
        /// <param name="index">Element index</param>
        /// <returns>Element at <paramref name="index"/></returns>
        public GradVal this[int index]
        {
            get => Values[index];
        }

        /// <summary>
        /// Use the gradient of each GradVal to build the Jacobian of the GradVector.
        /// Casts constant GradVals to the input space of non-constant values.
        /// </summary>
        /// <returns>Jacobian Matrix as a MathNet matrix of doubles</returns>
        /// <exception cref="InvalidOperationException">Thrown if the GradVector contains conflicting input spaces.</exception>
        public Matrix<double> GetJacobian()
        {
            if (_inputLength == 0) throw new InvalidOperationException("The Jacobian is undefined for non-differentiable expressions (vector of constants)");
            List<Vector<double>> rows = new List<Vector<double>>();
            // Since we treat null derivatives as all 0s we don't throw before we get here, so we gotta make sure the matrix won't throw em
            for (int i = 0; i < _length; i++)
                rows.Add(Values[i].GetGradient() ?? Vector<double>.Build.Dense(_inputLength, 0.0));
            return Matrix<double>.Build.DenseOfRowVectors(rows);
        }

        /// <summary>
        /// Creates a new GradVector of <paramref name="size"/> of constants.
        /// Because they are constants, they contain a null gradient and Hessian to be cast at runtime. 
        /// </summary>
        /// <param name="size">The length of the GradVector to create</param>
        /// <returns>New GradVector of all constants</returns>
        public static GradVector Zeros(int size)
        {
            return new GradVector(new GradVal[size]);
        }

        /// <summary>
        /// Extends the GradVector with multiple extra GradVals.
        /// Maintains input space continuity.
        /// </summary>
        /// <param name="extension">The enumerable to append to the end of this GradVector</param>
        /// <returns>A new GradVector with the extension</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="extension"/> has a different input space from this GradVector</exception>
        public GradVector Extend(IEnumerable<GradVal> extension)
        {
            var arr = extension.ToArray();
            GradVal[] ret = new GradVal[Length + arr.Length];
            for (int i = 0; i < _length; i++)
                ret[i] = Values[i]; // We already know these are the same input length since they would've already been checked
            for (int i = _length; i < ret.Length; i++)
            {
                AssertInputLengthContinuity(arr[i - _length]);
                ret[i] = arr[i - _length];
            }

            // If this vector is all constants, the input size comes from the extension (InputSize also doesn't allow mixed sizes so that's chill)
            return new GradVector(ret, ret.Length, _inputLength != 0 ? _inputLength : GradMath.InputSize(arr));
        }

        /// <summary>
        /// Appends a new value to the end of the GradVector.
        /// Maintains input space continuity.
        /// </summary>
        /// <param name="value">Value to append</param>
        /// <returns>New Vector with the value appended</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="value"/> has a different input space from this GradVector</exception>
        public GradVector Append(GradVal value)
        {
            AssertInputLengthContinuity(value);
            GradVal[] ret = new GradVal[_length + 1];
            for (int i = 0; i < _length; i++)
                ret[i] = Values[i];
            ret[_length] = value;
            return new GradVector(ret, ret.Length, _inputLength != 0 ? _inputLength : value.Length);
        }

        /*
            Concatenates two Vectors together. Keeps Derivatives.
        */
        /// <summary>
        /// Concatenates two GradVectors together, keeping gradients and Hessians.
        /// Maintains input space continuity.
        /// </summary>
        /// <param name="vec1">Left GradVector to concatenate</param>
        /// <param name="vec2">Right GradVector to concatenate</param>
        /// <returns>A new GradVector that contains copies of all elements of <paramref name="vec1"/> on the left and <paramref name="vec2"/> on the right</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="vec1"/> has a different input space from <paramref name="vec2"/></exception>
        public static GradVector Concat(GradVector vec1, GradVector vec2)
        {
            vec1.AssertInputLengthContinuity(vec2);
            GradVal[] ret = new GradVal[vec1._length + vec2._length];
            for (int i = 0; i < vec1._length; i++)
                ret[i] = vec1.Values[i];
            for (int i = vec1._length; i < ret.Length; i++)
                ret[i] = vec2.Values[i - vec1._length];

            return new GradVector(ret, ret.Length, vec1._inputLength != 0 ? vec1._inputLength : vec2._inputLength);
        }

        /// <summary>
        /// Apply a function that transforms GradVal across the given GradVector.
        /// </summary>
        /// <param name="func">Function that takes in and outputs a GradVal to apply a singular transformation.</param>
        /// <returns>A new GradVector containing the transformed GradVals</returns>
        public GradVector Apply(Func<GradVal, GradVal> func) =>
            new GradVector(Values.Select(func));

        // Operator definitions for Vector-Vector operations. Does not include * or / since they aren't well defined generally, they are kept as seperate functions.
        /// <summary>
        /// Element-wise addition of two GradVectors.
        /// Preserves accurate gradients and Hessians.
        /// </summary>
        /// <param name="a">Left GradVector to add</param>
        /// <param name="b">Right GradVector to add</param>
        /// <returns>New GradVector with the values, gradients, and Hessians resulting from element-wise addition</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the two GradVectors are not the same length</exception>
        /// <exception cref="ArgumentException">Thrown if the two GradVectors depend on different numbers of inputs</exception>
        public static GradVector operator +(GradVector a, GradVector b) => LinearCombine(a, 1, b, 1);
        /// <summary>
        /// Element-wise subtraction of two GradVectors.
        /// Preserves accurate gradients and Hessians.
        /// </summary>
        /// <param name="a">GradVector to get subtracted from</param>
        /// <param name="b">GradVector to subtract</param>
        /// <returns>New GradVector with the values, gradients, and Hessians resulting from element-wise subtraction</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the two GradVectors are not the same length</exception>
        /// <exception cref="ArgumentException">Thrown if the two GradVectors depend on different numbers of inputs</exception>
        public static GradVector operator -(GradVector a, GradVector b) => LinearCombine(a, 1, b, -1);

        // Operator definitions for scalar operations.
        /// <summary>
        /// Multiply every element of a GradVector by a constant.
        /// Preserves accurate gradients and Hessians.
        /// </summary>
        /// <param name="a">GradVector to be scalar multiplied</param>
        /// <param name="b">Constant to multiply by</param>
        /// <returns>New GradVector with the values, gradients, and Hessians resulting from scalar multiplication</returns>
        public static GradVector operator *(GradVector a, double b) => Scale(a, b);
        /// <summary>
        /// Multiply every element of a GradVector by a constant.
        /// Preserves accurate gradients and Hessians.
        /// </summary>
        /// <param name="a">Constant to multiply by</param>
        /// <param name="b">GradVector to be scalar multiplied</param>
        /// <returns>New GradVector with the values, gradients, and Hessians resulting from scalar multiplication</returns>
        public static GradVector operator *(double a, GradVector b) => Scale(b, a);

        /// <summary>
        /// Divide every element of a GradVector by a constant.
        /// Preserves accurate gradients and Hessians.
        /// </summary>
        /// <param name="a">GradVector to be scalar divided</param>
        /// <param name="b">Constant to divide by</param>
        /// <returns>New GradVector with the values, gradients, and Hessians resulting from scalar division</returns>
        public static GradVector operator /(GradVector a, double b) => Scale(a, 1.0/b);

        // Implement the enumerator functions
        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<GradVal> GetEnumerator()
        {
            return ((IEnumerable<GradVal>)Values).GetEnumerator(); ;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Assert that <paramref name="val"/> has a gradient that same length as GradVector input.
        /// If <paramref name="val"/> is a constant (thus a 0 gradient) then the check passes.
        /// </summary>
        /// <param name="val">GradVal to be added to the GradVector.</param>
        /// <exception cref="ArgumentException">Thrown if GradVal.Length (gradient size) is defined and not equal to _inputLength.</exception>
        private void AssertInputLengthContinuity(in GradVal val)
        {
            if (_inputLength != 0 && val.Length != 0 && _inputLength != val.Length) throw new ArgumentException($"Cannot add Cannot add GradVal with gradient size: {val.Length} to GradVector of input size: {_inputLength}");
        }

        /// <summary>
        /// Assert that the _inputLength of <paramref name="vec"/> is equivalent to this _inputLength.
        /// If <paramref name="vec"/> has _inputLength of 0 (composed of only constants) then the check passes.
        /// </summary>
        /// <param name="vec">GradVec to be checked against.</param>
        /// /// <exception cref="ArgumentException">Thrown if GradVal.Length (gradient size) is defined and not equal to _inputLength.</exception>
        private void AssertInputLengthContinuity(in GradVector vec)
        {
            if (_inputLength != 0 && vec._inputLength != 0 && _inputLength != vec._inputLength) throw new ArgumentException($"Cannot combine two GradVectors of different input sizes: {_inputLength} and {vec._inputLength}");
        }

        /*
            Creates a new Vector containing the linear combination of the two given vectors.
        */
        private static GradVector LinearCombine(GradVector a, double aCoef, GradVector b, double bCoef)
        {
            if (a._length != b._length) throw new IndexOutOfRangeException($"Vector of length {a._length} cannot be combined with vector of length {b._length}");
            a.AssertInputLengthContinuity(b);

            // Make a whole new vector so we don't mess up the grad graph
            GradVal[] ret = new GradVal[a._length];
            for (int i = 0; i < ret.Length; i++)
                ret[i] = a.Values[i] * aCoef + b.Values[i] * bCoef;
            return new GradVector(ret, ret.Length, a._inputLength != 0 ? a._inputLength : b._inputLength);
        }

        /*
            Creates a new Vector containing the values of the given vector scaled by the given coefficient.
        */
        private static GradVector Scale(GradVector a, double coef)
        {
            // We create a copy of each value so we don't risk changing the gradient graph with scaling
            GradVal[] ret = new GradVal[a._length];
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = a.Values[i] * coef;
            }
            return new GradVector(ret, ret.Length, a._inputLength);
        }

        /// <summary>
        /// A static factory to create a new GradVector of only constants.
        /// Constants mean that they have a 0 gradient.
        /// Input size of zero gradients is determined at runtime and thus not needed.
        /// </summary>
        /// <param name="values">A list of doubles that depend on no other values in the space.</param>
        /// <returns>A new GradVector of constants (0 gradient)</returns>
        public static GradVector Constants(params double[] values)
        {
            GradVal[] vals = new GradVal[values.Length];
            for (int i = 0; i < vals.Length; i++) vals[i] = new GradVal(values[i]);
            return new GradVector(vals, vals.Length, 0);
        }

        /// <summary>
        /// A static factory to create a new GradVector of only constants.
        /// Constants mean that they have a 0 gradient.
        /// Input size of zero gradients is determined at runtime and thus not needed.
        /// </summary>
        /// <param name="values">A list of doubles that depend on no other values in the space.</param>
        /// <returns>A new GradVector of constants (0 gradient)</returns>
        public static GradVector Constants(IEnumerable<double> values) =>
            Constants(values.ToArray());
    }

    /// <summary>
    /// Static math functions for GradVals and GradVectors, such as Dot and Sin, that carry the gradient and Hessian through.
    /// Also includes Chain for defining your own functions and InputSize for checking input dimensions.
    /// </summary>
    public static class GradMath
    {
        /// <summary>
        /// Calculates the dod product between two Vectors
        /// </summary>
        /// <param name="a">First Vector</param>
        /// <param name="b">Second Vector</param>
        /// <returns>a*b where * is the dot product</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the two vectors are not of the same size</exception>
        public static GradVal Dot(GradVector a, GradVector b)
        {
            if (a.Length != b.Length) throw new IndexOutOfRangeException($"Vector of length {a.Length} cannot be combined with vector of length {b.Length}");

            GradVal ret = default;

            for (int i = 0; i < a.Length; i++)
                ret += a[i] * b[i];

            return ret;
        }

        /// <summary>
        /// Get the size of the input of the given values.
        /// i.e. The number of values each non-constant element depends on.
        /// </summary>
        /// <param name="values">The GradVals to get input size from</param>
        /// <returns>0 if all values are constant, otherwise input size</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> contains mixed input dimensions</exception>
        public static int InputSize(IEnumerable<GradVal> values)
        {
            // Pull out all the distinct sizes to make sure they haven't mixed input dims
            var sizes = values.Select(val => val.Length).Where(size => size != 0).Distinct().ToArray();
            if (sizes.Length > 1) throw new ArgumentException($"Elements depend on different numbers of inputs: {string.Join(", ", sizes)}");
            return sizes.Length == 0 ? 0 : sizes[0];
        }

        /// <summary>
        /// Calculate the result of a scalar function including the value, gradient, and Hessian.
        /// Creates a new GradVal, so the passed GradVal stays intact and accurate.
        /// </summary>
        /// <remarks>This is a static wrapper for the member method GradVal.Chain() which performs the same function.</remarks>
        /// <param name="val">The value to pass through the function.</param>
        /// <param name="f">The result of passing <paramref name="val"/> through the function.</param>
        /// <param name="df">The result of passing <paramref name="val"/> through the first derivative of <paramref name="f"/></param>
        /// <param name="d2f">The result of passing <paramref name="val"/> through the second derivative of <paramref name="f"/></param>
        /// <returns>A new GradVal with the value, gradient, and Hessian fully calculated.</returns>
        public static GradVal Chain(GradVal val, double f, double df, double d2f) =>
            val.Chain(f, df, d2f);

        /// <summary>
        /// Calculate Sin in Radians.
        /// Automatically tracks gradient and Hessian.
        /// </summary>
        /// <param name="val">Value to Sin</param>
        /// <returns>A new GradVal containing the Sined value</returns>
        public static GradVal Sin(GradVal val) =>
            val.Sin();

        /// <summary>
        /// Calculate the Sin in Radians across the given Vector.
        /// Automatically tracking gradient and Hessian.
        /// Clones the values to a new GradVector and GradVals.
        /// </summary>
        /// <param name="vec">The Vector to be used to calculate the new GradVals.</param>
        /// <returns>A new GradVector containing new GradVals that contain the resulting Sin and associated derivs.</returns>
        public static GradVector Sin(GradVector vec) =>
            vec.Apply(Sin);
    }
}