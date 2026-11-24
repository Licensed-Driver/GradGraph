
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace GradGraph
{

    public readonly struct GradVal
    {

        // The derivative of it's vector with respect to the initial value
        private readonly Vector<double> _gradient;
        // The second derivative of it's vector with respect to the initial value
        private readonly Matrix<double> _hessian;
        // Private constructor to copy over all values from another GradVal into a new one
        private GradVal(double value, Vector<double> derivative, Matrix<double> secondDeriv) => (Value, _gradient, _hessian) = (value, derivative, secondDeriv);
        
        // The forward value of the initial number, so what our result will be if we never use the derivative
        public double Value { get; }

        /*
            Get the derivative vector.
        */

        public Vector<double> GetGradient() => _gradient?.Clone();

        /*
            Expose the Hessian
        */
        public Matrix<double> GetHessian() => _hessian?.Clone();

        /*
            Expose the Length of the derivative array.
        */
        public int Length => _gradient?.Count ?? 0;

        /// <summary>
        /// Create a new GradVal that represents a constant with no gradient
        /// </summary>
        /// <param name="value">The constant</param>
        public GradVal(double value) => (Value, _gradient, _hessian) = (value, default, default);

        /// <summary>
        /// Creates a new GradVal cloned from the given GradVal.
        /// </summary>
        /// <param name="value">GradVal whose value, gradient, and Hessian are cloned into the new GradVal.</param>
        public GradVal(GradVal value) => this = value;

        /*
            Initialize the initial value and create the deriv, which always starts as 1
        */
        public GradVal(IEnumerable<double> vals, int position)
        {
            var arr = vals.ToArray();
            if (position >= arr.Length) throw new IndexOutOfRangeException($"Position {position} is out of range for {arr.Length} values");
            Value = arr[position];
            _gradient = Vector<double>.Build.Dense(arr.Length, i => i == position ? 1.0 : 0.0);   // All start at 0
            _hessian = Matrix<double>.Build.Dense(arr.Length, arr.Length);
        }

        /// <summary>
        /// Creates a new GradVal.
        /// The value is selected as <paramref name="vals"/>[<paramref name="position"/>].
        /// The gradient and Hessian are created from <paramref name="vals"/>.
        /// </summary>
        /// <param name="vals">The vector this value is from (i.e. the values it depends on)</param>
        /// <param name="position">The position of this value in <paramref name="vals"/></param>
        public GradVal(double[] vals, int position)
        {
            if (position >= vals.Length) throw new IndexOutOfRangeException($"Position {position} is out of range for {vals.Length} values");
            Value = vals[position];
            _gradient = Vector<double>.Build.Dense(vals.Length, i => i == position ? 1.0 : 0.0);   // All start at 0
            _hessian = Matrix<double>.Build.Dense(vals.Length, vals.Length);
        }

        /*
            Creates a new GradVal of given size with given value at given position in the gradient.
        */
        public GradVal(double value, int position, int size)
        {
            if (position >= size) throw new IndexOutOfRangeException($"Position {position} is out of range for {size} values");
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

        /*
            Sine with preserved Gradient.
        */
        public GradVal Sin() => Chain(Math.Sin(Value), Math.Cos(Value), -Math.Sin(Value));

        /*
            Cosine with preseved Gradient.
        */
        public GradVal Cos() => Chain(Math.Cos(Value), -Math.Sin(Value), -Math.Cos(Value));

        // All operations involving scalars
        public static GradVal operator -(GradVal a) => new GradVal(-a.Value, Scale(a._gradient, -1), Scale(a._hessian, -1));
        public static implicit operator GradVal(double a) => new GradVal(a);

        public static GradVal operator +(GradVal a, double b) => new GradVal(a.Value+b, a._gradient, a._hessian);
        public static GradVal operator +(double a, GradVal b) => new GradVal(a + b.Value, b._gradient, b._hessian);

        public static GradVal operator -(GradVal a, double b) => new GradVal(a.Value-b, a._gradient, a._hessian);
        public static GradVal operator -(double a, GradVal b) => new GradVal(a - b.Value, Scale(b._gradient, -1), Scale(b._hessian, -1));

        public static GradVal operator *(GradVal a, double b) => new GradVal(a.Value * b, Scale(a._gradient, b), Scale(a._hessian, b));
        public static GradVal operator *(double a, GradVal b) => new GradVal(a * b.Value, Scale(b._gradient, a), Scale(b._hessian, a));

        public static GradVal operator /(GradVal a, double b) => new GradVal(a.Value / b, Scale(a._gradient, 1 / b), Scale(a._hessian, 1 / b));
        public static GradVal operator /(double a, GradVal b) => b.Chain(a / b.Value, -a / (b.Value * b.Value), 2 * a / Math.Pow(b.Value, 3));

        public static bool operator >(GradVal a, double b) => a.Value > b;
        public static bool operator >(double a, GradVal b) => a > b.Value;

        public static bool operator <(GradVal a, double b) => a.Value < b;
        public static bool operator <(double a, GradVal b) => a < b.Value;

        public static bool operator ==(GradVal a, double b) => a.Value == b;
        public static bool operator ==(double a, GradVal b) => a == b.Value;

        public static bool operator !=(GradVal a, double b) => a.Value != b;
        public static bool operator !=(double a, GradVal b) => a != b.Value;

        public static bool operator <=(GradVal a, double b) => a.Value <= b;
        public static bool operator <=(double a, GradVal b) => a <= b.Value;

        public static bool operator >=(GradVal a, double b) => a.Value >= b;
        public static bool operator >=(double a, GradVal b) => a >= b.Value;

        // All operations involving two GradVals
        public static GradVal operator +(GradVal a, GradVal b) => new GradVal(a.Value + b.Value, Combine(a._gradient, 1, b._gradient, 1), Combine(a._hessian, 1, b._hessian, 1));

        public static GradVal operator -(GradVal a, GradVal b) => new GradVal(a.Value - b.Value, Combine(a._gradient, 1, b._gradient, -1), Combine(a._hessian, 1, b._hessian, -1));

        public static GradVal operator *(GradVal a, GradVal b) => new GradVal(a.Value * b.Value, Combine(a._gradient, b.Value, b._gradient, a.Value), Combine(Combine(a._hessian, b.Value, b._hessian, a.Value), 1, Combine(OuterProduct(a._gradient, b._gradient), 1, OuterProduct(b._gradient, a._gradient), 1), 1));

        public static GradVal operator /(GradVal a, GradVal b) => a * (1/b);    // All steps copy so we chillin

        public static bool operator >(GradVal a, GradVal b) => a.Value > b.Value;

        public static bool operator <(GradVal a, GradVal b) => a.Value < b.Value;

        public static bool operator ==(GradVal a, GradVal b) => a.Value == b.Value;

        public static bool operator !=(GradVal a, GradVal b) => a.Value != b.Value;

        public static bool operator <=(GradVal a, GradVal b) => a.Value <= b.Value;

        public static bool operator >=(GradVal a, GradVal b) => a.Value >= b.Value;

        /// <summary>
        /// Evaluate if given object is equivalent to this GradVal in all aspects.
        /// Is not equivalent to == as this compares value and derivatives.
        /// </summary>
        /// <param name="obj">The object to compare against.</param>
        /// <returns>True if obj is a GradVal and has the same value, gradient, and hessian. False otherwise.</returns>
        public override bool Equals(object obj)
        {
            // Obviously they aren't equal if it isn't even a gradval
            if (obj is not GradVal other)
                return false;
            // Only true equal if everything is the same
            return this.Value.Equals(other.Value);
        }

        /// <summary>
        /// Get the hash for this GradVal.
        /// Hash preserves equivalence in all fields,
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

    public readonly struct GradVector : IEnumerable<GradVal>
    {
        private readonly GradVal[] Values;
        private readonly int _length;
        private readonly int _inputLength;
        public readonly int Length {get => _length;}

        /*
            Creates a new Vector with the given values.
        */
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
            Values = values.ToArray();
            _length = Values.Length;
            _inputLength = GradMath.InputSize(values);
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

        /*
            Creates a new Vector that creates a COPY of the values passed, not a reference.
        */
        public GradVector(params GradVal[] values)
        {
            Values = (GradVal[])values.Clone();
            _length = values.Length;
            _inputLength = GradMath.InputSize(values);
        }

        /*
            Creates a copy of the given Vector by copying the GradVal[] and thus safe for use in the GradGraph
        */
        public GradVector(GradVector toCopy)
        {
            Values = (GradVal[])toCopy.Values.Clone();
            _length = Values.Length;
            _inputLength = GradMath.InputSize(toCopy);
        }
        
        // Allow array indexing for reading but no setting
        public GradVal this[int index]
        {
            get => Values[index];
        }

        /*
            Build the Jacobian Matrix from the derivatives of the vector elements.
        */
        public Matrix<double> GetJacobian()
        {
            if (_inputLength == 0) throw new InvalidOperationException("The Jacobian is undefined for non-differentiable expressions (vector of constants)");
            List<Vector<double>> rows = new List<Vector<double>>();
            // Since we treat null derivatives as all 0s we don't throw before we get here, so we gotta make sure the matrix won't throw em
            for (int i = 0; i < _length; i++)
                rows.Add(Values[i].GetGradient() ?? Vector<double>.Build.Dense(_inputLength, 0.0));
            return Matrix<double>.Build.DenseOfRowVectors(rows);
        }

        /*
            Creates a new Vector of all zeroes.
        */
        public static GradVector Zeros(int size)
        {
            return new GradVector(new GradVal[size]);
        }

        /*
            Extends the Vector with the enumerable. Keeps Derivatives.
        */
        public GradVector Extend(IEnumerable<GradVal> extension)
        {
            
            var arr = extension.ToArray();
            GradVector ret = Zeros(Length + arr.Length);
            for (int i = 0; i < _length; i++)
                ret.Values[i] = Values[i];
            for (int i = _length; i < ret._length; i++)
                ret.Values[i] = arr[i - _length];

            return ret;
        }

        /// <summary>
        /// Appends a new value to the end of the Vector
        /// </summary>
        /// <param name="value">Value to append</param>
        /// <returns>New Vector with the value appended</returns>
        public GradVector Append(GradVal value)
        {
            AssertInputLengthContinuity(ref value);
            GradVector ret = Zeros(_length + 1);
            for (int i = 0; i < _length; i++)
                ret.Values[i] = Values[i];
            ret.Values[_length] = value;
            return ret;
        }

        /*
            Concatenates two Vectors together. Keeps Derivatives.
        */
        public static GradVector Concat(GradVector vec1, GradVector vec2)
        {
            vec1.AssertInputLengthContinuity(ref vec2);
            GradVector ret = Zeros(vec1.Length + vec2.Length);
            for (int i = 0; i < vec1._length; i++)
                ret.Values[i] = vec1.Values[i];
            for (int i = vec1._length; i < ret._length; i++)
                ret.Values[i] = vec2.Values[i - vec1._length];

            return ret;
        }

        /// <summary>
        /// Apply a function that transforms GradVal across the given GradVector.
        /// </summary>
        /// <param name="func">Function that takes in and outputs a GradVal to apply a singular transformation.</param>
        /// <param name="vec">The GradVector to apply the  transformation to.</param>
        /// <returns>A new GradVector containing the transformed GradVals</returns>
        public GradVector Apply(Func<GradVal, GradVal> func) =>
            new GradVector(Values.Select(func));

        // Operator definitions for Vector-Vector operations. Does not include * or / since they aren't well defined generally, they are kept as seperate functions.
        public static GradVector operator +(GradVector a, GradVector b) => LinearCombine(a, 1, b, 1);
        public static GradVector operator -(GradVector a, GradVector b) => LinearCombine(a, 1, b, -1);

        // Operator definitions for scalar operations.
        public static GradVector operator *(GradVector a, double b) => Scale(a, b);
        public static GradVector operator *(double a, GradVector b) => Scale(b, a);

        public static GradVector operator /(GradVector a, double b) => Scale(a, 1.0/b);

        // Implement the enumerator functions
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
        private void AssertInputLengthContinuity(ref GradVal val)
        {
            if (val.Length != 0 && _inputLength != val.Length) throw new ArgumentException($"Cannot add Cannot add GradVal with gradient size: {val.Length} to GradVector of input size: {_inputLength}");
        }

        /// <summary>
        /// Assert that the _inputLength of <paramref name="vec"/> is equivalent to this _inputLength.
        /// If <paramref name="vec"/> has _inputLength of 0 (composed of only constants) then the check passes.
        /// </summary>
        /// <param name="vec">GradVec to be checked against.</param>
        /// /// <exception cref="ArgumentException">Thrown if GradVal.Length (gradient size) is defined and not equal to _inputLength.</exception>
        private void AssertInputLengthContinuity(ref GradVector vec)
        {
            if (vec._inputLength != 0 && _inputLength != vec._inputLength) throw new ArgumentException($"Cannot combine two GradVectors of different input sizes: {_inputLength} and {vec._inputLength}");
        }

        /*
            Creates a new Vector containing the linear combination of the two given vectors.
        */
        private static GradVector LinearCombine(GradVector a, double aCoef, GradVector b, double bCoef)
        {
            if (a._length != b._length) throw new IndexOutOfRangeException($"Vector of length {a._length} cannot be combined with vector of length {b._length}");

            // Make a whole new vector so we don't mess up the grad graph
            var ret = Zeros(a._length);
            for (int i = 0; i < ret._length; i++)
                ret.Values[i] = a.Values[i] * aCoef + b.Values[i] * bCoef;
            return ret;
        }

        /*
            Creates a new Vector containing the values of the given vector scaled by the given coefficient.
        */
        private static GradVector Scale(GradVector a, double coef)
        {
            // We create a copy of each value so we don't risk changing the gradient graph with scaling
            var ret = Zeros(a._length);
            for (int i = 0; i < ret._length; i++)
            {
                ret.Values[i] = a.Values[i] * coef;
            }
            return ret;
        }
    }

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
        /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> contains mixed input dimensions<exception/>
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