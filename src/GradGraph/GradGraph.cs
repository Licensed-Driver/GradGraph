
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace GradGraph
{

    public readonly struct GradVal
    {
        // The forward value of the initial number, so what our result will be if we never use the derivative
        private readonly double Value;

        // The derivative of it's vector with respect to the initial value
        private readonly Vector<double> Gradient;
        // The second derivative of it's vector with respect to the initial value
        private readonly Matrix<double> Hessian;

        private GradVal(double value, Vector<double> derivative, Matrix<double> secondDeriv) => (Value, Gradient, Hessian) = (value, derivative, secondDeriv); 

        /*
            Initialize the initial value and create the deriv, which always starts as 1
        */
        public GradVal(IEnumerable<double> vals, int position)
        {
            Value = vals.ElementAt(position);
            Gradient = Vector<double>.Build.Dense(vals.Count(), i => i == position ? 1.0 : 0.0);   // All start at 0
            Hessian = Matrix<double>.Build.Dense(vals.Count(), vals.Count());
        }

        /*
            Creates a new GradVal of given size.
            Value defaults to 0 if not provided.
        */
        public GradVal(int size, int? position = null, double value = 0)
        {
            Value = value;
            Gradient = Vector<double>.Build.Dense(size, i => i == (position ?? -1) ? 1.0 : 0);
            Hessian = Matrix<double>.Build.Dense(size, size);
        }

        /*
            Get the real part.
        */
        public double GetValue() => Value;

        /*
            Get the derivative vector.
        */

        public Vector<double> GetGradient() => Gradient.Clone();

        /*
            Expose the Hessian
        */
        public Matrix<double> GetHessian() => Hessian.Clone();

        /*
            Expose the Length of the derivative array.
        */
        public int Length() => Gradient.Count;

        /*
            Sine with preserved Gradient.
        */
        public GradVal Sin() => new GradVal(System.Math.Sin(Value), Gradient * System.Math.Cos(Value), Gradient.OuterProduct(Gradient) * -System.Math.Sin(Value) + Hessian * System.Math.Cos(Value));

        /*
            Cosine with preseved Gradient.
        */
        public GradVal Cos() => new GradVal(System.Math.Cos(Value), Gradient * -System.Math.Sin(Value), -System.Math.Cos(Value) * Gradient.OuterProduct(Gradient) + -System.Math.Sin(Value) * Hessian);

        // All operations involving scalars
        public static GradVal operator -(GradVal a) => new GradVal(-a.Value, -a.Gradient, -a.Hessian);

        public static GradVal operator +(GradVal a, double b) => new GradVal(a.Value+b, a.Gradient, a.Hessian);
        public static GradVal operator +(double a, GradVal b) => new GradVal(a + b.Value, b.Gradient, b.Hessian);

        public static GradVal operator -(GradVal a, double b) => new GradVal(a.Value-b, a.Gradient, a.Hessian);
        public static GradVal operator -(double a, GradVal b) => new GradVal(a - b.Value, -b.Gradient, -b.Hessian);

        public static GradVal operator *(GradVal a, double b) => new GradVal(a.Value * b, Scale(a.Gradient, b), Scale(a.Hessian, b));
        public static GradVal operator *(double a, GradVal b) => new GradVal(a * b.Value, Scale(b.Gradient, a), Scale(b.Hessian, a));

        public static GradVal operator /(GradVal a, double b) => new GradVal(a.Value / b, Scale(a.Gradient, 1 / b), Scale(a.Hessian, 1 / b));
        public static GradVal operator /(double a, GradVal b) => new GradVal(a / b.Value, Scale(b.Gradient, -a / (b.Value * b.Value)), Combine(b.Gradient.OuterProduct(b.Gradient), (2 * a / System.Math.Pow(b.Value, 3)), b.Hessian, - (a / System.Math.Pow(b.Value, 2))));

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
        public static GradVal operator +(GradVal a, GradVal b) => new GradVal(a.Value + b.Value, Combine(a.Gradient, 1, b.Gradient, 1), Combine(a.Hessian, 1, b.Hessian, 1));

        public static GradVal operator -(GradVal a, GradVal b) => new GradVal(a.Value - b.Value, Combine(a.Gradient, 1, b.Gradient, -1), Combine(a.Hessian, 1, b.Hessian, -1));

        public static GradVal operator *(GradVal a, GradVal b) => new GradVal(a.Value * b.Value, Combine(a.Gradient, b.Value, b.Gradient, a.Value), Combine(Combine(a.Hessian, b.Value, b.Hessian, a.Value), 1, Combine(OuterProduct(a.Gradient, b.Gradient), 1, OuterProduct(b.Gradient, a.Gradient), 1), 1));

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
            return Value == other.Value && Gradient.Equals(other.Gradient) && Hessian.Equals(other.Hessian);
        }

        /// <summary>
        /// Get the hash for this GradVal.
        /// Hash preserves equivalence in all fields,
        /// i.e. GradVal1.Equals(GradVal2) = True => GradVal1.GetHashCode() == GradVal2.GetHashCode() = True
        /// </summary>
        /// <returns>Hash for this GradVal.</returns>
        public override int GetHashCode() =>
            HashCode.Combine(Value.GetHashCode(), Gradient.GetHashCode(), Hessian.GetHashCode());

        /*
            Throws if vectors are differing lengths, returns null if both are defined and the same length, the opposing scaled vector if one is undefined.
        */
        private static Vector<double> AssertLength(Vector<double> vec1, double coef1, Vector<double> vec2, double coef2)
        {
            if (vec1 == null) return Scale(vec2, coef2);
            if (vec2 == null) return Scale(vec1, coef1);

            if (vec1.Count() != vec2.Count()) throw new IndexOutOfRangeException($"Vector of length {vec1.Count()} cannot be combined with vector of length {vec2.Count()}");

            return null;
        }

        /*
            Throws if matrices are differing shapes, returns null if both are defined and the same shape, the opposing scaled matrix if one is undefined.
        */
        private static Matrix<double> AssertShape(Matrix<double> mat1, double coef1, Matrix<double> mat2, double coef2)
        {
            if (mat1 == null) return Scale(mat2, coef2);
            if (mat2 == null) return Scale(mat1, coef1);

            if (mat1.ColumnCount != mat2.ColumnCount || mat1.RowCount != mat2.RowCount) throw new IndexOutOfRangeException($"Matrix of shape [{mat1.RowCount}x{mat1.ColumnCount}] is not able to be linearly combined with matrix of shape [{mat2.RowCount}x{mat2.ColumnCount}]");

            return null;
        }

        /*
            Linearly combines the two given derivative arrays using the given coefficients.
        */
        private static Vector<double> Combine(Vector<double> deriv1, double coef1, Vector<double> deriv2, double coef2)
            // Since new GradVal[n] doesn't create the Derivative reference
            // So we take it as the derivative being all 0
            => AssertLength(deriv1, coef1, deriv2, coef2) ?? coef1 * deriv1 + coef2 * deriv2;

        /*
            Linearly combines the two given hessians or other matrices with the given coefficients.
        */
        private static Matrix<double> Combine(Matrix<double> hess1, double coef1, Matrix<double> hess2, double coef2)
            => AssertShape(hess1, coef1, hess2, coef2) ?? coef1 * hess1 + coef2 * hess2;

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
            if (vec1 == null && vec2 == null) throw new NullReferenceException("Both vectors in outer product are null, I don't know what to do with this!");
            if (vec1 == null) return Matrix<double>.Build.Dense(vec2.Count(), vec2.Count());
            if (vec2 == null) return Matrix<double>.Build.Dense(vec1.Count(), vec1.Count());

            return vec1.OuterProduct(vec2);
        }
    }

    public readonly struct Vector : IEnumerable<double>
    {
        private readonly GradVal[] Values;
        private readonly int _length;
        public readonly int Length {get => _length;}

        /*
            Creates a new Vector of all zeroes.
        */
        public Vector(int size)
        {
            Values = new GradVal[size];
            _length = Values.Length;
        }

        /*
            Creates a new Vector with the given values.
        */
        public Vector(IEnumerable<double> values)
        {
            Values = new GradVal[values.Count()];
            for (int i = 0; i < Values.Length; i++)
            {
                Values[i] = new GradVal(values, i);
            }
            _length = Values.Length;
        }

        /*
            Creates a new Vector that creates a COPY of the values passed, not a reference.
        */
        public Vector(GradVal[] values)
        {
            Values = (GradVal[])values.Clone();
            _length = Values.Length;
        }

        /*
            Creates a copy of the given Vector by copying the GradVal[] and thus safe for use in the GradGraph
        */
        public Vector(Vector toCopy)
        {
            Values = (GradVal[])toCopy.Values.Clone();
            _length = Values.Length;
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
            // Since we treat null derivatives as all 0s we don't throw before we get here, so we gotta make sure the matrix won't throw em
            int tempLength = Values.Length;
            Vector<double>[] rows = Values.Select(val => val.GetGradient() ?? Vector<double>.Build.Dense(tempLength, 0.0)).ToArray();
            return Matrix<double>.Build.DenseOfRowVectors(rows);
        }

        /*
            Extends the Vector with the enumerable. Keeps Derivatives.
        */
        public Vector Extend(IEnumerable<GradVal> extension)
        {
            Vector ret = new Vector(Length + extension.Count());
            for (int i = 0; i < _length; i++)
                ret.Values[i] = Values[i];
            for (int i = _length; i < ret._length; i++)
                ret.Values[i] = extension.ElementAt(i - _length);

            return ret;
        }

        /*
            Concatenates two Vectors together. Keeps Derivatives.
        */
        public static Vector Concat(Vector vec1, Vector vec2)
        {
            Vector ret = new Vector(vec1.Length + vec2.Count());
            for (int i = 0; i < vec1._length; i++)
                ret.Values[i] = vec1.Values[i];
            for (int i = vec1._length; i < ret._length; i++)
                ret.Values[i] = vec2.Values[i - vec1._length];

            return ret;
        }

        // Operator definitions for Vector-Vector operations. Does not include * or / since they aren't well defined generally, they are kept as seperate functions.
        public static Vector operator +(Vector a, Vector b) => LinearCombine(a, 1, b, 1);
        public static Vector operator -(Vector a, Vector b) => LinearCombine(a, 1, b, -1);

        // Operator definitions for scalar operations.
        public static Vector operator *(Vector a, double b) => Scale(a, b);
        public static Vector operator *(double a, Vector b) => Scale(b, a);

        public static Vector operator /(Vector a, double b) => Scale(a, 1.0/b);

        // Implement the enumerator functions
        public IEnumerator<double> GetEnumerator()
        {
            double[] ret = new double[Values.Length];
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = Values[i].GetValue();
            }
            return ((IEnumerable<double>)ret).GetEnumerator(); ;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /*
            Creates a new Vector containing the linear combination of the two given vectors.
        */
        private static Vector LinearCombine(Vector a, double aCoef, Vector b, double bCoef)
        {
            if (a._length != b._length) throw new IndexOutOfRangeException($"Vector of length {a._length} cannot be combined with vector of length {b._length}");

            // Make a whole new vector so we don't mess up the grad graph
            var ret = new Vector(a._length);
            for (int i = 0; i < ret._length; i++)
                ret.Values[i] = a.Values[i] * aCoef + b.Values[i] * bCoef;
            return ret;
        }

        /*
            Creates a new Vector containing the values of the given vector scaled by the given coefficient.
        */
        private static Vector Scale(Vector a, double coef)
        {
            // We create a copy of each value so we don't risk changing the gradient graph with scaling
            var ret = new Vector(a._length);
            for (int i = 0; i < ret._length; i++)
            {
                ret.Values[i] = a.Values[i] * coef;
            }
            return ret;
        }
    }

    public static class GradMath
    {
        /*
            Generate the dot product between two vectors.
        */
        public static GradVal Dot(Vector a, Vector b)
        {
            if (a.Length != b.Length) throw new IndexOutOfRangeException($"Vector of length {a.Length} cannot be combined with vector of length {b.Length}");

            GradVal ret = new GradVal(a[0].Length());

            for (int i = 0; i < a.Length; i++)
                ret += a[i] * b[i];

            return ret;
        }
    }
}