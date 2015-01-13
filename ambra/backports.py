import numpy as np
import scipy.sparse as sp

def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)

try:
    from sklearn.utils.validation import check_array
except ImportError:
    def _assert_all_finite(X):
        """Like assert_all_finite, but only for ndarray."""
        X = np.asanyarray(X)
        # First try an O(n) time, O(1) space solution for the common case that
        # everything is finite; fall back to O(n) space np.isfinite to prevent
        # false positives from overflow in sum method.
        if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
                and not np.isfinite(X).all()):
            raise ValueError("Input contains NaN, infinity"
                            " or a value too large for %r." % X.dtype)

    def _ensure_sparse_format(spmatrix, accept_sparse, dtype, order, copy,
                            force_all_finite):
        """Convert a sparse matrix to a given format.

        Checks the sparse format of spmatrix and converts if necessary.

        Parameters
        ----------
        spmatrix : scipy sparse matrix
            Input to validate and convert.

        accept_sparse : string, list of string or None (default=None)
            String[s] representing allowed sparse matrix formats ('csc',
            'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
            matrix input will raise an error.  If the input is sparse but not in
            the allowed format, it will be converted to the first listed format.

        dtype : string, type or None (default=none)
            Data type of result. If None, the dtype of the input is preserved.

        order : 'F', 'C' or None (default=None)
            Whether an array will be forced to be fortran or c-style.

        copy : boolean (default=False)
            Whether a forced copy will be triggered. If copy=False, a copy might
            be triggered by a conversion.

        force_all_finite : boolean (default=True)
            Whether to raise an error on np.inf and np.nan in X.

        Returns
        -------
        spmatrix_converted : scipy sparse matrix.
            Matrix that is ensured to have an allowed type.
        """
        if accept_sparse is None:
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required. Use X.toarray() to '
                            'convert to a dense numpy array.')
        sparse_type = spmatrix.format
        if dtype is None:
            dtype = spmatrix.dtype
        if sparse_type in accept_sparse:
            # correct type
            if dtype == spmatrix.dtype:
                # correct dtype
                if copy:
                    spmatrix = spmatrix.copy()
            else:
                # convert dtype
                spmatrix = spmatrix.astype(dtype)
        else:
            # create new
            spmatrix = spmatrix.asformat(accept_sparse[0]).astype(dtype)
        if force_all_finite:
            if not hasattr(spmatrix, "data"):
                warnings.warn("Can't check %s sparse matrix for nan or inf."
                            % spmatrix.format)
            else:
                _assert_all_finite(spmatrix.data)
        if hasattr(spmatrix, "data"):
            spmatrix.data = np.array(spmatrix.data, copy=False, order=order)
        return spmatrix

    def check_array(array, accept_sparse=None, dtype=None, order=None, copy=False,
                    force_all_finite=True, ensure_2d=True, allow_nd=False):
        """Input validation on an array, list, sparse matrix or similar.

        By default, the input is converted to an at least 2nd numpy array.

        Parameters
        ----------
        array : object
            Input object to check / convert.

        accept_sparse : string, list of string or None (default=None)
            String[s] representing allowed sparse matrix formats, such as 'csc',
            'csr', etc.  None means that sparse matrix input will raise an error.
            If the input is sparse but not in the allowed format, it will be
            converted to the first listed format.

        dtype : string, type or None (default=none)
            Data type of result. If None, the dtype of the input is preserved.

        order : 'F', 'C' or None (default=None)
            Whether an array will be forced to be fortran or c-style.

        copy : boolean (default=False)
            Whether a forced copy will be triggered. If copy=False, a copy might
            be triggered by a conversion.

        force_all_finite : boolean (default=True)
            Whether to raise an error on np.inf and np.nan in X.

        ensure_2d : boolean (default=True)
            Whether to make X at least 2d.

        allow_nd : boolean (default=False)
            Whether to allow X.ndim > 2.

        Returns
        -------
        X_converted : object
            The converted and validated X.
        """
        if isinstance(accept_sparse, str):
            accept_sparse = [accept_sparse]

        if sp.issparse(array):
            array = _ensure_sparse_format(array, accept_sparse, dtype, order,
                                        copy, force_all_finite)
        else:
            if ensure_2d:
                array = np.atleast_2d(array)
            array = np.array(array, dtype=dtype, order=order, copy=copy)
            if not allow_nd and array.ndim >= 3:
                raise ValueError("Found array with dim %d. Expected <= 2" %
                                array.ndim)
            if force_all_finite:
                _assert_all_finite(array)

        return array


try:
    from sklearn.utils import indexable
except ImportError:
    def check_consistent_length(*arrays):
        """Check that all arrays have consistent first dimensions.

        Checks whether all objects in arrays have the same shape or length.

        Parameters
        ----------
        arrays : list or tuple of input objects.
            Objects that will be checked for consistent length.
        """

        uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
        if len(uniques) > 1:
            raise ValueError("Found arrays with inconsistent numbers of samples: %s"
                            % str(uniques))

    def indexable(*iterables):
        """Make arrays indexable for cross-validation.

        Checks consistent length, passes through None, and ensures that everything
        can be indexed by converting sparse matrices to csr and converting
        non-interable objects to arrays.

        Parameters
        ----------
        iterables : lists, dataframes, arrays, sparse matrices
            List of objects to ensure sliceability.
        """
        result = []
        for X in iterables:
            if sp.issparse(X):
                result.append(X.tocsr())
            elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
                result.append(X)
            elif X is None:
                result.append(X)
            else:
                result.append(np.array(X))
        check_consistent_length(*result)
        return result
