"""MLIR sparse encoding generation (CSR MVP)."""

from typing import Optional
import numpy as np

from spardial import ir
from spardial.dialects import sparse_tensor as st


class SparseEncodingBuilder:
    """Builder class for constructing MLIR sparse tensor encodings.

    This class provides methods to create sparse tensor encoding attributes
    for different sparse formats. Currently supports CSR format only (MVP).
    """

    def __init__(self, context: ir.Context):
        """Initialize the encoding builder.

        Args:
            context: MLIR context in which to create encodings
        """
        self.context = context

    def build_csr(
        self,
        index_dtype: np.dtype = np.int64,
    ) -> st.EncodingAttr:
        """Generate sparse encoding for CSR format.

        CSR encoding:
        - Level 0 (rows): dense
        - Level 1 (cols): compressed
        - dimOrdering: identity (d0, d1) -> (d0, d1)

        Args:
            index_dtype: Index type (int32 or int64)

        Returns:
            SparseTensorEncodingAttr for CSR format
        """
        # Determine index/pointer bit widths from dtype
        if index_dtype == np.int32:
            pos_width = 32
            crd_width = 32
        else:  # int64
            pos_width = 64
            crd_width = 64

        with self.context:
            # CSR: dense for rows, compressed for columns
            levels = [
                st.EncodingAttr.build_level_type(st.LevelFormat.dense),
                st.EncodingAttr.build_level_type(st.LevelFormat.compressed),
            ]

            # Identity dimension ordering for CSR
            dim_to_lvl = ir.AffineMap.get_permutation([0, 1])
            lvl_to_dim = ir.AffineMap.get_permutation([0, 1])

            return st.EncodingAttr.get(levels, dim_to_lvl, lvl_to_dim, pos_width, crd_width)

    def build(
        self,
        format: str,
        index_dtype: np.dtype = np.int64,
    ) -> Optional[st.EncodingAttr]:
        """Generate encoding from format name.

        Args:
            format: "dense" or "csr"
            index_dtype: Index type

        Returns:
            SparseTensorEncodingAttr or None (for dense)
        """
        if format == "dense":
            return None

        if format == "csr":
            return self.build_csr(index_dtype)

        raise ValueError(f"Unsupported format: {format}. MVP supports 'csr' only.")
