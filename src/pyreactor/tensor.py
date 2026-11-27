
from __future__ import annotations
from collections.abc import Iterable, MutableSequence
from typing import Any

from pyreactor.helper.iteration import iter_coordinates
from pyreactor.helper.validation import (
    validate_indices,
    validate_metadata,
    validate_shape,
    validate_strides,
    is_integer_indices,
    prepare_getitem_indices,
    prepare_setitem_indices,
)

import math

class Tensor:
    """A strided tensor view over a flat mutable storage sequence.

    Shape describes the logical dimensions. Strides and offset are measured
    in storage elements. Views share storage with the tensor they originate
    from.

    Integer indexing selects the first axis. Tuple indexing can select
    multiple axes, returning a view for partial indices and a scalar for
    complete indices.
    """

    def __init__(
        self,
        storage: MutableSequence[Any],
        shape: Iterable[int],
        strides: Iterable[int] | None = None,
        offset: int = 0,
    ) -> None:
        """Create a tensor over storage with the specified layout."""
        self.storage = storage

        # Validate shape before using it to calculate default strides.
        self.shape = validate_shape(shape)

        if strides is None:
            self.strides = self._compute_contiguous_strides(self.shape)
        else:
            self.strides = validate_strides(strides, len(self.shape))

        self.offset = offset

        validate_metadata(
            storage_length=len(self.storage),
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
        )

    @staticmethod
    def _compute_contiguous_strides(
        shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Calculate standard row-major strides for a shape."""
        strides = [0] * len(shape)
        stride = 1

        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = stride
            stride *= shape[axis]

        return tuple(strides)

    def _storage_index(
        self,
        indices: tuple[int, ...],
        *,
        allow_partial: bool = False,
    ) -> int:
        """Translate logical coordinates into a physical storage index."""
        indices = validate_indices(
            indices,
            self.shape,
            allow_partial=allow_partial,
        )

        storage_index = self.offset

        for axis, coordinate in enumerate(indices):
            storage_index += self.strides[axis] * coordinate

        return storage_index

    def _create_view(
        self,
        shape: Iterable[int],
        strides: Iterable[int],
        offset: int,
    ) -> "Tensor":
        """Create a validated view that shares this tensor's storage."""
        return Tensor(
            storage=self.storage,
            shape=shape,
            strides=strides,
            offset=offset,
        )
    
    @property
    def T(self) -> "Tensor":
        """Return a view with the axis order reversed."""
        return self._create_view(
            shape=self.shape[::-1],
            strides=self.strides[::-1],
            offset=self.offset,
        )

    def transpose(self, dim1: int, dim2: int) -> "Tensor":
        """Return a view with two axes swapped."""
        ndim = len(self.shape)

        for name, dim in (("dim1", dim1), ("dim2", dim2)):
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError(f"{name} must be an integer.")
            if not 0 <= dim < ndim:
                raise IndexError(
                    f"{name}={dim} is out of range for a {ndim}D tensor."
                )

        shape = list(self.shape)
        strides = list(self.strides)

        shape[dim1], shape[dim2] = shape[dim2], shape[dim1]
        strides[dim1], strides[dim2] = strides[dim2], strides[dim1]

        return self._create_view(
            shape=shape,
            strides=strides,
            offset=self.offset,
        )

    def reshape(self, *shape):
        # Normalize input
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)

        total_elements = math.prod(self.shape)

        # Validate dimensions
        if any(not isinstance(dim, int) or isinstance(dim, bool)
            for dim in new_shape):
            raise TypeError("Shape dimensions must be integers.")

        inferred_dims = [
            i for i, dim in enumerate(new_shape) if dim == -1
        ]

        if len(inferred_dims) > 1:
            raise ValueError("Only one dimension can be inferred.")

        if any(dim < -1 for dim in new_shape):
            raise ValueError("Shape dimensions must be non-negative.")

        # Infer -1
        if inferred_dims:
            axis = inferred_dims[0]
            known_product = math.prod(
                dim for dim in new_shape if dim != -1
            )

            if known_product == 0 or total_elements % known_product != 0:
                raise ValueError("Cannot infer the requested dimension.")

            inferred = total_elements // known_product
            new_shape = list(new_shape)
            new_shape[axis] = inferred
            new_shape = tuple(new_shape)

        elif math.prod(new_shape) != total_elements:
            raise ValueError(
                "New shape must have the same number of elements."
            )

        # Try to compute strides for a view
        old_shape = self.shape
        old_strides = self.strides
        new_strides = [0] * len(new_shape)

        view_dim = len(new_shape) - 1
        tensor_numel = 1
        view_numel = 1
        chunk_base_stride = old_strides[-1] if old_strides else 1
        can_view = True

        for tensor_dim in range(len(old_shape) - 1, -1, -1):
            tensor_numel *= old_shape[tensor_dim]

            chunk_boundary = (
                tensor_dim == 0
                or (
                    old_shape[tensor_dim - 1] != 1
                    and old_strides[tensor_dim - 1]
                    != tensor_numel * chunk_base_stride
                )
            )

            if chunk_boundary:
                while (
                    view_dim >= 0
                    and (
                        view_numel < tensor_numel
                        or new_shape[view_dim] == 1
                    )
                ):
                    new_strides[view_dim] = view_numel * chunk_base_stride
                    view_numel *= new_shape[view_dim]
                    view_dim -= 1

                if view_numel != tensor_numel:
                    can_view = False
                    break

                if tensor_dim > 0:
                    chunk_base_stride = old_strides[tensor_dim - 1]

                tensor_numel = 1
                view_numel = 1

        if view_dim != -1:
            can_view = False

        if can_view:
            return self._create_view(
                shape=new_shape,
                strides=tuple(new_strides),
                offset=self.offset,
            )

        # If a view is impossible, copy in logical order and reshape
        return self.contiguous().reshape(new_shape)
    
    def flatten(self):
        return self.reshape(math.prod(self.shape))

    def squeeze(self, dim=None):
        if dim is None:
            axes = [
                i for i, size in enumerate(self.shape)
                if size != 1
            ]
        else:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError("Dimension must be an integer.")

            if dim < 0:
                dim += len(self.shape)

            if dim < 0 or dim >= len(self.shape):
                raise IndexError("Dimension out of range.")

            if self.shape[dim] != 1:
                return self

            axes = [
                i for i in range(len(self.shape))
                if i != dim
            ]

        new_shape = tuple(self.shape[i] for i in axes)
        new_strides = tuple(self.strides[i] for i in axes)

        return self._create_view(
            shape=new_shape,
            strides=new_strides,
            offset=self.offset,
        )
    
    def unsqueeze(self, dim):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("Dimension must be an integer.")

        ndim = len(self.shape)

        # Normalize negative dimension
        if dim < 0:
            dim += ndim + 1

        if dim < 0 or dim > ndim:
            raise IndexError("Dimension out of range.")

        # Insert new dimension
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        # Calculate the new stride
        if dim < ndim:
            new_stride = self.strides[dim] * self.shape[dim]
        else:
            new_stride = 1

        new_strides = list(self.strides)
        new_strides.insert(dim, new_stride)

        return self._create_view(
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            offset=self.offset,
        )
    
    def permute(self, *dims):
        # Support tuple/list input
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])

        ndim = len(self.shape)

        # Validate number of dimensions
        if len(dims) != ndim:
            raise ValueError(
                "Number of dimensions must match tensor dimensions."
            )

        # Validate dimension types
        if any(not isinstance(dim, int) or isinstance(dim, bool)
            for dim in dims):
            raise TypeError("Dimensions must be integers.")

        # Normalize negative dimensions
        dims = tuple(
            dim + ndim if dim < 0 else dim
            for dim in dims
        )

        # Validate dimension range
        if any(dim < 0 or dim >= ndim for dim in dims):
            raise IndexError("Dimension out of range.")

        # Ensure every dimension appears exactly once
        if len(set(dims)) != ndim:
            raise ValueError(
                "Each dimension must appear exactly once."
            )

        # Reorder shape and strides
        new_shape = tuple(self.shape[dim] for dim in dims)
        new_strides = tuple(self.strides[dim] for dim in dims)

        return self._create_view(
            shape=new_shape,
            strides=new_strides,
            offset=self.offset,
        )
    
    def is_contiguous(self) -> bool:
        """Check whether strides match the standard row-major layout."""
        return self.strides == self._compute_contiguous_strides(self.shape)

    def contiguous(self) -> "Tensor":
        """Return a row-major tensor, copying data only when necessary."""
        if self.is_contiguous():
            return self

        new_storage = []

        for coordinates in iter_coordinates(self.shape):
            storage_index = self._storage_index(coordinates)
            new_storage.append(self.storage[storage_index])

        return Tensor(new_storage, self.shape)

    def _normalize_slice(
        self,
        item: slice,
        axis: int,
    ) -> tuple[int, int, int]:

        if not isinstance(item, slice):
            raise TypeError("Expected a slice object.")

        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError("Axis must be an integer.")

        if not 0 <= axis < len(self.shape):
            raise IndexError(
                f"Axis {axis} is out of range for a {len(self.shape)}D tensor."
            )

        dim = self.shape[axis]

        start, stop, step = item.indices(dim)

        return start, stop, step
    
    def _get_integer_indexed(
            self,
            indices: tuple[int | slice, ...],
        ) -> Any | "Tensor":
            """Handle indexing expressions containing only integers.

            Complete indexing returns a scalar, while partial indexing
            returns a view over the remaining dimensions.

            Args:
                indices: A tuple of integer indices.

            Returns:
                Any: The selected scalar value.
                Tensor: A view sharing the original storage.
            """
            if not is_integer_indices(indices):
                raise TypeError("Integer indexing requires integer indices.")
            
            storage_index = self._storage_index(
                indices,
                allow_partial=True,
            )

            # Return the underlying value when every dimension is indexed.
            if len(indices) == len(self.shape):
                return self.storage[storage_index]

            # Preserve the dimensions that were not explicitly indexed.
            remaining_shape = self.shape[len(indices):]
            remaining_strides = self.strides[len(indices):]

            return self._create_view(
                shape=remaining_shape,
                strides=remaining_strides,
                offset=storage_index,
            )

    def _get_sliced(
        self,
        indices: tuple[int | slice | None, ...],
    ) -> Any | "Tensor":

        new_shape = []
        new_strides = []
        new_offset = self.offset

        axis = 0

        for item in indices:

            if item is None:
                new_shape.append(1)

                # A new axis has no effect on storage.
                new_strides.append(0)

                continue

            old_stride = self.strides[axis]

            if isinstance(item, int):
                normalized_item = validate_indices(
                    (item,),
                    (self.shape[axis],),
                )[0]

                new_offset += normalized_item * old_stride

            elif isinstance(item, slice):
                start, stop, step = self._normalize_slice(
                    item,
                    axis,
                )

                new_offset += start * old_stride

                new_shape.append(len(range(start, stop, step)))
                new_strides.append(old_stride * step)

            axis += 1

        # Preserve unspecified trailing dimensions.
        for remaining_axis in range(axis, len(self.shape)):
            new_shape.append(self.shape[remaining_axis])
            new_strides.append(self.strides[remaining_axis])

        return self._create_view(
            shape=new_shape,
            strides=new_strides,
            offset=new_offset,
        )
    
    def _set_integer_indexed(
        self,
        indices: tuple[int, ...],
        value: Any,
    ) -> None:
        if len(indices) != len(self.shape):
            raise IndexError("Assignment requires complete indices.")

        storage_index = self._storage_index(indices)
        self.storage[storage_index] = value
        
    def _set_sliced(
        self,
        indices: tuple[int | slice, ...],
        value: Any,
    ) -> None:
        view = self._get_sliced(indices)
        expected_size = math.prod(view.shape)

        if isinstance(value, (int, float, bool)):
            values = [value] * expected_size
        else:
            values = list(value)

            if len(values) != expected_size:
                raise ValueError(
                    f"Expected {expected_size} values, got {len(values)}."
                )

        for i, coordinate in enumerate(iter_coordinates(view.shape)):
            storage_index = view._storage_index(coordinate)
            self.storage[storage_index] = values[i]
        
    def __getitem__(
        self,
        indices: int | slice | tuple[int | slice, ...],
    ) -> Any | "Tensor":
        """Retrieve an element or a view using integer or slice indexing.

        Integer-only indexing follows the existing indexing behavior.
        If at least one slice is provided, indexing is delegated to
        the slicing implementation.

        Args:
            indices: An integer, slice, or tuple containing integers and slices.

        Returns:
            Any: A scalar value when all dimensions are indexed by integers.
            Tensor: A view sharing the original storage otherwise.

        Raises:
            TypeError: If the indexing expression contains unsupported types.
            IndexError: If the number of indices or an integer index is invalid.
            ValueError: If the slice contains an unsupported step.
        """
        indices = prepare_getitem_indices(
            indices,
            len(self.shape),
        )

        # Dispatch to the appropriate indexing implementation.
        if any(isinstance(index, slice) for index in indices):
            return self._get_sliced(indices)

        return self._get_integer_indexed(indices)

    def __setitem__(
        self,
        indices: int | slice | tuple[int | slice | None, ...],
        value: Any,
    ) -> None:
        indices = prepare_setitem_indices(indices, len(self.shape))

        if any(
            isinstance(index, (slice, type(None)))
            for index in indices):
            self._set_sliced(indices, value)
            return

        self._set_integer_indexed(indices, value)