
from __future__ import annotations
from collections.abc import Iterable, MutableSequence
from typing import Any

from pyreactor.helper.iteration import iter_coordinates
from pyreactor.helper.validation import (
    validate_indices,
    validate_metadata,
    validate_shape,
    validate_strides,
    validate_getitem_indices,
    is_integer_indices,
)


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
        validate_indices(
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

        if step < 0:
            raise ValueError("Negative slicing steps are not supported yet.")

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
        indices: tuple[int | slice, ...],
    ) -> Any | "Tensor":
        """Handle indexing expressions containing at least one slice.

        Integer indices remove their corresponding dimensions, while
        slices preserve dimensions with updated shapes and strides.
        Unspecified trailing dimensions remain unchanged.

        The resulting tensor shares storage with the original tensor.

        Args:
            indices: A tuple containing integer indices and slices.

        Returns:
            Tensor: A view sharing the original storage.
        """
        new_shape = []
        new_strides = []
        new_offset = self.offset

        for axis, item in enumerate(indices):
            old_stride = self.strides[axis]

            if isinstance(item, int):
                # Integer indexing selects one element along this axis
                # and removes the axis from the resulting view.
                validate_indices(
                    (item,),
                    (self.shape[axis],),
                )

                new_offset += item * old_stride

            else:
                # Slice indexing preserves the axis but changes
                # its logical size and physical storage stride.
                start, stop, step = self._normalize_slice(
                    item,
                    axis,
                )

                new_offset += start * old_stride

                new_shape.append(
                    len(range(start, stop, step))
                )

                new_strides.append(
                    old_stride * step
                )

        # Keep dimensions that were not explicitly indexed unchanged.
        for axis in range(len(indices), len(self.shape)):
            new_shape.append(self.shape[axis])
            new_strides.append(self.strides[axis])

        # Construct a view without copying the underlying storage.
        return self._create_view(
            shape=new_shape,
            strides=new_strides,
            offset=new_offset,
        )

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
        indices = validate_getitem_indices(
            indices,
            len(self.shape),
        )

        # Dispatch to the appropriate indexing implementation.
        if any(isinstance(index, slice) for index in indices):
            return self._get_sliced(indices)

        return self._get_integer_indexed(indices)

    def __setitem__(
        self,
        indices: int | tuple[int, ...],
        value: Any,
    ) -> None:
        """Assign a scalar using complete tensor coordinates.

        Partial-index assignment is not supported yet. Select a view first
        when you want to modify an element inside a sub-tensor.
        """
        if isinstance(indices, int) and not isinstance(indices, bool):
            indices = (indices,)
        elif not isinstance(indices, tuple):
            raise TypeError(
                "Indices must be an integer or a tuple of integers."
            )

        storage_index = self._storage_index(indices)
        self.storage[storage_index] = value
