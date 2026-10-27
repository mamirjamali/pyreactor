
from __future__ import annotations
from collections.abc import Iterable, MutableSequence
from typing import Any

from pyreactor.helper.iteration import iter_coordinates
from pyreactor.helper.validation import (
    validate_indices,
    validate_metadata,
    validate_shape,
    validate_strides,
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

    def __getitem__(self, indices: int | tuple[int, ...]) -> Any | "Tensor":
        """Return a scalar or shared-storage view for the given indices."""
        if isinstance(indices, int) and not isinstance(indices, bool):
            indices = (indices,)
        elif not isinstance(indices, tuple):
            raise TypeError(
                "Indices must be an integer or a tuple of integers."
            )

        storage_index = self._storage_index(
            indices,
            allow_partial=True,
        )

        if len(indices) == len(self.shape):
            return self.storage[storage_index]

        remaining_shape = self.shape[len(indices):]
        remaining_strides = self.strides[len(indices):]

        return self._create_view(
            shape=remaining_shape,
            strides=remaining_strides,
            offset=storage_index,
        )

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

    # def _normalize_slice(
    #     self,
    #     item: slice,
    #     dim: int,
    # ) -> tuple[int, int, int]:
    #     ...