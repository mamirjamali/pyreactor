
from collections.abc import Iterable, Sequence
from typing import TypeGuard


def validate_shape(shape: Iterable[int]) -> tuple[int, ...]:
    """Convert a shape to a tuple and validate its dimensions."""
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise TypeError("Shape must be an iterable of integers.") from exc

    for axis, dim in enumerate(dimensions):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(
                f"Shape dimension at axis {axis} must be an integer."
            )
        if dim < 0:
            raise ValueError(
                f"Shape dimension at axis {axis} cannot be negative."
            )

    return dimensions


def validate_strides(
    strides: Iterable[int],
    ndim: int,
) -> tuple[int, ...]:
    """Convert strides to a tuple and validate their dimensions and values."""
    try:
        stride_values = tuple(strides)
    except TypeError as exc:
        raise TypeError("Strides must be an iterable of integers.") from exc

    if len(stride_values) != ndim:
        raise ValueError(
            f"Expected {ndim} strides, but received {len(stride_values)}."
        )

    for axis, stride in enumerate(stride_values):
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError(
                f"Stride at axis {axis} must be an integer."
            )

    return stride_values


def validate_offset(offset: int, storage_length: int) -> int:
    """Validate a tensor's starting offset within its storage."""
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("Offset must be an integer.")

    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    if offset > storage_length:
        raise ValueError("Offset exceeds the storage boundary.")

    return offset


def validate_indices(
    indices: Sequence[int],
    shape: Sequence[int],
    *,
    allow_partial: bool = False,
) -> tuple[int, ...]:
    """Validate and normalize integer indices."""

    if allow_partial:
        if len(indices) > len(shape):
            raise IndexError(
                f"Too many indices: expected at most {len(shape)}, "
                f"but received {len(indices)}."
            )
    elif len(indices) != len(shape):
        raise IndexError(
            f"Expected {len(shape)} indices, but received {len(indices)}."
        )

    normalized_indices = []

    for axis, (index, dim) in enumerate(zip(indices, shape)):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(
                f"Index for axis {axis} must be an integer."
            )

        if index < 0:
            index += dim

        if index < 0 or index >= dim:
            raise IndexError(
                f"Index {index} is out of bounds for axis {axis} "
                f"with size {dim}."
            )

        normalized_indices.append(index)

    return tuple(normalized_indices)


def validate_metadata(
    storage_length: int,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
) -> None:
    """Ensure tensor metadata addresses only valid storage locations."""

    validate_offset(offset, storage_length)

    # Empty tensors contain no elements to address.
    if any(dim == 0 for dim in shape):
        return

    min_index = offset
    max_index = offset

    for dim, stride in zip(shape, strides):
        displacement = (dim - 1) * stride

        if displacement < 0:
            min_index += displacement
        else:
            max_index += displacement

    if min_index < 0 or max_index >= storage_length:
        raise ValueError(
            f"Tensor metadata exceeds storage bounds: "
            f"minimum index is {min_index}, "
            f"maximum index is {max_index}, "
            f"but storage length is {storage_length}."
        )


def validate_getitem_indices(
    indices: object,
    ndim: int,
) -> tuple[int | slice, ...]:

    if (
        isinstance(indices, (int, slice, type(Ellipsis)))
        and not isinstance(indices, bool)
    ):
        indices = (indices,)

    elif not isinstance(indices, tuple):
        raise TypeError(
            "Indices must be an integer, slice, or tuple."
        )

    indices = expand_ellipsis(indices, ndim)

    for axis, index in enumerate(indices):
        if isinstance(index, bool):
            raise TypeError(
                f"Index for axis {axis} cannot be a boolean."
            )

        if not isinstance(index, (int, slice)):
            raise TypeError(
                f"Index for axis {axis} must be an integer or slice."
            )

    return indices


def is_integer_indices(
    indices: tuple[int | slice, ...],
) -> TypeGuard[tuple[int, ...]]:
    """Check whether all indices are integers."""
    return all(
        isinstance(index, int) and not isinstance(index, bool)
        for index in indices
    )


def expand_ellipsis(
    indices: tuple,
    ndim: int,
) -> tuple:
    """Expand Ellipsis into the required number of full slices."""

    if Ellipsis not in indices:
        return indices

    missing_dims = ndim - (len(indices) - 1)

    expanded = []

    for index in indices:
        if index is Ellipsis:
            expanded.extend([slice(None)] * missing_dims)
        else:
            expanded.append(index)

    return tuple(expanded)


def normalize_indices(indices: object) -> tuple:
    """Normalize indexing input into a tuple."""

    if isinstance(indices, (int, slice, type(Ellipsis))):
        return (indices,)

    if not isinstance(indices, tuple):
        raise TypeError(
            "Indices must be an integer, slice, Ellipsis, or tuple."
        )

    return indices


def validate_indices_syntax(
    indices: tuple,
    ndim: int,
) -> None:
    """Validate indexing syntax without expanding dimensions."""

    ellipsis_count = sum(index is Ellipsis for index in indices)

    if ellipsis_count > 1:
        raise IndexError("An index can only have one ellipsis.")
    
    effective_count = sum(
        index is not None and index is not Ellipsis
        for index in indices
    )

    if effective_count > ndim:
        raise IndexError("Too many indices for tensor.")

    for axis, index in enumerate(indices):
        if index == None:
            continue

        if isinstance(index, bool):
            raise TypeError(
                f"Index for axis {axis} cannot be a boolean."
            )

        if not isinstance(index, (int, slice, type(Ellipsis))):
            raise TypeError(
                f"Index for axis {axis} must be an integer, slice, or Ellipsis."
            )


def expand_missing_dimensions(
    indices: tuple,
    ndim: int,
) -> tuple:
    """Fill missing dimensions with full slices."""
    consumed_dims = sum(index is not None for index in indices)
    missing_dims = ndim - consumed_dims

    if missing_dims < 0:
        raise IndexError("Too many indices for tensor.")

    return indices + (slice(None),) * missing_dims


def prepare_getitem_indices(
    indices: object,
    ndim: int,
) -> tuple[int | slice, ...]:
    """Normalize, validate, and expand indexing input."""

    indices = normalize_indices(indices)

    validate_indices_syntax(indices, ndim)

    indices = expand_ellipsis(indices, ndim)

    indices = expand_missing_dimensions(indices, ndim)

    return indices


def prepare_setitem_indices(
    indices: object,
    ndim: int,
) -> tuple[int | slice, ...]:
    """Normalize, validate, and expand assignment indices."""

    indices = normalize_indices(indices)

    validate_indices_syntax(indices, ndim)

    indices = expand_ellipsis(indices, ndim)
    
    indices = expand_missing_dimensions(indices, ndim)

    return indices