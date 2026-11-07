import pytest

from pyreactor.tensor import Tensor
from pyreactor.helper.validation import *


# ---------------------------------------------------------------------------
# Indexing and views
# ---------------------------------------------------------------------------

def test_integer_index_returns_contiguous_row_view():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

    row = t[2]

    assert row.shape == (2,)
    assert row.strides == (1,)
    assert row.offset == 4
    assert row[0] == 5
    assert row[1] == 6
    assert row.storage is t.storage


def test_integer_index_returns_strided_row_view():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2), strides=(1, 3))

    row = t[2]

    assert row.shape == (2,)
    assert row.strides == (3,)
    assert row.offset == 2
    assert row[0] == 3
    assert row[1] == 6
    assert row.storage is t.storage


def test_tuple_partial_index_returns_view():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, 2]

    assert view.shape == (4,)
    assert view.strides == (1,)
    assert view.offset == 20
    assert view[0] == 20
    assert view[3] == 23
    assert view.storage is t.storage


def test_nested_views_share_storage():
    t = Tensor(list(range(24)), (2, 3, 4))

    sub_tensor = t[1]
    row = sub_tensor[2]

    assert row.shape == (4,)
    assert row.offset == 20
    assert row.storage is t.storage

    row[1] = 999

    assert t[1, 2, 1] == 999


def test_complete_index_returns_scalar():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    assert t[0, 0] == 1
    assert t[1, 2] == 6


def test_slice_index_returns_view():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1:2]

    assert view.shape == (1, 3, 4)
    assert view.strides == (12, 4, 1)
    assert view.offset == 12
    assert view[0, 0, 0] == 12
    assert view[0, 2, 3] == 23
    assert view.storage is t.storage


def test_slice_index_on_middle_axis():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[:, 1:3]

    assert view.shape == (2, 2, 4)
    assert view.strides == (12, 4, 1)
    assert view.offset == 4
    assert view[0, 0, 0] == 4
    assert view[1, 1, 3] == 23


def test_slice_index_on_last_axis():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[:, :, 1:3]

    assert view.shape == (2, 3, 2)
    assert view.strides == (12, 4, 1)
    assert view.offset == 1
    assert view[0, 0, 0] == 1
    assert view[1, 2, 1] == 22


def test_slice_index_with_step():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[:, :, ::2]

    assert view.shape == (2, 3, 2)
    assert view.strides == (12, 4, 2)
    assert view.offset == 0
    assert view[0, 0, 1] == 2
    assert view[1, 2, 1] == 22


def test_mixed_integer_and_slice_indexing():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, :, 1:3]

    assert view.shape == (3, 2)
    assert view.strides == (4, 1)
    assert view.offset == 13
    assert view[0, 0] == 13
    assert view[1, 1] == 18
    assert view[2, 1] == 22


def test_multiple_slices_with_steps():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, 0:3:2, 1:4:2]

    assert view.shape == (2, 2)
    assert view.strides == (8, 2)
    assert view.offset == 13

    assert view[0, 0] == 13
    assert view[0, 1] == 15
    assert view[1, 0] == 21
    assert view[1, 1] == 23


def test_partial_slice_preserves_unspecified_dimensions():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, 1:3]

    assert view.shape == (2, 4)
    assert view.strides == (4, 1)
    assert view.offset == 16
    assert view[0, 0] == 16
    assert view[1, 3] == 23


def test_empty_slice_returns_empty_view():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1:1]

    assert view.shape == (0, 3, 4)
    assert view.strides == (12, 4, 1)
    assert view.offset == 12
    assert view.storage is t.storage


def test_slice_view_shares_storage():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, :, 1:3]

    view[0, 0] = 999

    assert t[1, 0, 1] == 999
    assert view.storage is t.storage


def test_slice_with_default_start_and_stop():
    t = Tensor(list(range(6)), (6,))

    view = t[2:]

    assert view.shape == (4,)
    assert view.strides == (1,)
    assert view.offset == 2
    assert [view[i] for i in range(4)] == [2, 3, 4, 5]


def test_slice_with_non_unit_step_and_nonzero_start():
    t = Tensor(list(range(10)), (10,))

    view = t[1:9:3]

    assert view.shape == (3,)
    assert view.strides == (3,)
    assert view.offset == 1
    assert [view[i] for i in range(3)] == [1, 4, 7]


def test_integer_index_assignment_updates_original_storage():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

    row = t[2]
    row[0] = 100

    assert row[0] == 100
    assert t[2, 0] == 100
    assert t.storage[4] == 100
    assert row.storage is t.storage


def test_slice_assignment():
    t = Tensor(list(range(6)), (6,))

    t[1:4] = [10, 20, 30]

    assert t.storage == [0, 10, 20, 30, 4, 5]


def test_slice_assignment_with_step():
    t = Tensor(list(range(6)), (6,))

    t[::2] = [10, 20, 30]

    assert t.storage == [10, 1, 20, 3, 30, 5]


def test_slice_assignment_with_nonzero_start_and_step():
    t = Tensor(list(range(10)), (10,))

    t[1:9:3] = [100, 200, 300]

    assert t.storage == [0, 100, 2, 3, 200, 5, 6, 300, 8, 9]


def test_slice_assignment_updates_original_storage():
    t = Tensor(list(range(24)), (2, 3, 4))

    view = t[1, :, 1:3]
    view[:, :] = [10, 20, 30, 40, 50, 60]

    assert t[1, 0, 1] == 10
    assert t[1, 0, 2] == 20
    assert t[1, 1, 1] == 30
    assert t[1, 1, 2] == 40
    assert t[1, 2, 1] == 50
    assert t[1, 2, 2] == 60


def test_mixed_integer_and_slice_assignment():
    t = Tensor(list(range(24)), (2, 3, 4))

    t[1, :, 1:3] = [100, 200, 300, 400, 500, 600]

    assert t[1, 0, 1] == 100
    assert t[1, 0, 2] == 200
    assert t[1, 1, 1] == 300
    assert t[1, 1, 2] == 400
    assert t[1, 2, 1] == 500
    assert t[1, 2, 2] == 600


def test_slice_assignment_with_wrong_number_of_values():
    t = Tensor(list(range(6)), (6,))

    with pytest.raises(ValueError):
        t[1:4] = [10, 20]


def test_slice_assignment_with_too_many_values():
    t = Tensor(list(range(6)), (6,))

    with pytest.raises(ValueError):
        t[1:3] = [10, 20, 30]


def test_slice_assignment_with_scalar_value():
    t = Tensor(list(range(6)), (6,))

    t[1:3] = 99

    assert t.storage == [0, 99, 99, 3, 4, 5]


def test_negative_index_returns_last_element():
    t = Tensor([10, 20, 30, 40], (4,))
    assert t[-1] == 40


def test_negative_index_returns_second_last_element():
    t = Tensor([10, 20, 30, 40], (4,))
    assert t[-2] == 30


def test_negative_index_in_multidimensional_tensor():
    t = Tensor(list(range(24)), (2, 3, 4))
    assert t[-1, -1, -1] == 23


def test_negative_partial_index_returns_view():
    t = Tensor(list(range(24)), (2, 3, 4))
    view = t[-1, -1]

    assert view.shape == (4,)
    assert view.offset == 20
    assert view[0] == 20


def test_negative_index_assignment():
    t = Tensor([10, 20, 30, 40], (4,))
    t[-1] = 999

    assert t.storage == [10, 20, 30, 999]


def test_negative_index_out_of_bounds():
    t = Tensor([10, 20, 30, 40], (4,))

    with pytest.raises(IndexError):
        _ = t[-5]


def test_negative_index_on_empty_dimension():
    t = Tensor([], (0,))

    with pytest.raises(IndexError):
        _ = t[-1]


def test_reverse_slice():
    t = Tensor([10, 20, 30, 40, 50], (5,))

    view = t[::-1]

    assert view.shape == (5,)
    assert view.strides == (-1,)
    assert view.offset == 4

    assert [view[i] for i in range(5)] == [50, 40, 30, 20, 10]


def test_negative_step_slice():
    t = Tensor([10, 20, 30, 40, 50], (5,))

    view = t[4:1:-1]

    assert view.shape == (3,)
    assert view.strides == (-1,)
    assert view.offset == 4

    assert [view[i] for i in range(3)] == [50, 40, 30]


def test_negative_step_with_larger_step():
    t = Tensor(list(range(10)), (10,))

    view = t[::-2]

    assert view.shape == (5,)
    assert view.strides == (-2,)
    assert view.offset == 9

    assert [view[i] for i in range(5)] == [9, 7, 5, 3, 1]


def test_reverse_slice_updates_original_storage():
    t = Tensor([10, 20, 30, 40], (4,))

    view = t[::-1]
    view[0] = 999

    assert t[3] == 999


def test_partial_scalar_assignment():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

    t[1] = 99

    assert t.storage == [1, 2, 99, 99, 5, 6]


def test_partial_iterable_assignment():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

    t[1] = [10, 20]

    assert t.storage == [1, 2, 10, 20, 5, 6]
# ---------------------------------------------------------------------------
# Ellipsis
# ---------------------------------------------------------------------------

def test_ellipsis_at_end():
    t = Tensor(list(range(24)), (2, 3, 4))

    result = t[..., 0]

    assert result.shape == (2, 3)


def test_ellipsis_at_start():
    t = Tensor(list(range(24)), (2, 3, 4))

    result = t[0, ...]

    assert result.shape == (3, 4)


def test_ellipsis_in_middle():
    t = Tensor(list(range(24)), (2, 3, 4))

    result = t[1, ..., 2]

    assert result.shape == (3,)


def test_ellipsis_alone():
    t = Tensor(list(range(24)), (2, 3, 4))

    result = t[...]

    assert result.shape == (2, 3, 4)


def test_multiple_ellipsis_raises():
    t = Tensor(list(range(24)), (2, 3, 4))

    with pytest.raises(IndexError):
        t[..., ..., 0]
# ---------------------------------------------------------------------------
# Transposition
# ---------------------------------------------------------------------------

def test_transpose_swaps_two_dimensions():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    transposed = t.transpose(0, 1)

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed.offset == t.offset
    assert transposed.storage is t.storage

    assert transposed[0, 0] == 1
    assert transposed[0, 1] == 4
    assert transposed[1, 0] == 2
    assert transposed[1, 1] == 5
    assert transposed[2, 0] == 3
    assert transposed[2, 1] == 6


def test_transposed_assignment_updates_original():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    transposed = t.transpose(0, 1)
    transposed[0, 1] = 99

    assert t[1, 0] == 99
    assert t.storage[3] == 99


def test_T_property_reverses_dimensions():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    transposed = t.T

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed.storage is t.storage
    assert transposed[0, 1] == 4
    assert transposed[2, 1] == 6


def test_transpose_rejects_invalid_dimensions():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    with pytest.raises(IndexError):
        t.transpose(0, 2)

    with pytest.raises(IndexError):
        t.transpose(-1, 0)

    with pytest.raises(TypeError):
        t.transpose(0, True)


# ---------------------------------------------------------------------------
# Contiguous tensors
# ---------------------------------------------------------------------------

def test_contiguous_tensor_returns_itself():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    result = t.contiguous()

    assert result is t
    assert result.storage is t.storage


def test_contiguous_transition_from_transposed_tensor():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    transposed = t.T

    contiguous_t = transposed.contiguous()

    assert contiguous_t is not transposed
    assert contiguous_t.storage is not t.storage
    assert contiguous_t.shape == (3, 2)
    assert contiguous_t.strides == (2, 1)
    assert contiguous_t.storage == [1, 4, 2, 5, 3, 6]

    assert contiguous_t[0, 0] == 1
    assert contiguous_t[0, 1] == 4
    assert contiguous_t[1, 0] == 2
    assert contiguous_t[2, 1] == 6


def test_contiguous_copy_is_independent():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    contiguous_t = t.T.contiguous()

    contiguous_t[0, 0] = 100

    assert t[0, 0] == 1
    assert contiguous_t[0, 0] == 100


def test_contiguous_preserves_nonzero_offset_view():
    t = Tensor(list(range(12)), (3, 4))

    row = t[1]
    result = row.contiguous()

    # This row is already contiguous, even though its offset is nonzero.
    assert result is row
    assert result.offset == 4
    assert result.storage is t.storage
    assert [result[i] for i in range(4)] == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Contiguity and metadata
# ---------------------------------------------------------------------------

def test_is_contiguous_for_standard_row_major_layout():
    t = Tensor(list(range(24)), (2, 3, 4))

    assert t.strides == (12, 4, 1)
    assert t.is_contiguous()


def test_transposed_tensor_is_not_contiguous():
    t = Tensor(list(range(6)), (2, 3))

    assert not t.T.is_contiguous()


def test_invalid_shape_dimension_type():
    with pytest.raises(TypeError):
        Tensor([1, 2], (2, True))


def test_negative_shape_dimension_is_rejected():
    with pytest.raises(ValueError):
        Tensor([], (2, -1))


def test_invalid_stride_count_is_rejected():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3, 4], (2, 2), strides=(1,))


def test_negative_stride_is_rejected():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], (3,), strides=(-1,))


def test_invalid_stride_type_is_rejected():
    with pytest.raises(TypeError):
        Tensor([1, 2, 3], (3,), strides=(1.5,))


def test_invalid_offset_type_is_rejected():
    with pytest.raises(TypeError):
        Tensor([1, 2, 3], (3,), offset=True)


def test_negative_offset_is_rejected():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], (3,), offset=-1)


def test_offset_beyond_storage_is_rejected():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], (1,), offset=4)


def test_metadata_exceeding_storage_is_rejected():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], (2, 2))


# ---------------------------------------------------------------------------
# Index validation
# ---------------------------------------------------------------------------

def test_too_many_indices_are_rejected():
    t = Tensor([1, 2, 3, 4], (2, 2))

    with pytest.raises(IndexError):
        _ = t[0, 1, 0]


def test_out_of_bounds_indices_are_rejected():
    t = Tensor([1, 2, 3, 4], (2, 2))

    with pytest.raises(IndexError):
        _ = t[2, 0]

    with pytest.raises(IndexError):
        _ = t[0, -3]


def test_boolean_indices_are_rejected():
    t = Tensor([1, 2, 3], (3,))

    with pytest.raises(TypeError):
        _ = t[True]

    with pytest.raises(TypeError):
        t[False] = 10


def test_non_integer_indices_are_rejected():
    t = Tensor([1, 2, 3], (3,))

    with pytest.raises(TypeError):
        _ = t[1.5]


def test_non_tuple_non_integer_indices_are_rejected():
    t = Tensor([1, 2, 3], (3,))

    with pytest.raises(TypeError):
        _ = t[[1]]


def test_slice_accepts_negative_step():
    t = Tensor(list(range(5)), (5,))

    view = t[::-1]

    assert view.shape == (5,)
    assert view.strides == (-1,)
    assert view.offset == 4


def test_slice_rejects_zero_step():
    t = Tensor(list(range(5)), (5,))

    with pytest.raises(ValueError):
        _ = t[::0]


def test_slice_rejects_unsupported_index_type():
    t = Tensor(list(range(5)), (5,))

    with pytest.raises(TypeError):
        _ = t[1.5:4]


def test_slice_rejects_too_many_indices():
    t = Tensor(list(range(24)), (2, 3, 4))

    with pytest.raises(IndexError):
        _ = t[:, :, :, :]


def test_normalize_single_index():
    assert normalize_indices(1) == (1,)


def test_normalize_ellipsis():
    assert normalize_indices(Ellipsis) == (Ellipsis,)


def test_normalize_tuple():
    assert normalize_indices((1, slice(None))) == (1, slice(None))


def test_validate_multiple_ellipsis():
    with pytest.raises(IndexError):
        validate_indices_syntax((Ellipsis, Ellipsis), 3)


def test_validate_too_many_indices():
    with pytest.raises(IndexError):
        validate_indices_syntax((1, 2, 3, 4), 3)


def test_expand_ellipsis():
    result = expand_ellipsis((1, Ellipsis, 2), 4)
    assert result == (1, slice(None), slice(None), 2)


def test_expand_missing_dimensions():
    result = expand_missing_dimensions((1,), 3)
    assert result == (1, slice(None), slice(None))


def test_expand_missing_dimensions_no_change():
    result = expand_missing_dimensions((1, 2), 2)
    assert result == (1, 2)


def test_validate_multiple_ellipsis():
    with pytest.raises(IndexError):
        validate_indices_syntax((Ellipsis, Ellipsis), 3)

        
def test_multiple_ellipsis_raises():
    t = Tensor(list(range(24)), (2, 3, 4))

    with pytest.raises(IndexError):
        t[..., ..., 0]

# ---------------------------------------------------------------------------
# Scalar and empty tensors
# ---------------------------------------------------------------------------

def test_scalar_tensor():
    t = Tensor([42], ())

    assert t.shape == ()
    assert t.strides == ()
    assert t.is_contiguous()
    assert t[()] == 42


def test_scalar_tensor_assignment():
    t = Tensor([42], ())

    t[()] = 99

    assert t[()] == 99
    assert t.storage == [99]


def test_scalar_tensor_requires_storage():
    with pytest.raises(ValueError):
        Tensor([], ())


def test_empty_tensor_with_zero_dimension():
    t = Tensor([], (0, 3))

    assert t.shape == (0, 3)
    assert t.strides == (3, 1)
    assert t.is_contiguous()
    assert t.contiguous() is t


def test_empty_tensor_rejects_indexing():
    t = Tensor([], (0, 3))

    with pytest.raises(IndexError):
        _ = t[0]


def test_empty_tensor_with_offset_at_storage_end():
    t = Tensor([1, 2, 3], (0, 3), offset=3)

    assert t.shape == (0, 3)
    assert t.offset == 3


def test_empty_tensor_offset_beyond_storage_is_rejected():
    with pytest.raises(ValueError):
        Tensor([], (0, 3), offset=1)


def test_slice_on_empty_tensor():
    t = Tensor([], (0, 3))

    view = t[:]

    assert view.shape == (0, 3)
    assert view.strides == (3, 1)
    assert view.offset == 0
    assert view.storage is t.storage