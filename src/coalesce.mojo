from std.sys.info import simd_width_of


comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.int64]()


def rows_differ(keys: I64Ptr, row: Int, columns: Int) -> Bool:
    var base = row * columns
    var previous = base - columns
    var column = 0
    if columns >= W * 2:
        while column + W <= columns:
            var current_values = keys.unsafe_load[width=W](base + column)
            var previous_values = keys.unsafe_load[width=W](previous + column)
            var mismatches = current_values.ne(previous_values)
            if mismatches.cast[DType.int64]().reduce_add() != 0:
                return True
            column += W
    while column < columns:
        if keys[unsafe_offset=base + column] != keys[
            unsafe_offset=previous + column
        ]:
            return True
        column += 1
    return False


@export("mwd_coalesce_indices")
def mwd_coalesce_indices(
    keys_addr: Int, rows: Int, columns: Int, indices_addr: Int
) abi("C") -> Int:
    if rows <= 0 or columns <= 0:
        return 0

    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var indices = I64Ptr(unsafe_from_address=indices_addr)
    indices[unsafe_offset=0] = 0
    var written = 1
    for row in range(1, rows):
        if rows_differ(keys, row, columns):
            indices[unsafe_offset=written] = Int64(row)
            written += 1

    return written
