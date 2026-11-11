from fractions import Fraction


def plan(n, fs, target=51200):
    if n < 4 or fs <= 0 or target <= 0:
        raise ValueError("Invalid grid")
    m = Fraction(n) * Fraction(str(target)) / Fraction(str(fs))
    if m.denominator != 1:
        raise ValueError(
            f"Noninteger output length {m}; choose an explicit duration policy"
        )
    return {"N": n, "M": int(m), "fs_in": fs, "fs_out": target, "duration": n / fs}
