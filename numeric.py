def full_round(x, digits=2):
    """
    Rounds a list of numbers to a given number of digits making sure that the
    total sum of the rounded numbers is the same as the sum of the original
    numbers.

    params
    ------
    x: list of numbers
    digits: number of digits to round to

    returns
    -------
    list of rounded numbers
    """
    out = []
    err = 0.0

    for a in x:
        a += err
        r = round(a, digits)
        err = a - r
        out.append(r)

    return out
