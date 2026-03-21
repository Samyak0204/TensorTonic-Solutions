def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    if not values:
        return []

    n=len(values)
    temp=values
    values=sorted(values)

    m=n//2
    if n%2==0:
        median=(values[m-1]+values[m])/2
        lower=values[:m]
        upper=values[m:]
    else:
        median=values[m]
        lower=values[:m]
        upper=values[m+1:]

    m=len(lower)
    if m==0:
        return [0]*n
    mid=m//2
    if m%2==0:
        q1=(lower[mid-1]+lower[mid])/2
    else:
        q1=lower[mid]

    m=len(upper)
    mid=m//2
    if m%2==0:
        q3=(upper[mid-1]+upper[mid])/2
    else:
        q3=upper[mid]

    iqr=q3-q1

    if iqr==0:
        return [0]*n

    return [(x-median)/iqr for x in temp]