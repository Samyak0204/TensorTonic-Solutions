def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp,fp,fn=0,0,0

    classes=set(y_pred) | set(y_true)

    for c in classes:
        for yt,yp in zip(y_true,y_pred):
            if yt==c and yp==c:
                tp+=1
            elif yt!=c and yp==c:
                fp+=1
            elif yt==c and yp!=c:
                fn+=1

    d=2*tp+fp+fn

    return (2*tp/d) if d!=0 else 0.0