def transform_fc(i,stride):
    res = []
    for j in range(stride):
        mean = []
        var = []
        resid = []
        fitted = []
        for k in range(len(i)):
            mean.append(i[k][0][j])
            var.append(i[k][1][j])
            resid.append(i[k][2])
            fitted.append(i[k][3])
        res.append([mean,var,resid,fitted])
    return res