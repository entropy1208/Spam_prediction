import itertools as it
def nemenyi_multitest(ranks):
    k = len(ranks)
    values = ranks.values()
    keys = ranks.keys()
    versus = list(it.combinations(range(k), 2))

    comparisons = [keys[vs[0]] + " vs " + keys[vs[1]] for vs in versus]
    z_values = [abs(values[vs[0]] - values[vs[1]]) for vs in versus]
    p_values = [2*(1-st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    m = int(k*(k-1)/2.)
    adj_p_values = [min(m*p_value,1) for p_value in p_values]
    
    return comparisons, z_values, p_values, adj_p_values
