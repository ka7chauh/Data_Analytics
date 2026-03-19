import numpy as np
import pandas as pd
from itertools import product, combinations
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def bad_segment_miner(
    df: pd.DataFrame,
    target_col: str,
    features: list[str] | None = None,
    n_quantiles: int = 10,                 # quantile bins per feature
    consider_pairs: bool = True,           # search 2D segments
    max_bins_per_feature: int = 2,         # allow 1 or 2 adjacent bins per feature per clause
    min_support: int = 300,                # minimum rows in a segment
    min_events: int = 15,                  # minimum events in a segment
    min_lift: float = 2.0,                 # segment event rate ≥ min_lift * baseline
    fdr_alpha: float = 0.05,               # FDR control (Benjamini–Hochberg)
    random_state: int = 42
):
    """
    Find 'bad' segments (high-risk) without trees using quantile binning + exhaustive subgroup discovery
    with significance testing and FDR control.
    """
    assert target_col in df.columns
    y = df[target_col].astype(int)
    baseline = y.mean()
    n = len(df)
    total_events = int(y.sum())

    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()

    # 1) Quantile bin numeric features
    rng = np.random.default_rng(random_state)
    binned = {}
    bin_edges = {}
    usable_feats = []
    for col in features:
        x = df[col]
        if x.isna().any():
            # Optionally impute or drop NaNs
            x = x.fillna(x.median())
        try:
            # ensure unique bin edges
            q = np.linspace(0, 1, n_quantiles + 1)
            edges = np.unique(np.quantile(x, q))
            # If too few distinct edges, reduce bins
            if len(edges) <= 2:  # constant or near-constant
                continue
            # cut with right-closed intervals
            bins = pd.cut(x, bins=edges, include_lowest=True, right=True)
            if bins.isna().all():
                continue
            binned[col] = bins
            bin_edges[col] = edges
            usable_feats.append(col)
        except Exception:
            # fallback: skip if binning fails
            continue

    if not usable_feats:
        raise ValueError("No usable numeric features after binning.")

    # Helper to generate adjacent-bin ranges for a single feature
    def adjacent_bin_ranges(categories, max_width=2):
        # categories are ordered intervals
        cats = list(pd.Categorical(categories).categories)
        L = len(cats)
        results = []
        for width in range(1, max_width + 1):
            for start in range(0, L - width + 1):
                end = start + width  # exclusive index
                results.append(cats[start:end])
        return results

    # 2) Build candidate clauses for 1D (and later combine for 2D)
    one_dim_clauses = []  # each clause is dict {feature: list_of_bins}
    for f in usable_feats:
        cat_series = binned[f]
        ranges = adjacent_bin_ranges(cat_series, max_width=max_bins_per_feature)
        for r in ranges:
            one_dim_clauses.append({f: r})

    # 3) Evaluate 1D candidates
    candidates = []
    total_pos = int(y.sum())
    total_neg = n - total_pos
    for clause in one_dim_clauses:
        f = next(iter(clause))
        allowed_bins = clause[f]
        mask = binned[f].isin(allowed_bins)
        support = int(mask.sum())
        if support < min_support:
            continue
        events = int(y[mask].sum())
        if events < min_events:
            continue

        rate = events / support
        if rate < min_lift * baseline:
            continue

        # 2x2 contingency for Fisher exact: [[a, b],[c, d]]
        # a = events in segment, b = non-events in segment
        # c = events outside, d = non-events outside
        a = events
        b = support - events
        c = total_pos - a
        d = total_neg - b
        # alternative = 'greater' tests whether odds in segment > outside
        _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')

        candidates.append({
            "features": [f],
            "bins": [allowed_bins],
            "support": support,
            "events": events,
            "rate": rate,
            "lift": rate / (baseline + 1e-12),
            "pval": pval
        })

    # 4) Optionally generate and evaluate 2D combinations (cartesian product of 1D clauses across different features)
    if consider_pairs:
        # pre-group 1D clauses by feature to avoid same-feature combinations
        by_feat = {}
        for c in one_dim_clauses:
            f = next(iter(c))
            by_feat.setdefault(f, []).append(c)

        pair_feats = list(combinations(usable_feats, 2))
        for f1, f2 in pair_feats:
            # to keep runtime in check, sample if too many combinations
            cands1 = by_feat.get(f1, [])
            cands2 = by_feat.get(f2, [])
            if len(cands1) == 0 or len(cands2) == 0:
                continue
            # Light subsampling heuristic
            max_per_feat = 40
            if len(cands1) > max_per_feat:
                cands1 = list(rng.choice(cands1, size=max_per_feat, replace=False))
            if len(cands2) > max_per_feat:
                cands2 = list(rng.choice(cands2, size=max_per_feat, replace=False))

            for c1, c2 in product(cands1, cands2):
                bins1 = c1[f1]; bins2 = c2[f2]
                mask = binned[f1].isin(bins1) & binned[f2].isin(bins2)
                support = int(mask.sum())
                if support < min_support:
                    continue
                events = int(y[mask].sum())
                if events < min_events:
                    continue
                rate = events / support
                if rate < min_lift * baseline:
                    continue

                a = events
                b = support - events
                c = total_pos - a
                d = total_neg - b
                _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')

                candidates.append({
                    "features": [f1, f2],
                    "bins": [bins1, bins2],
                    "support": support,
                    "events": events,
                    "rate": rate,
                    "lift": rate / (baseline + 1e-12),
                    "pval": pval
                })

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        print(f"No segments met min_support={min_support}, min_events={min_events}, lift≥{min_lift}.")
        return {
            "segments": pd.DataFrame(),
            "baseline": baseline,
            "n": n,
            "total_events": total_events
        }

    # 5) FDR correction
    rej, pval_adj, _, _ = multipletests(cand_df["pval"].values, alpha=fdr_alpha, method="fdr_bh")
    cand_df["pval_adj"] = pval_adj
    cand_df["significant"] = rej

    # 6) Nicely formatted rule strings
    def bins_to_str(feature, bins_list):
        # bins_list are Interval objects
        parts = []
        for inter in bins_list:
            parts.append(f"{feature} ∈ [{inter.left:.4g}, {inter.right:.4g}]")
        # Merge adjacent bins display: if multiple bins, join with OR
        return "(" + " OR ".join(parts) + ")"

    rule_strs = []
    for _, row in cand_df.iterrows():
        feats = row["features"]
        bins_lists = row["bins"]
        clauses = [bins_to_str(f, bl) for f, bl in zip(feats, bins_lists)]
        rule_strs.append(" AND ".join(clauses))
    cand_df["rule"] = rule_strs

    # 7) Sort: significant first, then by lift, then support
    cand_df = cand_df.sort_values(
        by=["significant", "lift", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    cols = ["rule", "features", "support", "events", "rate", "lift", "pval", "pval_adj", "significant"]
    return {
        "segments": cand_df[cols],
        "baseline": baseline,
        "n": n,
        "total_events": total_events
    }

