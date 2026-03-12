import json

with open("results/evaluation_results.json", "r", encoding="utf-8") as f:
    z = json.load(f)

with open("stanza/results/evaluation_results.json", "r", encoding="utf-8") as f:
    s = json.load(f)

indices = ["raw", "lemma", "hybrid"]
groups = ["all", "entity-heavy", "long-morph", "cultural", "other"]
ks = [3, 5, 10]
metric_names = ["Recall", "Precision", "MRR", "Hit"]

def get_metrics(data, group, k):
    """Get metrics for a group at a given k."""
    kkey = f"k={k}"
    if group == "all":
        return data.get(kkey, {})
    else:
        clf = data.get("by_clf_label", {}).get(group, {})
        return clf.get(kkey, {})

def get_count(data, group):
    """Get query count for a group."""
    if group == "all":
        k1 = data.get("k=1", {})
        return k1.get("num_queries", "?")
    else:
        clf = data.get("by_clf_label", {}).get(group, {})
        k1 = clf.get("k=1", {})
        return k1.get("num_queries", "?")

print("=" * 110)
print("ZEMBEREK vs STANZA - TUM METRIKLER (k=3, 5, 10)")
print("=" * 110)

for idx in indices:
    print(f"\n{'#'*110}")
    print(f"  INDEX: {idx.upper()}")
    print(f"{'#'*110}")

    zidx = z.get(idx, {})
    sidx = s.get(idx, {})

    for g in groups:
        zcount = get_count(zidx, g)
        scount = get_count(sidx, g)
        label = "TUM SORGULAR" if g == "all" else g.upper()

        print(f"\n  [{label}] (Zemberek: {zcount} sorgu, Stanza: {scount} sorgu)")
        header = f"  {'k':>4} | {'Metrik':>10} | {'Zemberek':>10} | {'Stanza':>10} | {'Fark':>10} | Kazanan"
        print(header)
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+--------")

        for k in ks:
            zm = get_metrics(zidx, g, k)
            sm = get_metrics(sidx, g, k)
            for m in metric_names:
                mkey = f"{m}@{k}"
                zval = zm.get(mkey)
                sval = sm.get(mkey)
                if zval is not None and sval is not None:
                    diff = sval - zval
                    if diff > 0.001:
                        winner = "Stanza"
                    elif diff < -0.001:
                        winner = "Zemberek"
                    else:
                        winner = "Esit"
                    print(f"  k={k:>2} | {m:>10} | {zval:>10.4f} | {sval:>10.4f} | {diff:>+10.4f} | {winner}")
            if k < 10:
                print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+--------")
