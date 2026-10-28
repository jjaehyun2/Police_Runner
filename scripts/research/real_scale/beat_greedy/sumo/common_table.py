"""Success rates and paired differences on ONE common denominator: the cases where every listed policy ran
without error. (sumo/paired.py drops errors per pair, while the raw "SUMO SR" lines count errors as failures,
so the two disagreed slightly; this prints both kinds of number from the same case set.)
Usage: python sumo/common_table.py <result.json> <base spec> <spec> [<spec> ...]"""
import json, math, sys

def main():
    d = json.load(open(sys.argv[1])); base, specs = sys.argv[2], sys.argv[2:]
    R = {s: {r[0]: r[1] for r in d[s]} for s in specs}
    ids = [i for i in R[base] if all(i in R[s] and not R[s][i].startswith("error") for s in specs)]
    n = len(ids); errs = {s: sum(v.startswith("error") for v in R[s].values()) for s in specs}
    print(f"common cases n={n} (errors per policy: {', '.join(f'{s[:6]}={e}' for s, e in errs.items())})")
    for s in specs:
        sr = sum(R[s][i] == "success" for i in ids) / n
        line = f"  {s[:34]:34s} SR {100*sr:5.1f}"
        if s != base:
            w = sum(R[s][i] == "success" and R[base][i] != "success" for i in ids)
            l = sum(R[base][i] == "success" and R[s][i] != "success" for i in ids)
            ci = 1.96 * math.sqrt(max(w + l - (w - l) ** 2 / n, 0)) / n
            line += f"  vs {base[:6]}: {100*(w-l)/n:+5.1f} ± {100*ci:4.1f}  (W/L {w}/{l})"
        print(line)

if __name__ == "__main__":
    main()
