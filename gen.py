#!/usr/bin/env python3
# gen_input_twolanes.py
from pathlib import Path
import argparse, random

def even_positions(L, n):
    return [(i * L) // n for i in range(n)] if n > 0 else []

def main():
    ap = argparse.ArgumentParser(description="Generate two-lane input for NaSch sim")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--vmax", type=int, required=True)
    ap.add_argument("--p-dec", type=float, required=True)
    ap.add_argument("--p-start", type=float, required=True)
    ap.add_argument("--steps", type=int, default=1000)

    ap.add_argument("--pos", choices=["even","random"], default="even",
                    help="initial positions layout per lane")
    ap.add_argument("--vel", choices=["zero","random"], default="zero",
                    help="initial velocity layout")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("input.txt"))
    args = ap.parse_args()

    if args.n <= 0 or args.L <= 0 or args.vmax < 0: raise SystemExit("n,L>0; vmax>=0")
    if not (0.0 <= args.p_dec <= 1.0 and 0.0 <= args.p_start <= 1.0): raise SystemExit("probs in [0,1]")

    rng = random.Random(args.seed)

    # Random lane assignment for each car (0 or 1)
    lanes = [rng.randint(0, 1) for _ in range(args.n)]
    n0 = sum(1 for x in lanes if x == 0)
    n1 = args.n - n0

    # Positions are unique within each lane
    if args.pos == "even":
        pos0 = even_positions(args.L, n0)
        pos1 = even_positions(args.L, n1)
    else:
        if n0 > args.L or n1 > args.L:
            raise SystemExit("per-lane: n_lane must be <= L for no initial collisions")
        pos0 = sorted(rng.sample(range(args.L), k=n0))
        pos1 = sorted(rng.sample(range(args.L), k=n1))

    # Velocities
    if args.vel == "zero":
        vel = [0] * args.n
    else:
        vel = [rng.randint(0, args.vmax) for _ in range(args.n)]

    # Stitch per-lane positions back to per-car order
    p_assign = [None] * args.n
    c0 = c1 = 0
    for i, ln in enumerate(lanes):
        if ln == 0:
            p_assign[i] = pos0[c0]; c0 += 1
        else:
            p_assign[i] = pos1[c1]; c1 += 1

    # Write single-file input
    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"{args.n}\n{args.L}\n{args.vmax}\n{args.p_dec}\n{args.p_start}\n{args.steps}\n{args.seed}\n\n")
        for ln, p, v in zip(lanes, p_assign, vel):
            f.write(f"{ln} {p} {v}\n")

    print(f"Wrote {args.out.resolve()}")

if __name__ == "__main__":
    main()
