// Scoring extracted dots against ground truth (PLAN.md §2.6). Truth and found dots are matched one to one by
// minimum total distance, among pairs closer than a tolerance. Greedy nearest-first matching can mis-pair inside
// dense clusters, which are exactly the cases this project is about, so the assignment is optimal (Hungarian
// algorithm) within each connected group of candidate pairs; groups are small, so the O(k³) cost is local.

const GRID_CELL_TOLS = 1; // spatial grid cell size, in tolerances

/**
 * Optimal assignment for a square cost matrix (Kuhn–Munkres with potentials, O(n³)).
 *
 * @param {number[][]} cost  n × n
 * @returns {Int32Array}  column assigned to each row
 */
export function hungarian(cost) {
    const n = cost.length;
    // 1-based arrays as in the standard formulation; index 0 is the virtual start column
    const u = new Float64Array(n + 1);
    const v = new Float64Array(n + 1);
    const p = new Int32Array(n + 1); // p[j]: row matched to column j
    const way = new Int32Array(n + 1);
    for (let i = 1; i <= n; i++) {
        p[0] = i;
        let j0 = 0;
        const minv = new Float64Array(n + 1).fill(Infinity);
        const used = new Uint8Array(n + 1);
        do {
            used[j0] = 1;
            const i0 = p[j0];
            let delta = Infinity;
            let j1 = 0;
            for (let j = 1; j <= n; j++) {
                if (used[j])
                    continue;
                const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (let j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] !== 0);
        do {
            const j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    const rowCol = new Int32Array(n);
    for (let j = 1; j <= n; j++)
        rowCol[p[j] - 1] = j - 1;
    return rowCol;
}

/**
 * Match found dots to true dots.
 *
 * @param {{x: number, y: number}[]} truth
 * @param {{x: number, y: number}[]} found
 * @param {number} tol  px; pairs farther apart are never matched (PLAN.md uses r/2)
 * @returns {{pairs: {truth: number, found: number, d: number}[], missed: number[], extra: number[],
 *            precision: number, recall: number, countError: number, rmse: number}}
 */
export function matchDots(truth, found, tol) {
    // candidate pairs via a grid over the found dots
    const cell = GRID_CELL_TOLS * tol;
    const grid = new Map();
    found.forEach((f, j) => {
        const key = `${Math.floor(f.x / cell)},${Math.floor(f.y / cell)}`;
        if (!grid.has(key))
            grid.set(key, []);
        grid.get(key).push(j);
    });
    const edges = truth.map(t => {
        const gx = Math.floor(t.x / cell);
        const gy = Math.floor(t.y / cell);
        const out = [];
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                for (const j of grid.get(`${gx + dx},${gy + dy}`) ?? []) {
                    const d = Math.hypot(t.x - found[j].x, t.y - found[j].y);
                    if (d <= tol)
                        out.push({ j, d });
                }
            }
        }
        return out;
    });

    // connected groups of truth and found dots linked by candidate pairs (union–find over truth ∪ found)
    const parent = Int32Array.from({ length: truth.length + found.length }, (_, i) => i);
    const root = i => {
        while (parent[i] !== i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    edges.forEach((es, i) => {
        for (const { j } of es)
            parent[root(i)] = root(truth.length + j);
    });
    const groups = new Map();
    edges.forEach((es, i) => {
        if (!es.length)
            return;
        const r = root(i);
        if (!groups.has(r))
            groups.set(r, { t: [], f: new Set() });
        groups.get(r).t.push(i);
        for (const { j } of es)
            groups.get(r).f.add(j);
    });

    const pairs = [];
    for (const { t, f } of groups.values()) {
        const fs = [...f];
        // square matrix: real pairs at their distance, a dummy partner at tol for each side (leaving both of a pair
        // unmatched costs 2·tol, more than any allowed pair), and a prohibitive cost for pairs beyond tol
        const n = t.length + fs.length;
        const big = 1e6 * tol;
        const cost = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let a = 0; a < n; a++) {
            for (let b = 0; b < n; b++) {
                if (a < t.length && b < fs.length) {
                    const e = edges[t[a]].find(x => x.j === fs[b]);
                    cost[a][b] = e ? e.d : big;
                }
                else if (a < t.length || b < fs.length) {
                    cost[a][b] = tol;
                }
            }
        }
        const assign = hungarian(cost);
        for (let a = 0; a < t.length; a++) {
            const b = assign[a];
            if (b < fs.length && cost[a][b] <= tol)
                pairs.push({ truth: t[a], found: fs[b], d: cost[a][b] });
        }
    }

    const matchedT = new Set(pairs.map(p => p.truth));
    const matchedF = new Set(pairs.map(p => p.found));
    const missed = truth.map((_, i) => i).filter(i => !matchedT.has(i));
    const extra = found.map((_, j) => j).filter(j => !matchedF.has(j));
    const rmse = pairs.length ? Math.sqrt(pairs.reduce((s, p) => s + p.d * p.d, 0) / pairs.length) : NaN;
    return {
        pairs, missed, extra,
        precision: found.length ? pairs.length / found.length : 1,
        recall: truth.length ? pairs.length / truth.length : 1,
        countError: found.length - truth.length,
        rmse,
    };
}
