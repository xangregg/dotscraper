// A stroke in the page's color around opaque dots (song-cents: a white stroke on a white page). On a dot alone it
// shows nowhere. Where a dot lies over another, it cuts an arc of the page's color through the one below, often
// through its center, and the opaque model, which has no stroke, pays for every pixel of the arc that a dot placed
// there covers: song-cents lost 27 of its 325 dots to it, most with half or more of their fill in view.
//
// The stroke is read before the fit, from the dots that are whole (a family's ink in one round piece: the dots on
// top, each bounded all round by its own stroke) and the cut ones beside them (a piece of ink smaller than a dot:
// what shows of a dot under others). Outward from a whole dot's edge, the page shows for some width before a cut
// piece's ink. Under a stroke that width is the same in every such direction, round every dot: the piece was cut by
// this dot's stroke. Without one, a dot is cut only by ink that touches it, a piece beside a whole dot merely lies
// near it, at whatever distance, and the gap between them widens along it. Whole dots beside whole dots tell
// nothing: packed side by side (a beeswarm), small dots have a ring of the page round each of them, about one width,
// and no stroke. The fit then ignores the pixels of the page's color in gaps as thin as the stroke between dot inks
// (layers.js buildMask, `strokes`).

import { backgroundColors } from './axes.js';
import { colorDistance } from './image.js';
import { components8 } from './layers.js';
import { median } from './stats.js';

const INK_MIN_COVERAGE = 0.5; // a pixel is a family's ink from this coverage (Estimate.D of an opaque family)
// a piece of one family's ink, by its area against a dot's (× π r²): a speck (an edge's pixels in another family's
// color, or a thin mark in the family's own: no ink here), a cut dot, a whole one (with a disc's box too), or more
// than one
const PIECE_MIN_AREA = 0.08;
const PIECE_MIN_DEPTH = 2; // px
const WHOLE_AREA = [0.85, 1.12];
const WHOLE_BOX_TOL = 1.5; // px, of each side of a whole dot's box against 2 r
const PIECE_SPECK = 1;
const PIECE_CUT = 2;
const PIECE_WHOLE = 3;
const PIECE_MANY = 4;
// a pixel is the page's when its color is within this share of the way from the page to the nearest family's color,
// and never farther than PAGE_MAX_TOL
const PAGE_TOL_SHARE = 0.25;
const PAGE_MAX_TOL = 40; // 0–255 units
// how far out from a whole dot's edge other ink counts as beside it, × r, within [GAP_MIN_REACH, GAP_MAX_REACH]
const GAP_REACH_RADII = 0.5;
const GAP_MIN_REACH = 4; // px
const GAP_MAX_REACH = 8; // px
const RADIAL_STEP = 0.5; // px
const FAR_EDGE = 0.5; // px: the rest of the far edge's pixel, taken as covered like the sample that reached it
// The dots have a stroke when the gaps from whole dots to cut pieces are STROKE_MIN_GAPS directions or more (about
// as many px of arc: an arc and a half) and STROKE_MIN_SHARE of them are within STROKE_WIDTH_TOL px of their median.
// The rest, under a stroke, are wider: gaps to pieces that other dots cut. Measured on the 145 local images:
// song-cents 760 gaps, 0.83 of one width; the two images without a stroke that have 40 gaps or more, 0.43 and 0.53
// (a beeswarm over a box, cantons).
const STROKE_MIN_GAPS = 40;
const STROKE_MIN_SHARE = 0.75;
const STROKE_WIDTH_TOL = 0.75; // px
// Gaps up to this much wider than the stroke between dot inks are the stroke's: its edges' pixels under half
// coverage, and a row or column crossing it aslant. Measured on song-cents (stroke 2.37 px): at 1.25, 1.75 and
// 2.75 px the fit finds 316, 316 and 315 of 325 dots, with 10, 10 and 13 that aren't there and position RMSE 0.13,
// 0.09 and 0.24 px.
const MASK_EXTRA_REACH = 1.75; // px
// … and the page shows in a gap where a pixel is within this share of the way from the page to the nearest family's
// color: more than PAGE_TOL_SHARE, which finds the stroke. In a blurred image a stroke 3 px wide between two dots
// never reaches the page's color (tsne: gray dots of 154, the stroke between them 204 to 238, 0.17 to 0.5 of the
// way), and with the strict test few of its gaps were masked: 8 gray dots under others missed.
const MASK_PAGE_SHARE = 0.5;

/**
 * The stroke in the page's color around the image's opaque dots, if they have one, and the pixels it shows in.
 *
 * @param {import('./image.js').RgbaImage} image  the image as analyzed
 * @param {import('./style.js').Estimate[]} estimates  every family of the image
 * @returns {{width: number, gaps: number, share: number, mask: Uint8Array|null}}  width: px, the stroke's, from the
 *   edge of the dot it is around to the ink beyond, their pixels by their coverage (NaN without gaps); gaps: the
 *   directions from whole dots in which a cut piece lies beyond the page; share: of those gaps, the ones of the
 *   stroke's width; mask: 1 on the pixels of the page's color in gaps no wider than the stroke between dot inks, or
 *   null when the dots have no such stroke
 */
export function pageStroke(image, estimates) {
    const { width, height, data } = image;
    const n = width * height;
    const fams = estimates.filter(e => e.style && e.style.alpha >= 1 && e.style.radius && e.D && !e.outlineMask);
    const none = { width: NaN, gaps: 0, share: 0, mask: null };
    if (!fams.length)
        return none;
    // the image's page, not a family's own ground: hair1's pink fills lie on their black outlines, 3.5 px wide and
    // one width all round, which are no stroke of the page's color
    const page = backgroundColors(image)[0];
    const tol = Math.min(PAGE_MAX_TOL, PAGE_TOL_SHARE * Math.min(...fams.map(e => colorDistance(e.style.color, page))));
    // per pixel: what piece of dot ink it is in (PIECE_…, 0 for none), and whether it is the page's color
    const piece = new Uint8Array(n);
    const wholes = [];
    fams.forEach(e => {
        const r = e.style.radius;
        const own = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            own[p] = !piece[p] && e.D[p] >= INK_MIN_COVERAGE ? 1 : 0;
        for (const c of components8(own, width, height)) {
            const area = c.pixels.length / (Math.PI * r * r);
            const round = Math.abs(c.x1 - c.x0 + 1 - 2 * r) <= WHOLE_BOX_TOL && Math.abs(c.y1 - c.y0 + 1 - 2 * r) <= WHOLE_BOX_TOL;
            // (a piece with no pixel PIECE_MIN_DEPTH px inside it is a line's or a dash's in the family's color)
            const deep = p => own[p - PIECE_MIN_DEPTH] && own[p + PIECE_MIN_DEPTH] && own[p - PIECE_MIN_DEPTH * width] && own[p + PIECE_MIN_DEPTH * width];
            const kind = area < PIECE_MIN_AREA || !c.pixels.some(deep) ? PIECE_SPECK
                : area < WHOLE_AREA[0] ? PIECE_CUT : area > WHOLE_AREA[1] ? PIECE_MANY : round ? PIECE_WHOLE : PIECE_CUT;
            let sx = 0;
            let sy = 0;
            for (const p of c.pixels) {
                piece[p] = kind;
                sx += p % width + 0.5;
                sy += Math.floor(p / width) + 0.5;
            }
            if (kind === PIECE_WHOLE)
                wholes.push({ x: sx / c.pixels.length, y: sy / c.pixels.length, r });
        }
    });
    // … and how much of it dot ink covers, 0–1: a gap's width is the page along it, its edges' pixels by their share
    const paper = new Uint8Array(n);
    const cover = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        paper[p] = (!piece[p] || piece[p] === PIECE_SPECK) && Math.hypot(data[4 * p] - page[0], data[4 * p + 1] - page[1], data[4 * p + 2] - page[2]) <= tol ? 1 : 0;
        for (const e of fams) {
            if (e.D[p] > cover[p])
                cover[p] = Math.min(1, e.D[p]);
        }
    }
    const gaps = [];
    for (const dot of wholes)
        cutGaps(piece, paper, cover, width, height, dot, gaps);
    if (!gaps.length)
        return none;
    const stroke = median(gaps);
    const share = gaps.filter(g => Math.abs(g - stroke) <= STROKE_WIDTH_TOL).length / gaps.length;
    const found = gaps.length >= STROKE_MIN_GAPS && share >= STROKE_MIN_SHARE;
    if (!found)
        return { width: stroke, gaps: gaps.length, share, mask: null };
    const faded = new Uint8Array(n);
    const fadedTol = MASK_PAGE_SHARE * Math.min(...fams.map(e => colorDistance(e.style.color, page)));
    for (let p = 0; p < n; p++)
        faded[p] = (!piece[p] || piece[p] === PIECE_SPECK) && Math.hypot(data[4 * p] - page[0], data[4 * p + 1] - page[1], data[4 * p + 2] - page[2]) <= fadedTol ? 1 : 0;
    return { width: stroke, gaps: gaps.length, share, mask: gapMask(piece, faded, width, height, stroke + MASK_EXTRA_REACH) };
}

// Round a whole dot, a direction for each px of its circle: where a cut piece's ink lies beyond the page within reach
// of the dot's edge, the width of the page between is added to `gaps`. O(r · reach) per dot.
function cutGaps(piece, paper, cover, width, height, dot, gaps) {
    const { x: cx, y: cy, r } = dot;
    const reach = Math.min(GAP_MAX_REACH, Math.max(GAP_MIN_REACH, GAP_REACH_RADII * r));
    const count = Math.ceil(2 * Math.PI * (r + reach));
    for (let i = 0; i < count; i++) {
        const a = 2 * Math.PI * i / count;
        // the pixel rho px out from the center, or -1 beyond the image
        const at = rho => {
            const x = Math.floor(cx + rho * Math.cos(a));
            const y = Math.floor(cy + rho * Math.sin(a));
            return x < 0 || y < 0 || x >= width || y >= height ? -1 : y * width + x;
        };
        // outward: the dot's own ink to its edge, then, within reach of the edge, the page and other dot ink
        let rho = r - 2;
        let open = 0; // px of the radius not covered by dot ink, from inside the dot
        for (let p = at(rho); p >= 0 && piece[p] === PIECE_WHOLE; p = at(rho += RADIAL_STEP))
            open += (1 - cover[p]) * RADIAL_STEP;
        const end = rho + reach;
        let seenPaper = false;
        for (let p = at(rho); p >= 0 && rho <= end; p = at(rho += RADIAL_STEP)) {
            open += (1 - cover[p]) * RADIAL_STEP;
            if (paper[p]) {
                seenPaper = true;
            }
            else if (piece[p] > PIECE_SPECK) {
                // (the far edge's pixels, at half coverage or more, are in `open` by what they lack)
                if (piece[p] === PIECE_CUT && seenPaper)
                    gaps.push(open + (1 - cover[p]) * FAR_EDGE);
                break;
            }
        }
    }
}

// The pixels that aren't dot ink, in gaps up to `reach` px between dot inks with the page's color showing in them:
// across the gap in one of four directions, as near as a row, a column or a diagonal comes to it.
function gapMask(piece, paper, width, height, reach) {
    const mask = new Uint8Array(width * height);
    // (any family's ink, a sliver's too: what shows of a dot under a stroke may be no more)
    const ink = p => piece[p] > 0;
    for (const [dx, dy] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const most = Math.floor(reach / Math.hypot(dx, dy));
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (!ink(y * width + x))
                    continue;
                // from an ink pixel across to the next one
                let k = 1;
                let seenPaper = false;
                let end = false;
                for (; k <= most + 1; k++) {
                    const u = x + k * dx;
                    const v = y + k * dy;
                    if (u < 0 || v < 0 || u >= width || v >= height)
                        break;
                    const p = v * width + u;
                    if (ink(p)) {
                        end = true;
                        break;
                    }
                    seenPaper ||= paper[p] === 1;
                }
                if (!end || !seenPaper)
                    continue;
                for (let j = 1; j < k; j++)
                    mask[(y + j * dy) * width + x + j * dx] = 1;
            }
        }
    }
    return mask;
}
