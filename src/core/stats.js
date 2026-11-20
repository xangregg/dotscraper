// Order statistics of small arrays. None of them changes its argument, and each is O(n log n).
//
// There are two medians because the rules were measured with two, and at an even count they differ: `median` gives
// the mean of the middle two, a value between the measurements (radii, widths), and `upperMedian` the upper of the
// two, always one of the values given (a color channel that some pixel has, a count some glyph has).

const ascending = (x, y) => x - y;

/**
 * The median: the middle value, or the mean of the middle two. NaN for an empty array.
 *
 * @param {ArrayLike<number>} a
 * @returns {number}
 */
export function median(a) {
    const s = [...a].sort(ascending);
    return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

/**
 * The middle value, or the upper of the middle two: always one of the values given. Undefined for an empty array.
 *
 * @param {ArrayLike<number>} a
 * @returns {number|undefined}
 */
export function upperMedian(a) {
    return [...a].sort(ascending)[a.length >> 1];
}

/**
 * The value at the share q of the way through the sorted values (the nearest one: no interpolation).
 *
 * @param {ArrayLike<number>} a
 * @param {number} q  0–1
 * @returns {number}
 */
export function quantile(a, q) {
    const s = [...a].sort(ascending);
    return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];
}
