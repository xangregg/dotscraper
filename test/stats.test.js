import test from 'node:test';
import assert from 'node:assert/strict';
import { median, upperMedian, quantile } from '../src/core/stats.js';

test('median: the middle value, or the mean of the middle two', () => {
    assert.equal(median([5, 1, 3]), 3);
    assert.equal(median([4, 1, 3, 2]), 2.5);
    assert.equal(median([7]), 7);
    assert.ok(Number.isNaN(median([])));
    assert.equal(median(new Float32Array([0.5, 0.25])), 0.375);
});

test('upperMedian: the middle value, or the upper of the middle two, always one of those given', () => {
    assert.equal(upperMedian([5, 1, 3]), 3);
    assert.equal(upperMedian([4, 1, 3, 2]), 3);
    assert.equal(upperMedian([10, 200]), 200);
    assert.equal(upperMedian([]), undefined);
    assert.equal(upperMedian(new Uint8Array([9, 2, 4, 4])), 4);
});

test('the order statistics sort by value and leave their argument as it was', () => {
    // (the default sort compares as text: 10 before 9)
    const a = [10, 9, 100, 1];
    assert.equal(median(a), 9.5);
    assert.equal(upperMedian(a), 10);
    assert.equal(quantile(a, 1), 100);
    assert.deepEqual(a, [10, 9, 100, 1]);
});

test('quantile: the nearest sorted value, without interpolation', () => {
    assert.equal(quantile([1, 2, 3, 4, 5], 0.5), 3);
    assert.equal(quantile([1, 2, 3, 4, 5], 0), 1);
    assert.equal(quantile([1, 2, 3, 4, 5], 0.9), 5);
    assert.equal(quantile([1, 2, 3, 4], 0.4), 2);
});
