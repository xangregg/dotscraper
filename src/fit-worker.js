// Module worker that runs the fit off the main thread, so the page stays responsive and can show progress.
// Messages in: { inputs: [{ family, input }], opts } (see fitDots), one per dot family, fitted in turn. Out:
// { type: 'progress', family, progress } while fitting, { type: 'family', family, result } as each family finishes,
// then { type: 'done' }, or { type: 'error', message }. Stopping is done by terminating the worker.

import { fitDots } from './core/fit.js';

self.onmessage = e => {
    const { inputs, opts } = e.data;
    try {
        for (const { family, input } of inputs) {
            const result = fitDots(input, opts, progress => self.postMessage({ type: 'progress', family, progress }));
            self.postMessage({ type: 'family', family, result });
        }
        self.postMessage({ type: 'done' });
    }
    catch (err) {
        self.postMessage({ type: 'error', message: err.message });
    }
};
