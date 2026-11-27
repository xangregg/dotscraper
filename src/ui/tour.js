// The seven steps of the About dialog's "Using it", numbered on the controls they use: with short labels over the
// app when it loads, until the first sign of activity, and as numbers alone while the About dialog is open, so its
// list can be matched to the page.
//
// The layer is a manual popover so that, shown after the dialog opens, it is in the top layer above the dialog's
// backdrop; it takes no pointer events, so the click that dismisses it still reaches the control under it. While
// numbered, the dialog moves over the image view, clear of the header and panel where most steps are; a number it
// would still cover is left out rather than drawn over its text.

const BADGE = 30; // px, the number's circle (its width in styles.css)
const GAP = 6; // px between a target and its number, and a number and its label
const INSET = 40; // px from a region's corner to a number placed inside it, clear of the toolbar above it
const DROP = 10; // px further below a section's name for underAligned, clear of the labels above it
const OVERLAP = 10; // px of a number over its button's bottom edge (onStart, onEnd)
const MARGIN = 8; // px kept clear at the window's edges
const NUMBER_WEIGHT = 10; // how much worse covering a number or label is than covering as much of a control
const DIALOG_CLEAR = 6; // px around the About dialog that numbers stay out of
const DOCK_MIN_WIDTH = 500; // px, the narrowest image view the About dialog moves over
const DOCK_MARGIN = 16; // px between the moved dialog and the view's edges

// target: the element the step uses; text: measure its first text node (a section's name) instead of the element;
// layouts: where its number and label go, in order of preference (layoutAt); about: layouts tried first for the
// number alone beside the About dialog, where it may cover its own target; alignTo: for underAligned, the text whose
// rightOf number its number lines up with; label: none for a step numbered only beside the About dialog. A step
// takes the first layout that stays in the window and covers no header control, step target, number or label;
// failing those (a narrow screen), the same of FALLBACKS, and failing all, whichever covers least. In a big window
// the labels read in order: 1 and 3 side by side under their buttons, 2 under its section's name, its number over
// 6's when About shows 6. Beside About the numbers sit close to what they number: 1 and 3 over their buttons' bottom
// edges, 5 left of the edit buttons.
const STEPS = [
    { target: '#open-file', layouts: ['underEnd', 'underStart'], about: ['onEnd'], label: 'Import an image' },
    { target: '#sec-families > summary', text: true, alignTo: '#sec-axes > summary', layouts: ['underAligned', 'underStart', 'rightOf'], label: 'Review and edit dot families' },
    { target: '#fit-run', layouts: ['underStart', 'underEnd'], about: ['onStart'], label: 'Find dots' },
    { target: '#stage', layouts: ['inside'], label: 'Review the dots found' },
    { target: '[aria-label="Edit dots"]', layouts: ['underEnd', 'underStart'], about: ['leftOf'], label: 'Fix with Add, Remove and Move' },
    // Axes and Export aren't needed to get started, and labels there would break the labels' reading order
    { target: '#sec-axes > summary', text: true, layouts: ['rightOf', 'underStart'], label: null },
    { target: '[aria-label="Export"]', layouts: ['leftOf', 'underEnd'], label: null },
];
// the number below, right or left of the target, each with its label on any side of it
const FALLBACKS = ['below', 'right', 'left'].flatMap(b => ['right', 'left', 'under', 'underLeft', 'above'].map(l => `${b}:${l}`));
// what neither a number nor a label covers, besides the steps' targets
const KEEP_CLEAR = 'header button:not([hidden]), header select:not([hidden])';

const ACTIVITY = ['pointerdown', 'keydown', 'wheel', 'dragenter', 'paste'];

/**
 * Builds the steps layer and shows it with its labels; the first click, key, wheel turn, drag or paste hides it.
 * @param {HTMLDialogElement} about the About dialog, whose list the numbers match
 * @returns {{ showIntro: () => void, showWithAbout: () => void }} showIntro shows it with its labels again (on a
 *     click, so the click's own pointerdown has passed); showWithAbout, called right after the dialog opens,
 *     numbers the page beside it
 */
export function wireTour(about) {
    const layer = document.createElement('div');
    layer.id = 'tour';
    layer.popover = 'manual';
    // the same steps are in the About dialog's list, as text
    layer.setAttribute('aria-hidden', 'true');
    const items = STEPS.map((step, i) => {
        const badge = Object.assign(document.createElement('span'), { className: 'step-badge', textContent: String(i + 1) });
        const label = Object.assign(document.createElement('span'), { className: 'step-label', textContent: step.label ?? '' });
        layer.append(badge, label);
        return { step, badge, label };
    });
    document.body.append(layer);

    let mode = null; // 'intro' with labels, 'about' numbers alone, or null when hidden
    const place = () => {
        if (mode === 'about')
            dock(about);
        if (mode)
            placeSteps(items, mode === 'about' ? about.getBoundingClientRect() : null);
    };
    const show = m => {
        mode = m;
        if (m === 'intro') {
            for (const type of ACTIVITY)
                window.addEventListener(type, onActivity, { capture: true, passive: true });
        }
        // shown again so that it is the newest in the top layer, above a dialog opened since
        if (layer.matches(':popover-open'))
            layer.hidePopover();
        layer.showPopover();
        place();
    };
    const hide = () => {
        mode = null;
        if (layer.matches(':popover-open'))
            layer.hidePopover();
    };

    function onActivity() {
        for (const type of ACTIVITY)
            window.removeEventListener(type, onActivity, true);
        if (mode === 'intro')
            hide();
    }
    about.addEventListener('close', () => {
        about.removeAttribute('style');
        if (mode === 'about')
            hide();
    });
    // the targets move as the header wraps and the panel fills, and the dialog's size decides which numbers it covers
    window.addEventListener('resize', place);
    // the panel's sections, not the panel: its own size stays put while a family list filling in moves Axes down
    const observer = new ResizeObserver(place);
    for (const el of [document.querySelector('header'), document.querySelector('aside'), ...document.querySelectorAll('aside > details'), about])
        observer.observe(el);

    show('intro');
    return { showIntro: () => show('intro'), showWithAbout: () => show('about') };
}

// Moves the open About dialog over the image view, below its toolbar, when the view is wide enough; otherwise it
// stays where a modal dialog goes, in the middle of the window.
function dock(about) {
    const view = document.getElementById('stage').getBoundingClientRect();
    if (view.width < DOCK_MIN_WIDTH) {
        about.removeAttribute('style');
        return;
    }
    // left and right both set with automatic margins: the dialog takes its max-width, centered in the view
    const top = view.top + DOCK_MARGIN;
    Object.assign(about.style, {
        inset: `${top}px ${window.innerWidth - view.right + DOCK_MARGIN}px auto ${view.left + DOCK_MARGIN}px`,
        margin: '0 auto',
        maxHeight: `${window.innerHeight - top - DOCK_MARGIN}px`,
    });
}

// Positions each number by its target and, with labels, each label beside it. Numbers whose target is hidden, and
// beside the dialog any the dialog would cover, are hidden; beside it, each number is placed as if it had no label,
// so it stays by its target.
function placeSteps(items, dialog) {
    const W = window.innerWidth;
    const H = window.innerHeight;
    const rects = items.map(({ step }) => targetRect(step.target, step.text));
    // the image view is a region numbered inside, not a control to keep clear of
    const taken = [...rects.filter((r, i) => r && !items[i].step.layouts.includes('inside')),
        ...[...document.querySelectorAll(KEEP_CLEAR)].filter(el => el.getClientRects().length).map(el => el.getBoundingClientRect())];
    // numbers and labels count for more than controls
    const marks = [];
    // (own: the target's box, which an About layout may cover; its button is in taken twice, as a target and a control)
    const covered = (b, own) => taken.reduce((sum, t) => (own && sameBox(t, own) ? sum : sum + overlapArea(b, t, GAP / 2)), 0)
        + NUMBER_WEIGHT * marks.reduce((sum, t) => sum + overlapArea(b, t, GAP / 2), 0);
    const inWindow = b => b.top >= 0 && b.bottom <= H;
    items.forEach(({ step, badge, label }, i) => {
        const r = rects[i];
        const labeled = !!step.label && !dialog;
        // measured while shown
        label.hidden = !labeled;
        const w = labeled ? label.offsetWidth : 0;
        const h = labeled ? label.offsetHeight : 0;
        if (!r || (!step.label && !dialog)) {
            badge.hidden = label.hidden = true;
            return;
        }
        let best = null;
        let bestCost = Infinity;
        const align = step.alignTo ? targetRect(step.alignTo, true) : null;
        const about = dialog && step.about ? step.about : [];
        for (const name of [...about, ...step.layouts, ...FALLBACKS].filter(n => n !== 'underAligned' || align)) {
            const at = layoutAt(name, r, w, h, W, align);
            const boxes = labeled ? [at.badge, at.label] : [at.badge];
            if (!boxes.every(inWindow))
                continue;
            const own = about.includes(name) ? r : null;
            const cost = boxes.reduce((sum, b) => sum + covered(b, own), 0);
            if (cost < bestCost) {
                best = at;
                bestCost = cost;
            }
            if (cost === 0)
                break;
        }
        best ??= layoutAt(step.layouts.find(n => n !== 'underAligned'), r, w, h, W);
        badge.hidden = !!dialog && overlaps(best.badge, dialog, DIALOG_CLEAR);
        label.hidden = !labeled || badge.hidden;
        if (badge.hidden)
            return;
        badge.style.left = `${best.badge.left}px`;
        badge.style.top = `${best.badge.top}px`;
        marks.push(best.badge);
        if (label.hidden)
            return;
        label.style.left = `${best.label.left}px`;
        label.style.top = `${best.label.top}px`;
        marks.push(best.label);
    });
}

// The boxes of a number and its label (w × h px; 0 × 0 for none) in one layout by the target's box r, slid along
// together to stay within the window's width W. The named layouts keep the label after its number, on one line:
// underStart and underEnd under the target, starting at its left or ending at its right; underAligned further
// under it, its number where rightOf would put the number of the text a (a's right plus GAP); onStart and onEnd
// like underStart and underEnd but moved up over the target's bottom edge; rightOf and leftOf beside it; inside its
// top-left corner. A fallback, `number:label`, puts the number below, right or left of the target
// and the label right, left, under, under-and-left-aligned or above the number.
function layoutAt(name, r, w, h, W, a) {
    const span = BADGE + (w ? GAP + w : 0);
    const below = r.bottom + GAP;
    const middle = r.top + r.height / 2 - BADGE / 2;
    const named = {
        underStart: [r.left, below, 'right'],
        underEnd: [r.right - span, below, 'right'],
        underAligned: a && [a.right + GAP, below + DROP, 'right'],
        onStart: [r.left, r.bottom - OVERLAP, 'right'],
        onEnd: [r.right - span, r.bottom - OVERLAP, 'right'],
        rightOf: [r.right + GAP, middle, 'right'],
        leftOf: [r.left - GAP - span, middle, 'left'],
        inside: [r.left + INSET, r.top + INSET, 'right'],
    };
    let x;
    let y;
    let side;
    if (name in named) {
        [x, y, side] = named[name];
        // leftOf's label comes first, so its number is at the group's right end
        if (name === 'leftOf')
            x += span - BADGE;
    }
    else {
        const [place, labelSide] = name.split(':');
        side = labelSide;
        y = place === 'below' ? below : middle;
        x = place === 'below' ? r.left + r.width / 2 - BADGE / 2 : place === 'right' ? r.right + GAP : r.left - GAP - BADGE;
    }
    const lx = {
        right: x + BADGE + GAP,
        left: x - GAP - w,
        under: x,
        underLeft: x + BADGE - w,
        above: x,
    }[side];
    const ly = side === 'under' || side === 'underLeft' ? y + BADGE + GAP : side === 'above' ? y - GAP - h : y + (BADGE - h) / 2;
    const left = Math.min(x, lx);
    const right = Math.max(x + BADGE, lx + w);
    const shift = Math.max(MARGIN - left, Math.min(0, W - MARGIN - right));
    const box = (bx, by, bw, bh) => ({ left: bx + shift, top: by, right: bx + shift + bw, bottom: by + bh });
    return { badge: box(x, y, BADGE, BADGE), label: box(lx, ly, w, h) };
}

function overlapArea(a, b, clear) {
    const w = Math.min(a.right, b.right + clear) - Math.max(a.left, b.left - clear);
    const h = Math.min(a.bottom, b.bottom + clear) - Math.max(a.top, b.top - clear);
    return w > 0 && h > 0 ? w * h : 0;
}

function sameBox(a, b) {
    return a.left === b.left && a.top === b.top && a.right === b.right && a.bottom === b.bottom;
}

function overlaps(a, b, clear) {
    return a.left < b.right + clear && a.right > b.left - clear && a.top < b.bottom + clear && a.bottom > b.top - clear;
}

// The box on screen of the element the selector picks, or of its first text node (text), or null when it isn't
// shown
function targetRect(selector, text) {
    const el = document.querySelector(selector);
    if (!el || !el.getClientRects().length)
        return null;
    if (!text)
        return el.getBoundingClientRect();
    const range = document.createRange();
    range.selectNodeContents(el.firstChild);
    return range.getBoundingClientRect();
}
