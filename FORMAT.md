# Recording format (OS-neutral)

A session is a JSONL file: one compact JSON object per line, one input event per
line, sorted by ascending `t`. The format is deliberately OS-neutral so a
session can be captured on any platform and sent anywhere without conversion.

## Coordinates and time

* `x`, `y` — pointer position as a fraction of the primary surface, each in
  `[0.0, 1.0]`. Fraction-of-surface is used rather than device pixels so the
  same behaviour maps across different screen sizes, resolutions and
  multi-monitor layouts. Top-left is `(0, 0)`.
* `t` — event time in **milliseconds** from an arbitrary-but-monotonic epoch
  (e.g. `os.clock_gettime` / `QPC` / milliseconds since boot). The pipeline
  only ever consumes *differences* between `t` values, so the epoch itself is
  irrelevant and the format requires no wall-clock synchronisation.

Only these three properties are interpreted by the model. Any other keys are
ignored, so platform-specific extras (button ids, scan codes, vendor HID
details) may be carried along without breaking interoperability.

## Event kinds

### `pointer`
A movement sample of the pointing device.

```json
{"t": 1042, "kind": "pointer", "x": 0.5013, "y": 0.3378}
```

### `click`
A button press or release.

```json
{"t": 1141, "kind": "click", "x": 0.5013, "y": 0.3378, "button": "left", "action": "down"}
```

### `text`
Characters as they arrive. The `source` field distinguishes the three ways text
can enter:

| `source` | meaning |
|---|---|
| `"typed"` | a physical keyboard (keystrokes) |
| `"onscreen"` | an on-screen keyboard the user taps |
| `"inserted"` | software-inserted text (dictation, paste, assistive tech) |

`chars` is the characters delivered in this batch. A backspace is the literal
`\b` character inside a `"typed"` batch:

```json
{"t": 1507, "kind": "text", "source": "typed", "chars": "a"}
{"t": 1581, "kind": "text", "source": "typed", "chars": "\b"}
{"t": 1652, "kind": "text", "source": "typed", "chars": "e"}
```

A dictation chunk is a single `"inserted"` batch; a later on-screen edit is
`"onscreen"`:

```json
{"t": 2200, "kind": "text", "source": "inserted", "chars": "recognize the user"}
{"t": 3350, "kind": "text", "source": "onscreen",   "chars": "y"}
```

## The invariant

The format is **exactly the events the operating system already sees**:

* pointer movement and clicks, and
* text as it arrives — physical keyboard, on-screen keyboard, or software
  insertion (dictation/paste), including the pattern of corrections.

It contains no imprint of the user's name, no account handle, no device serial,
and no content from open documents — only timing and motion. That is what keeps
enrolment and scoring zero-knowledge with respect to *who* the user is: the
system learns *how they move*, not *who they are*.

## Detecting input style from a recording

The pipeline detects, rather than assumes, the input style:

* a session whose text is predominantly `"typed"` is the keyboard style;
* a session with `"inserted"`/`"onscreen"` chunks and slow pointer motion is
  the dictation style.

Both feed the same 13-feature vector; features that do not occur simply read
`0`, so a single model and a single format serve every style and every device.