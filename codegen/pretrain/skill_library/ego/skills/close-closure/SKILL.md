---
name: close-closure
description: Swing a closure shut with both hands on it and come off it; nothing is held afterwards.
---

# close-closure

```python
close_closure(robot, closure_name, side, other_side, axis, angle_deg)
```

## When to use

- A subtask like "Close the `<lid|cover|case|door|latch>` (of|on)
  `<X>`" where the video shows BOTH hands on the closure — machine lids, cases
  and bins are often closed two-handed.
- The closure SWINGS shut about a hinge (`rotate`), and both hands come off it
  in the same slice.
- `angle_deg` is the CLOSING sense — the negation of what `open_closure` uses
  on the same closure.
- Example closures: 'case lid', 'tray lid', 'machine lid', 'pot lid',
  'storage box lid', 'backpack'. Typical values: `axis='lateral'`,
  `angle_deg=-90.0`.

## When NOT to use

- One hand closes it while the other braces the body: use `open_closure`'s
  shape with the closing angle sign, or spell it out.
- The closure is SNAPPED/PRESSED home rather than swung: `press_onto_target`.
- The closure is screwed shut: that is the reverse of `unscrew_cap` — spell it
  out.
- Opening the same closure: `open_closure`.

## Hand contract (derived)

    close(<other_side>) -> close(<side>) -> open(<other_side>) -> open(<side>)

`grasp` closes, `rotate` is transparent, `release` opens. Both hands are clear
afterwards.

Contract collision: `move_item_two_handed` derives the same
close/close/open/open shape. Choose by the motion: a swing (`rotate`) here, a
translation (`move_arm`) there.

## Known limits

- The contract is derived from the code; a lid pressed shut with flat palms
  shows little visible change in hand aperture.
- "Close" is realized several ways — a swing (here), a two-handed press or
  `attach` (`press_onto_target`), and a braced latch press (`grasp` +
  `press`, written inline). This body covers only the swing.
- `angle_deg` is a semantic swing amount, not a measured one.
