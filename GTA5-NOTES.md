# GTA V on KytyPS5 — working notes

Branch `hdr-pr423-test`. Everything below was verified in game unless marked otherwise.

## Running it

```bash
./_Build/profile-buffer-readback/kyty_emulator.exe \
  --game "<game dir>" --screen-width 2560 --screen-height 1440 --hdr-clamp 1 \
  --keymap "LeftStickUp=w" --keymap "LeftStickDown=s" --keymap "LeftStickLeft=a" --keymap "LeftStickRight=d" \
  --keymap "RightStickUp=i" --keymap "RightStickDown=k" --keymap "RightStickLeft=j" --keymap "RightStickRight=l" \
  --keymap "L2=Mouse:Right" --keymap "R2=Mouse:Left" --keymap "R3=Mouse:Middle" \
  --keymap "L1=Mouse:X1" --keymap "R1=Mouse:X2" \
  --keymap "Cross=space" --keymap "Circle=r" --keymap "Square=left ctrl" --keymap "Triangle=f" \
  --keymap "L3=left shift" --keymap "Options=escape" --keymap "TouchPad=tab" \
  --keymap "Up=up" --keymap "Down=down" --keymap "Left=left" --keymap "Right=right" \
  --keymap "MouseSensitivity=3.0"
```

`--hdr-clamp 1` is **required** — without it the screen whites out (root cause still open, see below).

`--keymap` REPLACES the defaults rather than merging, so the full list must be given.
Emulator keys: **F7** mouse look (off by default), **F2** pause, **F3** mute/unmute the
controller speaker, **F11** fullscreen, **F12** RenderDoc capture, **Ctrl/Shift+Esc** quit.

**F3** routes pad-speaker audio (phone calls) to the normal output device instead of the
DualSense, for playing late through a headset. It prints the new state to the console.
`--pad-speaker-muted t` starts muted. Muting drops whatever is buffered, so unmuting does not
replay a stale sentence.

Driving is R2/L2 (left/right mouse) — W and S only steer, they do not accelerate.

## Fixed

| Symptom | Cause |
|---|---|
| Props/grates drawn over everything, through walls | An HTile read-modify-write dispatch was mistaken for a fast clear, wiping scene depth mid-frame. Ported from brandostrong `9faf09a`. |
| Water jug on the top layer | A stale uncommitted pre-#370 revert in the working tree. `d231895` had already fixed the red it existed to work around. |
| Washed-out haze over everything | `--hdr-clamp` applied to every float target, crushing the buffers the bloom/tonemap chain needs. Now scoped to FP16_ABGR exports only. |
| Crash while streaming (hills, entering cars) | Remapping guest direct memory is not atomic; a guest thread touching the placeholder window faulted on memory committed before and after. Now retried instead of aborted. |
| Crash on a fault near a mapping boundary | `HandleFault` qualified on an 8-byte window, so a fault within 8 bytes of a mapping end was declined and became fatal. Upstream PR #422. |
| Saves written but never loadable | `SaveDataSetParam` discarded metadata, `GetParam` returned zeros, the dir search blanked params — so a valid save read as an empty slot. Upstream PR #434. |
| Red static on windows | `DecodePackedColorClear` had no `R32Sfloat` case, so the clear was discarded and the target kept stale contents. |
| Emulator quitting/freezing at random | ESC quit instantly and SPACE toggled pause — both constantly pressed in play. Now Ctrl+Esc and F2. |

## Still open

- **HDR white-out without `--hdr-clamp`.** Measured in a confirmed-white frame: ~2.65e36 in two
  R32_SFLOAT targets, and 35840 in the B10G11R11 bloom chain (54x30 / 160x90 / 1280x720) while every
  other target read 0. The small ones are auto-exposure adaptation buffers that feed back each frame,
  so this looks like a diverging exposure loop. No NaN/Inf anywhere. Ruled out: DCC clears
  (`has_dcc=0` on every target), stale image readback (upstream #423/#424 both applied, no effect),
  format aliasing, missing clears.
- **Vehicle wheels render wrong.** Not investigated. It is a rendering transform issue, not physics.
- **Menu text drops glyphs** ("R play Missi n").

## Gotchas

- Launching a second instance fails instantly (exit 65) — it cannot reserve the 13.8 GB of guest
  direct memory the first one holds. Check for a live instance before blaming a crash.
- Exit code 65 usually means a guest fault, not a clean exit. Read the log.
- `LOGF` is silenced by default, so diagnostics must use `::printf` or be folded into `EXIT`.
- Save backups live in `C:\Users\konze\KytyPS5-SaveBackups\`.
