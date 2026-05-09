# Solar System Scale Reference

## Units
- **Distance**: 1 unit = 1 AU (Astronomical Unit = 149,597,870.7 km)
- **Time**: 1 unit = 1 year
- **Mass**: GM_sun = 4pi^2 (so dt=0.001 gives ~1000 steps/year)

## Visual Radius Scale

**VR_SCALE = 0.05 / 69,911 = 7.15e-7 AU/km**

This is approximately **107x real size** (real: 1 km = 6.685e-9 AU).

| Body     | Real Radius (km) | Real Radius (AU)  | Visual Radius (AU) | Magnification |
|----------|------------------:|------------------:|--------------------:|--------------:|
| Sun      | 695,700           | 0.00465           | 0.03 (capped)       | 6.5x          |
| Mercury  | 2,440             | 0.0000163         | 0.00174             | 107x          |
| Venus    | 6,052             | 0.0000404         | 0.00433             | 107x          |
| Earth    | 6,371             | 0.0000426         | 0.00456             | 107x          |
| Mars     | 3,390             | 0.0000227         | 0.00242             | 107x          |
| Jupiter  | 69,911            | 0.000467          | 0.0500              | 107x          |
| Saturn   | 58,232            | 0.000389          | 0.0416              | 107x          |
| Uranus   | 25,362            | 0.000170          | 0.0181              | 107x          |
| Neptune  | 24,622            | 0.000165          | 0.0176              | 107x          |

## Why 107x?

Real proportions make planets invisible (Earth would be 0.00004 AU at 1 AU distance).
107x keeps planets proportionally correct relative to each other while remaining visible
at solar-system zoom levels. With billboard quads (no hardware size limit), the camera
can zoom arbitrarily close to any body.

## Asteroids / Kuiper Belt
- Asteroid belt bodies: 0.001 AU visual radius
- Kuiper belt bodies: 0.0012 AU visual radius
- These are not physically scaled — purely for visibility.

## Sun Cap
The Sun's proportional visual radius at 107x would be ~0.5 AU, which would visually
engulf Mercury's orbit (0.387 AU). It is capped at 0.03 AU (~6.5x real).

## To Change the Scale
In `src/simulation/InitialConditions.cpp`:
```cpp
constexpr float VR_SCALE = 0.05f / R_JUPITER_KM;
```
Increase `0.05f` for larger planets, decrease for smaller.
The Sun cap is set separately in the `bodies[]` array (first entry, last field).
