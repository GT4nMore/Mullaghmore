# Mullaghmore

A browser fly-through of **mp_mullaghmore**, my Call of Duty 4 custom map, built from the Radiant `.map` source.

Play: https://oisinmg.ie/Mullaghmore/

Controls: mouse to look, W A S D to fly, Space / C up and down, Shift for speed, scroll to change speed, R to respawn.

## How it works

`tools/build.py` converts the map straight from the CoD4 mod tools into a single `.glb`:

- `mapparse.py` reads the `.map` (iwmap 4) and every prefab it places: brushes, patch meshes, bezier curves and models.
- Brushes are turned into polygons by clipping each face plane against the others; curves are tessellated.
- `xmodel.py` reads the raw `xmodel` / `xmodelsurfs` files for the 325 models; each model is written once and drawn with GPU instancing.
- `assets.py` reads the binary materials and decodes the `.iwi` textures from `raw/images` and the game's `.iwd` archives, saved as WebP.

`docs/` is the published site (three.js viewer + `mullaghmore.glb`). To rebuild: `python tools/build.py` (needs the CoD4 mod tools install, Python 3, numpy and Pillow).
