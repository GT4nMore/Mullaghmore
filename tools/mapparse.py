"""Parser for CoD4 Radiant .map files (iwmap 4), including prefabs.

Produces a flat list of world-space primitives:
  brushes  -> list of (planes, material, texinfo) per face
  patches  -> vertex grids (mesh = explicit grid, curve = bezier control grid)
  models   -> (model name, 4x4 transform)
"""
import math
import os
import re

import numpy as np

FACE_RE = re.compile(
    r'\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s+(\S+)\s+(.*)')
KV_RE = re.compile(r'"([^"]*)"\s+"([^"]*)"')


def angles_matrix(pitch, yaw, roll):
    """Quake/CoD angles (degrees) to a 3x3 rotation matrix (columns = forward, left, up)."""
    p, y, r = (math.radians(a) for a in (pitch, yaw, roll))
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    forward = np.array([cp * cy, cp * sy, -sp])
    left = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp])
    up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
    return np.stack([forward, left, up], axis=1)


def entity_matrix(ent):
    m = np.eye(4)
    if 'angles' in ent:
        a = [float(v) for v in ent['angles'].split()]
        m[:3, :3] = angles_matrix(*a)
    elif 'angle' in ent:
        m[:3, :3] = angles_matrix(0, float(ent['angle']), 0)
    if 'modelscale' in ent:
        m[:3, :3] *= float(ent['modelscale'])
    if 'origin' in ent:
        m[:3, 3] = [float(v) for v in ent['origin'].split()]
    return m


class Brush:
    def __init__(self, faces):
        self.faces = faces  # list of dict(points=(3,3) array, material, tex=[...])


class Patch:
    def __init__(self, kind, material, rows):
        self.kind = kind          # 'mesh' or 'curve'
        self.material = material
        self.rows = rows          # list of rows, each a list of (xyz, uv-in-texels)


def tokenize_blocks(lines):
    """Split the file into entities -> raw primitive line blocks."""
    entities = []
    i, n = 0, len(lines)
    depth = 0
    ent = None
    prim = None
    while i < n:
        s = lines[i].strip()
        i += 1
        if not s or s.startswith('//'):
            continue
        if s == '{':
            depth += 1
            if depth == 1:
                ent = {'kv': {}, 'prims': []}
            elif depth == 2:
                prim = []
            else:
                prim.append(s)
            continue
        if s == '}':
            if depth == 2:
                ent['prims'].append(prim)
                prim = None
            elif depth == 1:
                entities.append(ent)
                ent = None
            else:
                prim.append(s)
            depth -= 1
            continue
        if depth == 1:
            m = KV_RE.match(s)
            if m:
                ent['kv'][m.group(1)] = m.group(2)
        elif depth >= 2:
            prim.append(s)
    return entities


def parse_patch(prim):
    kind = None
    j = 0
    while j < len(prim):
        if prim[j] in ('mesh', 'curve'):
            kind = prim[j]
            break
        j += 1
    body = [l for l in prim[j + 1:] if l not in ('{', '}')]
    # skip contents/toolFlags lines
    body = [l for l in body if not l.startswith(('contents', 'toolFlags', 'smoothing'))]
    material = body[0]
    # body[1] = lightmap material, body[2] = "rows cols ..." header
    rows = []
    cur = None
    for l in body[3:]:
        if l == '(':
            cur = []
        elif l == ')':
            rows.append(cur)
        elif l.startswith('v '):
            t = l.split()
            xyz = (float(t[1]), float(t[2]), float(t[3]))
            ti = t.index('t')
            uv = (float(t[ti + 1]), float(t[ti + 2]))
            cur.append((xyz, uv))
    return Patch(kind, material, rows)


def parse_brush(prim):
    faces = []
    for l in prim:
        m = FACE_RE.match(l)
        if not m:
            continue
        pts = np.array([float(m.group(k)) for k in range(1, 10)]).reshape(3, 3)
        material = m.group(10)
        tex = m.group(11).split()
        faces.append({'points': pts, 'material': material,
                      'tex': [float(v) for v in tex[:6]]})
    return Brush(faces) if faces else None


class MapScene:
    def __init__(self):
        self.brushes = []   # (Brush, 4x4)
        self.patches = []   # (Patch, 4x4)
        self.models = []    # (name, 4x4)
        self.worldspawn = {}
        self.entities = []  # (kv, 4x4) for every point entity


def load_map(path, scene=None, xform=None, map_root=None, depth=0):
    if scene is None:
        scene = MapScene()
    if xform is None:
        xform = np.eye(4)
    if map_root is None:
        map_root = os.path.dirname(path)
    with open(path, encoding='latin1') as f:
        lines = f.read().split('\n')
    for ent in tokenize_blocks(lines):
        kv = ent['kv']
        cls = kv.get('classname', '')
        if cls == 'worldspawn' and depth == 0:
            scene.worldspawn = kv
        if cls == 'misc_prefab':
            sub = os.path.join(map_root, kv['model'].replace('/', os.sep))
            if os.path.exists(sub):
                load_map(sub, scene, xform @ entity_matrix(kv), map_root, depth + 1)
            else:
                print('missing prefab', kv['model'])
            continue
        if cls == 'misc_model':
            scene.models.append((kv['model'], xform @ entity_matrix(kv)))
            continue
        if cls == 'script_model' and 'model' in kv:
            scene.models.append((kv['model'], xform @ entity_matrix(kv)))
        # brush entities: brushes are stored in world space already (relative to prefab)
        for prim in ent['prims']:
            if any(p in ('mesh', 'curve') for p in prim[:4]):
                scene.patches.append((parse_patch(prim), xform))
            else:
                b = parse_brush(prim)
                if b:
                    scene.brushes.append((b, xform))
        if not ent['prims']:
            scene.entities.append((kv, xform @ entity_matrix(kv)))
    return scene


if __name__ == '__main__':
    import sys
    s = load_map(sys.argv[1])
    print('brushes', len(s.brushes), 'patches', len(s.patches), 'models', len(s.models))
    from collections import Counter
    print(Counter(m for m, _ in s.models).most_common(15))
