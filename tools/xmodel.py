"""Reader for CoD4 raw (version 25) xmodel / xmodelparts / xmodelsurfs files.

Only what a static viewer needs: LOD0 geometry in model space with UVs, normals,
vertex colours and one material per surface.
"""
import os
import struct

import numpy as np


def _str(d, o):
    e = d.index(b'\0', o)
    return d[o:e].decode('latin1'), e + 1


def read_surfs(path):
    """Return list of dicts: pos (n,3), nrm (n,3), uv (n,2), col (n,4 uint8), idx (m,3)."""
    d = open(path, 'rb').read()
    _, n_surf = struct.unpack_from('<HH', d, 0)
    o = 4
    out = []
    for _ in range(n_surf):
        tile = d[o]
        n_vert, n_tri = struct.unpack_from('<HH', d, o + 3)
        o += 7
        first, = struct.unpack_from('<H', d, o)
        if first == n_vert:
            # whole surface on one bone: 60-byte vertices
            o += 6
            raw = np.frombuffer(d, dtype=np.uint8, count=n_vert * 60, offset=o).reshape(n_vert, 60)
            o += n_vert * 60
            f = raw.copy()
            pos = f[:, 48:60].copy().view('<f4').reshape(n_vert, 3).astype(np.float64)
        else:
            # (count, bone) blocks terminated by a zero count; vertices carry their own bone
            while True:
                cnt, _ = struct.unpack_from('<HH', d, o)
                o += 4
                if cnt == 0:
                    break
            f = np.zeros((n_vert, 60), np.uint8)
            pos = np.zeros((n_vert, 3))
            for k in range(n_vert):
                f[k, :48] = np.frombuffer(d, np.uint8, 48, o)
                n_w = d[o + 48]
                pos[k] = struct.unpack_from('<3f', d, o + 51)
                o += 63 + n_w * 4   # extra weights: (bone, weight) u16 pairs, unused
        nrm = f[:, 0:12].copy().view('<f4').reshape(n_vert, 3).astype(np.float64)
        np.nan_to_num(nrm, copy=False)
        col = f[:, 12:16].copy()
        uv = f[:, 16:24].copy().view('<f4').reshape(n_vert, 2)
        idx = np.frombuffer(d, dtype='<u2', count=n_tri * 3, offset=o).reshape(n_tri, 3).astype(np.uint32)
        o += n_tri * 6
        out.append({'pos': pos, 'nrm': nrm, 'uv': uv.astype(np.float64), 'col': col, 'idx': idx, 'tile': tile})
    if o != len(d):
        raise ValueError(f'surfs size mismatch {o} != {len(d)}')
    return out


def read_xmodel(raw_dir, name, material_exists):
    d = open(os.path.join(raw_dir, 'xmodel', name), 'rb').read()
    o = 27
    _, o = _str(d, o)                    # physics preset
    o += 4                               # lod0 distance
    lod0, o = _str(d, o)
    # Raw xmodelsurfs store positions in model space, so bones are not needed.
    surfs = read_surfs(os.path.join(raw_dir, 'xmodelsurfs', lod0))
    mats = find_material_list(d, len(surfs), material_exists)
    if mats is None:
        raise ValueError('material list not found')
    for s, m in zip(surfs, mats):
        s['material'] = m
    return surfs


def find_material_list(d, count, material_exists):
    """Locate the LOD0 material list: u16 count followed by `count` material names."""
    want = struct.pack('<H', count)
    start = 27
    while True:
        i = d.find(want, start)
        if i < 0:
            return None
        start = i + 1
        o = i + 2
        names = []
        try:
            for _ in range(count):
                s, o = _str(d, o)
                if not s or not material_exists(s):
                    raise ValueError
                names.append(s)
        except ValueError:
            continue
        return names
