"""Convert mp_mullaghmore.map (+ prefabs, models, textures) into a web-ready .glb.

    python build.py [MAP_NAME GLB_NAME OUT_DIR]
    python build.py mp_prison prison ../../Prison/docs

Defaults to mp_mullaghmore -> ../docs/mullaghmore.glb. Output:
    GLB_NAME.glb      world geometry merged per material, models as GPU instances
    sky/0..5.webp     sky cubemap faces
    spawn.json        camera start points
"""
import io
import json
import math
import os
import struct
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image

import assets
import mapparse
import xmodel

warnings.filterwarnings('ignore', category=RuntimeWarning)

GAME = r'C:/Program Files (x86)/Steam/steamapps/common/Call of Duty 4 NEW!'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')

MAX_TEX = 1024
SKIP_TECHSETS = {'tools', 'shadowcaster', 'sky'}
CURVE_STEPS = 6

# CoD is Z-up in inches; glTF is Y-up in metres.
SCALE = 0.0254
AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def to_gltf(p):
    return (p @ AXIS.T) * SCALE


def to_gltf_dir(n):
    return n @ AXIS.T


# --- brushes ---------------------------------------------------------------

BASE_AXES = [
    ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
]


def texture_axes(normal, rot_deg):
    best, best_dot = 0, -1
    for i, (n, _, _) in enumerate(BASE_AXES):
        d = float(np.dot(normal, n))
        if d > best_dot + 1e-6:
            best, best_dot = i, d
    _, s, t = BASE_AXES[best]
    s, t = np.array(s, float), np.array(t, float)
    if rot_deg:
        a = math.radians(rot_deg)
        sa, ca = math.sin(a), math.cos(a)
        sv = int(np.argmax(np.abs(s)))
        tv = int(np.argmax(np.abs(t)))
        out = []
        for v in (s, t):
            v = v.copy()
            ns = ca * v[sv] - sa * v[tv]
            nt = sa * v[sv] + ca * v[tv]
            v[sv], v[tv] = ns, nt
            out.append(v)
        s, t = out
    return s, t


def clip_poly(poly, n, d, eps=0.01):
    """Keep the part of `poly` with dot(n, p) <= d."""
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da, db = np.dot(n, a) - d, np.dot(n, b) - d
        if da <= eps:
            out.append(a)
        if (da < -eps and db > eps) or (da > eps and db < -eps):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return out


def brush_polygons(brush):
    planes = []
    for f in brush.faces:
        p0, p1, p2 = f['points']
        n = np.cross(p0 - p1, p2 - p1)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            planes.append(None)
            continue
        n /= ln
        planes.append((n, float(np.dot(n, p1))))
    polys = []
    for i, f in enumerate(brush.faces):
        if planes[i] is None:
            continue
        n, d = planes[i]
        # huge quad on the plane
        a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(n, a)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = n * d
        R = 131072.0
        poly = [c + (-u - v) * R, c + (u - v) * R, c + (u + v) * R, c + (-u + v) * R]
        for j, pl in enumerate(planes):
            if j == i or pl is None:
                continue
            poly = clip_poly(poly, pl[0], pl[1])
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            polys.append((f, n, np.array(poly)))
    return polys


def brush_face_uv(points, normal, tex, img_size):
    size_x, size_y, off_x, off_y, rot = tex[0], tex[1], tex[2], tex[3], tex[4]
    s, t = texture_axes(normal, rot)
    size_x = size_x or 64.0
    size_y = size_y or 64.0
    u = (points @ s + off_x) / size_x
    v = (points @ t + off_y) / size_y
    return np.stack([u, v], axis=1)


# --- patches ---------------------------------------------------------------

def bezier3(p0, p1, p2, t):
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    c = t * t
    return a[..., None] * p0 + b[..., None] * p1 + c[..., None] * p2


def patch_grid(patch):
    """Return (pos grid (R,C,3), uv grid (R,C,2) in texels, alpha grid (R,C,1))."""
    rows = patch.rows
    P = np.array([[v[0] for v in r] for r in rows], float)
    T = np.array([[v[1] for v in r] for r in rows], float)
    A = np.array([[[v[2]] for v in r] for r in rows], float)
    if patch.kind == 'mesh':
        return P, T, A
    # curve: quadratic bezier control grid with odd dimensions
    R, C = P.shape[:2]
    if R < 3 or C < 3 or R % 2 == 0 or C % 2 == 0:
        return P, T, A
    steps = CURVE_STEPS
    t = np.linspace(0, 1, steps + 1)
    out_rows = []
    out_uv = []
    # tessellate along columns first then rows
    def tess(G):
        # G (R, C, k) -> (Rn, Cn, k)
        cols = []
        for c0 in range(0, C - 1, 2):
            seg = bezier3(G[:, c0, None], G[:, c0 + 1, None], G[:, c0 + 2, None], t[None, :])
            cols.append(seg if c0 == 0 else seg[:, 1:])
        H = np.concatenate(cols, axis=1)
        rws = []
        for r0 in range(0, R - 1, 2):
            seg = bezier3(H[r0][None], H[r0 + 1][None], H[r0 + 2][None], t[:, None])
            rws.append(seg if r0 == 0 else seg[1:])
        return np.concatenate(rws, axis=0)
    return tess(P), tess(T), tess(A)


def grid_mesh(P, UV, A):
    R, C = P.shape[:2]
    idx = []
    for r in range(R - 1):
        for c in range(C - 1):
            a, b, cc, d = r * C + c, r * C + c + 1, (r + 1) * C + c + 1, (r + 1) * C + c
            idx.append((a, b, cc))
            idx.append((a, cc, d))
    idx = np.array(idx, np.uint32)
    pos = P.reshape(-1, 3)
    uv = UV.reshape(-1, 2)
    nrm = vertex_normals(pos, idx)
    return pos, nrm, uv, A.reshape(-1), idx


def vertex_normals(pos, idx):
    n = np.zeros_like(pos)
    if len(idx):
        a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
        fn = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(n, idx[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


# --- textures --------------------------------------------------------------

class TextureBank:
    def __init__(self, db):
        self.db = db
        self.cache = {}   # image name -> dict(webp bytes, size, alpha)

    def get(self, image_name, keep_alpha):
        key = (image_name.lower(), keep_alpha)
        if key in self.cache:
            return self.cache[key]
        img = self.db.image(image_name)
        if img is None:
            self.cache[key] = None
            return None
        orig_size = img.size
        if max(img.size) > MAX_TEX:
            f = MAX_TEX / max(img.size)
            img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
        buf = io.BytesIO()
        has_alpha = False
        if keep_alpha:
            a = np.asarray(img)[..., 3]
            has_alpha = bool((a < 250).any())
        if has_alpha:
            img.save(buf, 'WEBP', quality=82, method=6)
        else:
            img.convert('RGB').save(buf, 'WEBP', quality=80, method=6)
        entry = {'data': buf.getvalue(), 'size': orig_size, 'alpha': has_alpha}
        self.cache[key] = entry
        return entry


# --- glTF writer ------------------------------------------------------------

class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.gltf = {
            'asset': {'version': '2.0', 'generator': 'mullaghmore build.py'},
            'extensionsUsed': ['EXT_texture_webp', 'EXT_mesh_gpu_instancing', 'KHR_mesh_quantization'],
            'extensionsRequired': ['EXT_texture_webp', 'KHR_mesh_quantization'],
            'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [],
            'materials': [], 'textures': [], 'images': [], 'samplers': [
                {'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
            'accessors': [], 'bufferViews': [], 'buffers': [],
        }
        self.mat_index = {}
        self.img_index = {}

    def normals(self, nrm):
        """Unit normals as normalized int8, padded to 4 bytes per vertex (KHR_mesh_quantization)."""
        q = np.zeros((len(nrm), 4), np.int8)
        q[:, :3] = np.clip(np.round(nrm * 127), -127, 127)
        bv = self._view(q.tobytes(), 34962)
        self.gltf['bufferViews'][bv]['byteStride'] = 4
        self.gltf['accessors'].append({'bufferView': bv, 'componentType': 5120, 'normalized': True,
                                       'count': len(nrm), 'type': 'VEC3'})
        return len(self.gltf['accessors']) - 1

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin.extend(data)
        bv = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            bv['target'] = target
        self.gltf['bufferViews'].append(bv)
        return len(self.gltf['bufferViews']) - 1

    def accessor(self, arr, kind, normalized=False, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        ctype = {np.dtype('float32'): 5126, np.dtype('uint32'): 5125, np.dtype('uint16'): 5123,
                 np.dtype('uint8'): 5121}[arr.dtype]
        bv = self._view(arr.tobytes(), target)
        acc = {'bufferView': bv, 'componentType': ctype, 'count': int(arr.shape[0]), 'type': kind}
        if normalized:
            acc['normalized'] = True
        if minmax:
            acc['min'] = arr.min(axis=0).tolist()
            acc['max'] = arr.max(axis=0).tolist()
        self.gltf['accessors'].append(acc)
        return len(self.gltf['accessors']) - 1

    def image(self, name, data):
        if name in self.img_index:
            return self.img_index[name]
        bv = self._view(data)
        self.gltf['images'].append({'bufferView': bv, 'mimeType': 'image/webp', 'name': name})
        self.gltf['textures'].append({'sampler': 0, 'extensions': {'EXT_texture_webp': {'source': len(self.gltf['images']) - 1}}})
        self.img_index[name] = len(self.gltf['textures']) - 1
        return self.img_index[name]

    def material(self, key, spec):
        if key in self.mat_index:
            return self.mat_index[key]
        self.gltf['materials'].append(spec)
        self.mat_index[key] = len(self.gltf['materials']) - 1
        return self.mat_index[key]

    def mesh(self, name, prims):
        self.gltf['meshes'].append({'name': name, 'primitives': prims})
        return len(self.gltf['meshes']) - 1

    def node(self, node):
        self.gltf['nodes'].append(node)
        idx = len(self.gltf['nodes']) - 1
        self.gltf['scenes'][0]['nodes'].append(idx)
        return idx

    def write(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.gltf['buffers'] = [{'byteLength': len(self.bin)}]
        js = json.dumps(self.gltf, separators=(',', ':')).encode()
        while len(js) % 4:
            js += b' '
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack('<II', len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack('<II', len(self.bin), 0x004E4942))
            f.write(self.bin)


# --- material -> glTF material ---------------------------------------------

def make_material(glb, bank, mat):
    """Return (material index, texture pixel size) or (None, None) to skip."""
    if mat.is_tool or mat.techset in SKIP_TECHSETS or mat.techset.startswith('wc_tools'):
        return None, None
    alpha_test = any(p.startswith('t0') for p in mat.techset.split('_'))
    blend = mat.blend or mat.additive or mat.techset.startswith(('unlit_add', 'unlit_multiply', 'water'))
    tex = bank.get(mat.color_map, keep_alpha=alpha_test or blend) if mat.color_map else None
    if tex is None:
        return None, None
    spec = {
        'name': mat.name,
        'pbrMetallicRoughness': {
            'baseColorTexture': {'index': glb.image(mat.color_map.lower() + ('#a' if tex['alpha'] else ''), tex['data'])},
            'metallicFactor': 0.0, 'roughnessFactor': 1.0,
        },
        'doubleSided': True,
        'extras': {'techset': mat.techset},
    }
    if alpha_test and tex['alpha']:
        spec['alphaMode'] = 'MASK'
        spec['alphaCutoff'] = 0.5
    elif blend:
        spec['alphaMode'] = 'BLEND'
        if mat.techset.startswith('water'):
            spec['pbrMetallicRoughness']['baseColorFactor'] = [0.55, 0.7, 0.8, 0.75]
        spec['extras']['decal'] = True
        if mat.additive or mat.techset.startswith('unlit_add'):
            spec['extras']['additive'] = True
        if mat.techset.startswith('unlit_multiply'):
            spec['extras']['multiply'] = True
    if mat.techset.startswith('unlit'):
        spec['extras']['unlit'] = True
    return glb.material(mat.name, spec), tex['size']


# --- main ------------------------------------------------------------------

def main(map_name='mp_mullaghmore', glb_name='mullaghmore', out_dir=None):
    global OUT
    if out_dir:
        OUT = os.path.abspath(out_dir)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    MAP = f'{GAME}/map_source/{map_name}.map'
    map_iwd = f'{GAME}/usermaps/{map_name}/{map_name}.iwd'
    db = assets.AssetDB(GAME, [map_iwd] if os.path.exists(map_iwd) else [])
    bank = TextureBank(db)
    glb = GLB()
    scene = mapparse.load_map(MAP)
    print(f'map: {len(scene.brushes)} brushes, {len(scene.patches)} patches, {len(scene.models)} models')

    world = defaultdict(lambda: {'pos': [], 'nrm': [], 'uv': [], 'alpha': [], 'idx': [], 'n': 0})

    def add(mat_name, pos, nrm, uv, idx, alpha=None):
        w = world[mat_name]
        w['pos'].append(pos)
        w['nrm'].append(nrm)
        w['uv'].append(uv)
        w['alpha'].append(np.ones(len(pos)) if alpha is None else alpha)
        w['idx'].append(idx + w['n'])
        w['n'] += len(pos)

    mat_info = {}

    def resolve(name):
        if name not in mat_info:
            mat_info[name] = make_material(glb, bank, db.material(name))
        return mat_info[name]

    # brushes
    for brush, xf in scene.brushes:
        for face, n, poly in brush_polygons(brush):
            mi, size = resolve(face['material'])
            if mi is None:
                continue
            uv = brush_face_uv(poly, n, face['tex'], size)
            m = len(poly)
            idx = np.array([(0, k, k + 1) for k in range(1, m - 1)], np.uint32)
            pos = poly @ xf[:3, :3].T + xf[:3, 3]
            nn = n @ xf[:3, :3].T
            nn /= np.linalg.norm(nn)
            add(face['material'], pos, np.tile(nn, (m, 1)), uv, idx)
    print(f'brushes done {time.time() - t0:.1f}s')

    # patches
    for patch, xf in scene.patches:
        mi, size = resolve(patch.material)
        if mi is None or len(patch.rows) < 2 or len(patch.rows[0]) < 2:
            continue
        P, T, A = patch_grid(patch)
        P = P @ xf[:3, :3].T + xf[:3, 3]
        pos, nrm, uv, alpha, idx = grid_mesh(P, T, A)
        uv = uv / np.array(size, float)
        add(patch.material, pos, nrm, uv, idx, alpha)
    print(f'patches done {time.time() - t0:.1f}s')

    # world meshes, one primitive per material
    prims = []
    tri_total = 0
    for name, w in world.items():
        mi, _ = mat_info[name]
        pos = to_gltf(np.concatenate(w['pos'])).astype(np.float32)
        nrm = to_gltf_dir(np.concatenate(w['nrm'])).astype(np.float32)
        uv = np.concatenate(w['uv']).astype(np.float32)
        idx = np.concatenate(w['idx']).astype(np.uint32).reshape(-1)
        tri_total += len(idx) // 3
        attrs = {
            'POSITION': glb.accessor(pos, 'VEC3', target=34962, minmax=True),
            'NORMAL': glb.normals(nrm),
            'TEXCOORD_0': glb.accessor(uv, 'VEC2', target=34962),
        }
        alpha = np.clip(np.concatenate(w['alpha']), 0, 1)
        if (alpha < 0.999).any() and glb.gltf['materials'][mi].get('alphaMode') in ('BLEND', 'MASK'):
            col = np.full((len(alpha), 4), 255, np.uint8)
            col[:, 3] = np.round(alpha * 255)
            attrs['COLOR_0'] = glb.accessor(col, 'VEC4', normalized=True, target=34962)
        prims.append({
            'attributes': attrs,
            'indices': glb.accessor(idx.astype(np.uint16) if pos.shape[0] < 65536 else idx, 'SCALAR', target=34963),
            'material': mi,
        })
    glb.node({'name': 'world', 'mesh': glb.mesh('world', prims)})
    print(f'world: {len(prims)} materials, {tri_total} tris')

    # models: one instanced node per unique model
    by_model = defaultdict(list)
    for name, m in scene.models:
        by_model[name].append(m)
    inst_tris = 0
    for name, mats in sorted(by_model.items()):
        try:
            surfs = xmodel.read_xmodel(db.raw, name, db.has_material)
        except Exception as e:
            print('  skip model', name, e)
            continue
        mprims = []
        for s in surfs:
            mi, _ = resolve(s['material'])
            if mi is None or not len(s['idx']):
                continue
            pos = to_gltf(s['pos']).astype(np.float32)
            nrm = to_gltf_dir(s['nrm']).astype(np.float32)
            attrs = {
                'POSITION': glb.accessor(pos, 'VEC3', target=34962, minmax=True),
                'NORMAL': glb.normals(nrm),
                'TEXCOORD_0': glb.accessor(s['uv'].astype(np.float32), 'VEC2', target=34962),
            }
            col = s['col']
            if (col != 255).any():
                attrs['COLOR_0'] = glb.accessor(col.astype(np.uint8), 'VEC4', normalized=True, target=34962)
            idx = s['idx'].reshape(-1)
            mprims.append({'attributes': attrs, 'material': mi,
                           'indices': glb.accessor(idx.astype(np.uint16) if len(pos) < 65536 else idx.astype(np.uint32),
                                                   'SCALAR', target=34963)})
            inst_tris += len(s['idx']) * len(mats)
        if not mprims:
            continue
        mesh = glb.mesh(name, mprims)
        T, Rq, S = [], [], []
        for m in mats:
            # convert the CoD transform into glTF space: A * M * A^-1 (A = axis swap + scale for translation)
            M3 = AXIS @ m[:3, :3] @ AXIS.T
            sc = np.cbrt(abs(np.linalg.det(M3)))
            Rm = M3 / sc
            T.append(to_gltf(m[:3, 3]))
            Rq.append(mat_to_quat(Rm))
            S.append([sc, sc, sc])
        glb.node({'name': name, 'mesh': mesh, 'extensions': {'EXT_mesh_gpu_instancing': {'attributes': {
            'TRANSLATION': glb.accessor(np.array(T, np.float32), 'VEC3'),
            'ROTATION': glb.accessor(np.array(Rq, np.float32), 'VEC4'),
            'SCALE': glb.accessor(np.array(S, np.float32), 'VEC3'),
        }}}})
    print(f'models: {len(by_model)} unique, {inst_tris} instanced tris')

    out_glb = os.path.join(OUT, glb_name + '.glb')
    glb.write(out_glb)
    print(f'glb: {os.path.getsize(out_glb) / 1e6:.1f} MB, {len(glb.gltf["images"])} textures')

    sky = next((db.material(n) for n in mat_info if db.material(n).techset == 'sky'), None)
    write_sky(db, sky.color_map if sky else None)
    write_spawns(scene)
    print(f'done in {time.time() - t0:.1f}s')


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def write_sky(db, image):
    sky_dir = os.path.join(OUT, 'sky')
    os.makedirs(sky_dir, exist_ok=True)
    d = db.image_bytes(image) if image else None
    if d is None:
        print('no sky material found, using cobra_ft')
        d = db.image_bytes('cobra_ft')
    print('sky:', image)
    fmt = d[4]
    w = struct.unpack_from('<H', d, 6)[0]
    # cubemap: six full-size faces (+X -X +Y -Y +Z -Z) at the end of the file
    fs = {assets.FMT_DXT1: w * w // 2, assets.FMT_DXT3: w * w, assets.FMT_DXT5: w * w,
          assets.FMT_ARGB32: w * w * 4, assets.FMT_RGB24: w * w * 3}[fmt]
    base = len(d) - 6 * fs
    for i in range(6):
        face = d[:28] + d[base + i * fs:base + (i + 1) * fs]
        face = face[:6] + struct.pack('<HH', w, w) + face[10:]
        img = assets.decode_iwi(face)
        img.convert('RGB').save(os.path.join(sky_dir, f'{i}.webp'), 'WEBP', quality=85)


def write_spawns(scene):
    spawns = []
    for kv, m in scene.entities:
        if kv.get('classname') in ('mp_global_intermission', 'mp_tdm_spawn', 'mp_dm_spawn'):
            p = to_gltf(m[:3, 3] + np.array([0, 0, 60.0]))
            yaw = float(kv.get('angles', '0 0 0').split()[1]) if 'angles' in kv else 0.0
            spawns.append({'type': kv['classname'], 'pos': [round(float(v), 3) for v in p], 'yaw': yaw})
    spawns.sort(key=lambda s: s['type'] != 'mp_global_intermission')
    with open(os.path.join(OUT, 'spawn.json'), 'w') as f:
        json.dump(spawns, f, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4])
