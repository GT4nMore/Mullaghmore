"""CoD4 material + image lookup.

Materials are the binary files in raw/materials. Images (.iwi) come from raw/images,
the map's own .iwd and the stock main/iw_*.iwd archives (later archives win, like the game).
"""
import io
import os
import struct
import zipfile

import numpy as np
from PIL import Image


def _cstr(d, o):
    e = d.index(b'\0', o)
    return d[o:e].decode('latin1')


class Material:
    def __init__(self, name, techset, color_map, ref_image):
        self.name = name
        self.techset = techset
        self.color_map = color_map or ref_image
        self.is_tool = techset in ('tools', '') or techset.startswith('wc_tools')
        parts = techset.split('_')
        self.blend = any(p.startswith('b0') for p in parts)
        self.additive = any(p.startswith('a0') for p in parts)

    def __repr__(self):
        return f'Material({self.name}, {self.techset}, {self.color_map})'


class AssetDB:
    def __init__(self, game_dir, map_iwds=()):
        self.raw = os.path.join(game_dir, 'raw')
        self.material_names = set(os.listdir(os.path.join(self.raw, 'materials')))
        self._materials = {}
        self.images = {}
        main = os.path.join(game_dir, 'main')
        iwds = sorted(f for f in os.listdir(main) if f.endswith('.iwd'))
        sources = [os.path.join(main, f) for f in iwds] + list(map_iwds)
        for src in sources:                      # later sources override earlier ones
            z = zipfile.ZipFile(src)
            for n in z.namelist():
                low = n.lower()
                if low.startswith('images/') and low.endswith('.iwi'):
                    self.images[os.path.basename(low)[:-4]] = (src, n)
        raw_images = os.path.join(self.raw, 'images')
        for f in os.listdir(raw_images):          # loose raw images win (newest custom work)
            if f.lower().endswith('.iwi'):
                self.images[f.lower()[:-4]] = (None, os.path.join(raw_images, f))
        self._zips = {}

    def has_material(self, name):
        return name in self.material_names

    def material(self, name):
        if name in self._materials:
            return self._materials[name]
        path = os.path.join(self.raw, 'materials', name)
        if not os.path.exists(path):
            m = Material(name, '', None, None)
        else:
            d = open(path, 'rb').read()
            ref_image = _cstr(d, struct.unpack_from('<I', d, 4)[0])
            techset = _cstr(d, struct.unpack_from('<I', d, 0x34)[0])
            color_map = None
            key = d.find(b'\0colorMap\0')
            if key >= 0:
                key_ofs = struct.pack('<I', key + 1)
                # texture table entries: (name offset, sampler/semantic u32, image offset)
                i = d.find(key_ofs, 0, 0x100)
                while i >= 0 and i % 4:
                    i = d.find(key_ofs, i + 1, 0x100)
                if i >= 0:
                    color_map = _cstr(d, struct.unpack_from('<I', d, i + 8)[0])
            m = Material(name, techset, color_map, ref_image)
        self._materials[name] = m
        return m

    def image_bytes(self, name):
        key = name.lower()
        if key not in self.images:
            return None
        src, entry = self.images[key]
        if src is None:
            return open(entry, 'rb').read()
        if src not in self._zips:
            self._zips[src] = zipfile.ZipFile(src)
        return self._zips[src].read(entry)

    def image(self, name):
        d = self.image_bytes(name)
        return decode_iwi(d) if d else None


# --- IWI (version 6) -------------------------------------------------------

FMT_ARGB32, FMT_RGB24, FMT_GA16, FMT_A8, FMT_L8 = 1, 2, 3, 4, 6
FMT_DXT1, FMT_DXT3, FMT_DXT5 = 11, 12, 13


def _dds(fourcc, w, h, data):
    hdr = struct.pack('<4sI', b'DDS ', 124)
    hdr += struct.pack('<6I', 0x1007 | 0x80000, h, w, len(data), 0, 1)
    hdr += b'\0' * 44
    hdr += struct.pack('<2I4s5I', 32, 4, fourcc, 0, 0, 0, 0, 0)
    hdr += struct.pack('<5I', 0x1000, 0, 0, 0, 0)
    return hdr + data


def decode_iwi(d):
    if d[:3] != b'IWi':
        raise ValueError('not an IWI file')
    fmt = d[4]
    w, h = struct.unpack_from('<HH', d, 6)
    # mip levels are stored smallest first; the full-size level is at the end of the file
    if fmt in (FMT_DXT1, FMT_DXT3, FMT_DXT5):
        bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
        size = bw * bh * (8 if fmt == FMT_DXT1 else 16)
        data = d[len(d) - size:]
        fourcc = {FMT_DXT1: b'DXT1', FMT_DXT3: b'DXT3', FMT_DXT5: b'DXT5'}[fmt]
        img = Image.open(io.BytesIO(_dds(fourcc, w, h, data)))
        img.load()
        return img.convert('RGBA')
    bpp = {FMT_ARGB32: 4, FMT_RGB24: 3, FMT_GA16: 2, FMT_A8: 1, FMT_L8: 1}.get(fmt)
    if bpp is None:
        raise ValueError(f'unsupported IWI format {fmt}')
    a = np.frombuffer(d[len(d) - w * h * bpp:], np.uint8).reshape(h, w, bpp)
    if fmt == FMT_ARGB32:      # stored as BGRA
        rgba = a[..., [2, 1, 0, 3]]
    elif fmt == FMT_RGB24:     # stored as BGR
        rgba = np.dstack([a[..., [2, 1, 0]], np.full((h, w), 255, np.uint8)])
    elif fmt == FMT_GA16:      # luminance + alpha
        rgba = np.dstack([a[..., 0], a[..., 0], a[..., 0], a[..., 1]])
    elif fmt == FMT_A8:
        rgba = np.dstack([np.full((h, w, 3), 255, np.uint8), a[..., 0]])
    else:
        rgba = np.dstack([a[..., 0]] * 3 + [np.full((h, w), 255, np.uint8)])
    return Image.fromarray(np.ascontiguousarray(rgba), 'RGBA')
