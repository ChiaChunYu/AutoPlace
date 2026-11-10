"""LEF / DEF reader and DEF writer: no PDK-specific rule, any LEF 5.x / DEF 5.x design.

The DEF is read and written by the compiled module cpp_to_py/def_io when it is built (several times faster on a
100 MB DEF); the Python code below gives the same result and is what the compiled reader is tested against.

Only what a global placer needs is read:
  LEF  SITE sizes; per MACRO its size, class and the centre of every pin (bounding box of the pin's shapes)
  DEF  units, DIEAREA, ROWs, COMPONENTS (name, macro, placement status), PINS (location), NETS (connections),
       BLOCKAGES (placement)
Everything else in the DEF (tracks, vias, special nets, ...) is kept as text and written back unchanged.
"""
import re
from collections import namedtuple

import numpy as np

from cpp_to_py import def_io

Macro = namedtuple("Macro", "width height cls pins")              # microns; pins: {name: (x, y)} from the lower left
Lef = namedtuple("Lef", "sites macros")                           # sites: {name: (width, height)} microns

def parse_lef(paths):
    """Read one or more LEF files (technology and cell LEFs in any order). One pass over the tokens."""
    sites, macros = {}, {}
    for path in paths:
        with open(path) as f:
            t = re.sub(r"#[^\n]*", "", f.read()).replace(";", " ; ").split()
        i, n = 0, len(t)

        def numbers(j):                                  # the numbers of the statement starting at t[j], and its end
            if t[j] == "MASK":
                j += 2
            k = j
            while k < n and t[k] != ";":
                k += 1
            return [float(v) for v in t[j:k] if v[0] in "-.0123456789"], k

        while i < n:                                     # every branch leaves i on the name after its END
            if t[i] == "PROPERTYDEFINITIONS":            # its lines may start with MACRO / SITE
                while i < n and not (t[i] == "END" and t[i + 1] == "PROPERTYDEFINITIONS"):
                    i += 1
                i += 1
            elif t[i] == "SITE" and i + 2 < n and t[i + 2] != ";":
                name = t[i + 1]
                while not (t[i] == "END" and t[i + 1] == name):
                    if t[i] == "SIZE":
                        sites[name] = (float(t[i + 1]), float(t[i + 3]))
                    i += 1
                i += 1
            elif t[i] == "MACRO":
                name, size, origin, cls, pins = t[i + 1], None, (0.0, 0.0), "CORE", {}
                i += 2
                while not (t[i] == "END" and t[i + 1] == name):
                    if t[i] == "SIZE":
                        size = (float(t[i + 1]), float(t[i + 3]))
                    elif t[i] == "ORIGIN":
                        origin = (float(t[i + 1]), float(t[i + 2]))
                    elif t[i] == "CLASS":
                        j = t.index(";", i)
                        cls = " ".join(t[i + 1:j]).upper()
                    elif t[i] == "OBS":                  # ends at its own END
                        while t[i] != "END":
                            i += 1
                    elif t[i] == "PIN":
                        pin, xs, ys, in_port = t[i + 1], [], [], False
                        i += 2
                        while in_port or not (t[i] == "END" and t[i + 1] == pin):
                            if t[i] == "PORT":
                                in_port = True
                            elif t[i] == "END":
                                in_port = False
                            elif t[i] in ("RECT", "POLYGON"):
                                v, i = numbers(i + 1)
                                xs += v[0::2]
                                ys += v[1::2]
                            i += 1
                        if xs:      # shapes are relative to the origin; the lower left of the cell is at (-ox, -oy)
                            pins[pin] = ((min(xs) + max(xs)) / 2 + origin[0], (min(ys) + max(ys)) / 2 + origin[1])
                        i += 1
                    i += 1
                if size:
                    macros[name] = Macro(size[0], size[1], cls, pins)
                i += 1
            i += 1
    return Lef(sites, macros)


class Def(object):
    """The parsed DEF. Coordinates are DEF database units (integers)."""

    def __init__(self):
        self.path = ""
        self.text = None                 # the file content (Python reader only: its writer edits the text)
        self.design = ""
        self.dbu = 1000                  # database units per micron
        self.die = (0, 0, 0, 0)
        self.rows = []                   # (site, x, y, num_x, num_y, step_x, step_y)
        self.comp_span = (0, 0)          # Python reader: [start, end) of the COMPONENTS statements in text
        self.comp_stmt = None            # compiled reader: int64 [n, 2], byte range of every component statement
        self.comp_name = []
        self.macro_names = []            # the distinct macros
        self.comp_macro_id = None        # int64 [n]: index into macro_names
        self.comp_fixed = None           # bool array: FIXED or COVER
        self.comp_placed = None          # bool array: has a location (PLACED / FIXED / COVER)
        self.comp_xy = None              # int64 [n, 2], lower left
        self.comp_orient = []
        self.io_name = []
        self.io_xy = None                # float64 [n, 2], centre of the pin shape; nan = no location
        self.net_name = []
        self.net_start = None            # int64 [nets + 1]: pins of net i are [net_start[i], net_start[i + 1])
        self.pin_comp = None             # int64 [pins]: component index, or -1 - io index for a design pin
        self.pin_names = [""]            # the distinct pin names; "" = a design pin
        self.pin_name = None             # int64 [pins]: index into pin_names
        self.blockages = []              # hard placement blockages (x0, y0, x1, y1)
        self.soft_blockages = 0          # SOFT / PARTIAL placement blockages (ignored)


def _section(text, name):
    """[start, end) of the statements between '<name> <count> ;' and 'END <name>' (None if absent)."""
    head = text.find("\n%s " % name)
    if head < 0:
        return None
    start = text.index(";", head) + 1
    end = text.find("\nEND %s" % name, start)
    return start, (end if end >= 0 else len(text))


_NET_PIN = re.compile(r"\(\s*(\S+)\s+(\S+)[^)]*\)")         # ( component pin ) or ( PIN name ), extras ignored
_NET_END = re.compile(r"\)\s*\+\s")                          # the first '+' after the connections


# a point (x, y) of a shape, relative to the placement point of a design pin, in each orientation
_PIN_ORIENT = {"N": lambda x, y: (x, y), "S": lambda x, y: (-x, -y), "W": lambda x, y: (-y, x),
               "E": lambda x, y: (y, -x), "FN": lambda x, y: (-x, y), "FS": lambda x, y: (x, -y),
               "FW": lambda x, y: (y, x), "FE": lambda x, y: (-y, -x)}


def orient_size(w, h, orient):
    """Bounding-box size of a w x h cell placed with `orient`."""
    return (h, w) if orient in ("W", "E", "FW", "FE") else (w, h)


def orient_point(x, y, w, h, orient):
    """Point (x, y) of a w x h cell (from its lower left, orientation N) after placing it with `orient`."""
    return {"N": (x, y), "S": (w - x, h - y), "FN": (w - x, y), "FS": (x, h - y),
            "W": (h - y, x), "E": (y, w - x), "FW": (y, x), "FE": (h - y, w - x)}[orient]


def parse_def(path, compiled=True):
    """Read a DEF. compiled=False forces the Python reader."""
    d = Def()
    d.path = path
    if compiled and def_io:
        for key, value in def_io.parse_def(path).items():
            setattr(d, key, value)
        d.rows = [(site,) + tuple(int(v) for v in r) for site, r in zip(d.row_site, d.rows)]
        d.comp_fixed, d.comp_placed = d.comp_fixed.astype(bool), d.comp_placed.astype(bool)
        d.blockages = [tuple(int(v) for v in b) for b in d.blockages]
        return d
    with open(path) as f:
        text = d.text = f.read()
    span = _section(text, "COMPONENTS")
    head = text[:span[0]] if span else text          # everything read by pattern comes before COMPONENTS
    m = re.search(r"^[ \t]*DESIGN[ \t]+(\S+)", head, re.M)
    d.design = m.group(1) if m else ""
    m = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)", head)
    d.dbu = int(m.group(1)) if m else 1000
    m = re.search(r"^[ \t]*DIEAREA([^;]*);", head, re.M)
    if m:
        v = [int(t) for t in re.findall(r"-?\d+", m.group(1))]
        d.die = (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))
    for m in re.finditer(r"^[ \t]*ROW\s+\S+\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+\S+"
                         r"(?:\s+DO\s+(\d+)\s+BY\s+(\d+)(?:\s+STEP\s+(-?\d+)\s+(-?\d+))?)?", head, re.M):
        g = m.groups()
        d.rows.append((g[0], int(g[1]), int(g[2]), int(g[3] or 1), int(g[4] or 1), int(g[5] or 0), int(g[6] or 0)))

    fixed, placed, xy, macro, macro_id = [], [], [], [], {}
    if span:
        d.comp_span = span
        for stmt in text[span[0]:span[1]].split(";"):
            t = stmt.split()
            if len(t) < 3 or t[0] != "-":
                continue
            d.comp_name.append(t[1])
            macro.append(macro_id.setdefault(t[2], len(macro_id)))
            status, x, y, orient = "", 0, 0, "N"
            for i in range(3, len(t) - 6):
                if t[i] == "+" and t[i + 1] in ("PLACED", "FIXED", "COVER"):
                    status, x, y, orient = t[i + 1], int(t[i + 3]), int(t[i + 4]), t[i + 6]
                    break
            fixed.append(status in ("FIXED", "COVER"))
            placed.append(status != "")
            xy.append((x, y))
            d.comp_orient.append(orient)
    d.macro_names, d.comp_macro_id = list(macro_id), np.array(macro, dtype=np.int64)
    d.comp_fixed = np.array(fixed, dtype=bool)
    d.comp_placed = np.array(placed, dtype=bool)
    d.comp_xy = np.array(xy, dtype=np.int64).reshape(-1, 2)

    span = _section(text, "PINS")
    io_xy = []
    if span:
        for stmt in text[span[0]:span[1]].split(";"):
            t = stmt.split()
            if len(t) < 2 or t[0] != "-":
                continue
            rect, place = None, None
            for i in range(2, len(t)):
                if t[i] == "LAYER" and rect is None:
                    j = t.index("(", i)
                    rect = (int(t[j + 1]), int(t[j + 2]), int(t[j + 5]), int(t[j + 6]))
                elif t[i] in ("PLACED", "FIXED", "COVER") and t[i - 1] == "+" and place is None:
                    place = (int(t[i + 2]), int(t[i + 3]), t[i + 5])
            d.io_name.append(t[1])
            if place is None:
                io_xy.append((np.nan, np.nan))
            else:
                cx, cy = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0) if rect else (0.0, 0.0)
                ox, oy = _PIN_ORIENT[place[2]](cx, cy)
                io_xy.append((place[0] + ox, place[1] + oy))
    d.io_xy = np.array(io_xy, dtype=np.float64).reshape(-1, 2)

    span = _section(text, "NETS")
    net_start, pin_comp, pin_name, name_id = [0], [], [], {"": 0}
    if span:
        comp_id = {n: i for i, n in enumerate(d.comp_name)}
        io_id = {n: i for i, n in enumerate(d.io_name)}
        for stmt in text[span[0]:span[1]].split(";"):
            t = stmt.split(None, 2)
            if len(t) < 2 or t[0] != "-":
                continue
            if "+" in stmt:                                     # attributes (and routing) follow the connections
                end = _NET_END.search(stmt)
                stmt = stmt[:end.start() + 1] if end else stmt
            for a, b in _NET_PIN.findall(stmt):
                if a == "PIN":
                    pin_comp.append(-1 - io_id[b])
                    pin_name.append(0)
                else:
                    pin_comp.append(comp_id[a])
                    pin_name.append(name_id.setdefault(b, len(name_id)))
            d.net_name.append(t[1])
            net_start.append(len(pin_comp))
    d.pin_names = list(name_id)
    d.pin_name = np.array(pin_name, dtype=np.int64)
    d.net_start = np.array(net_start, dtype=np.int64)
    d.pin_comp = np.array(pin_comp, dtype=np.int64)

    span = _section(text, "BLOCKAGES")
    if span:
        for stmt in text[span[0]:span[1]].split(";"):
            t = stmt.split()
            if len(t) < 2 or t[0] != "-" or t[1] != "PLACEMENT":
                continue
            if "SOFT" in t or "PARTIAL" in t:
                d.soft_blockages += 1
                continue
            for i in range(2, len(t) - 8):
                if t[i] == "RECT":
                    d.blockages.append((int(t[i + 2]), int(t[i + 3]), int(t[i + 6]), int(t[i + 7])))
    return d


def write_def(d, path, comp_index, xy, orient="N"):
    """Write the DEF back with the components `comp_index` PLACED at `xy` (lower left, database units).

    Every other statement, and every other attribute of a moved component, is copied from the input.
    """
    if d.text is None:
        return def_io.write_def(d.path, path, d.comp_stmt, np.ascontiguousarray(comp_index, dtype=np.int64),
                                np.ascontiguousarray(xy, dtype=np.int64), orient)
    new_xy = dict(zip((int(i) for i in comp_index), ((int(x), int(y)) for x, y in xy)))
    start, end = d.comp_span
    out, k = [], 0
    for stmt in d.text[start:end].split(";"):
        t = stmt.split()
        if len(t) < 3 or t[0] != "-":
            if stmt.strip():
                out.append(stmt + ";")
            continue
        if k in new_xy:
            keep, i = [], 3
            while i < len(t):                                   # drop the old placement status, keep the rest
                if t[i] == "+" and i + 1 < len(t) and t[i + 1] == "UNPLACED":
                    i += 2
                elif t[i] == "+" and i + 1 < len(t) and t[i + 1] in ("PLACED", "FIXED", "COVER"):
                    i += 7
                else:
                    keep.append(t[i])
                    i += 1
            x, y = new_xy[k]
            out.append("\n- %s %s%s + PLACED ( %d %d ) %s ;" % (t[1], t[2], (" " + " ".join(keep)) if keep else "",
                                                               x, y, orient))
        else:
            out.append(stmt + ";")
        k += 1
    with open(path, "w") as f:
        f.write(d.text[:start])
        f.write("".join(out))
        f.write("\n")
        f.write(d.text[end:])
