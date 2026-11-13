"""The placement problem as arrays, built from the parsed LEF + DEF.

Length unit inside the placer: one site width (the narrowest site used by a ROW), origin at the lower left of the
placement region (the bounding box of the rows). Cell positions are cell centres.
"""
import numpy as np

from utils.io_parser import orient_point, orient_size


class PlaceData(object):
    def __init__(self, lef, d):
        self.d = d
        self.name = d.design
        if not d.rows:
            raise ValueError("the DEF has no ROW: no placement region")
        sites = sorted(set(r[0] for r in d.rows))
        missing = [s for s in sites if s not in lef.sites]
        if missing:
            raise ValueError("ROW site(s) not in the LEF: %s" % ", ".join(missing))
        dbu = d.dbu
        self.unit_dbu = int(round(min(lef.sites[s][0] for s in sites) * dbu))        # one site width
        self.row_height_dbu = int(round(min(lef.sites[s][1] for s in sites) * dbu))

        # row rectangles (database units) and the placement region
        rows = []
        for site, x, y, nx, ny, sx, sy in d.rows:
            sw, sh = (int(round(v * dbu)) for v in lef.sites[site])
            rows.append((x, y, x + (nx - 1) * sx + sw, y + (ny - 1) * sy + sh))
        self.row_rects = np.array(rows, dtype=np.int64)
        self.region = (int(self.row_rects[:, 0].min()), int(self.row_rects[:, 1].min()),
                       int(self.row_rects[:, 2].max()), int(self.row_rects[:, 3].max()))
        u = float(self.unit_dbu)
        self.width = (self.region[2] - self.region[0]) / u
        self.height = (self.region[3] - self.region[1]) / u
        self.row_height = self.row_height_dbu / u

        # components
        names, comp_macro = d.macro_names, d.comp_macro_id
        unknown = [m for m in names if m not in lef.macros]
        if unknown:
            raise ValueError("%d macro(s) of the DEF are not in the LEF: %s" % (len(unknown), ", ".join(unknown[:5])))
        macro_size = np.array([[lef.macros[m].width, lef.macros[m].height] for m in names]).reshape(-1, 2) * dbu
        size_dbu = macro_size[comp_macro]
        # fixed = FIXED / COVER in the DEF, and every placed component that is not a core cell (pads, corner and
        # ring cells, endcaps, macro blocks): those stay where the DEF puts them
        core = np.array([lef.macros[m].cls.startswith("CORE") for m in names], dtype=bool)
        self.fixed = d.comp_fixed | (d.comp_placed & ~core[comp_macro])
        self.kept_in_place = int((self.fixed & ~d.comp_fixed).sum())
        self.mov_index = np.nonzero(~self.fixed)[0]                   # component index of every movable cell
        self.num_mov = len(self.mov_index)
        mov_id = np.full(len(comp_macro), self.num_mov, dtype=np.int64)
        mov_id[self.mov_index] = np.arange(self.num_mov)
        self.size = size_dbu[self.mov_index] / u                      # [num_mov, 2], orientation N
        self.mov_area = float(self.size.prod(1).sum())

        # fixed components and hard placement blockages: area no cell may use
        block = [tuple(b) for b in d.blockages]
        for i in np.nonzero(self.fixed)[0]:
            w, h = orient_size(size_dbu[i, 0], size_dbu[i, 1], d.comp_orient[i])
            x, y = d.comp_xy[i]
            block.append((x, y, x + w, y + h))
        self.block_rects = np.array(block, dtype=np.int64).reshape(-1, 4)

        # pins: movable pin = cell index + offset from the cell centre; fixed pin = index num_mov + its position
        origin = np.array(self.region[:2], dtype=np.float64)
        io = d.pin_comp < 0
        comp = np.where(io, 0, d.pin_comp)
        # pin centre from the lower left, looked up once per (macro, pin name) pair the netlist uses
        key = comp_macro[comp] * len(d.pin_names) + d.pin_name
        used = np.zeros(len(names) * len(d.pin_names), dtype=bool)
        used[key] = True
        pair = np.nonzero(used)[0]
        slot = np.zeros(len(used), dtype=np.int32)
        slot[pair] = np.arange(len(pair))
        pair_xy = np.full((len(pair), 2), np.nan)
        for k, (m, pin) in enumerate(zip(pair // len(d.pin_names), pair % len(d.pin_names))):
            xy_m = lef.macros[names[m]].pins.get(d.pin_names[pin])
            if xy_m is not None:
                pair_xy[k] = xy_m
        xy = pair_xy[slot[key]] * dbu
        bad = np.nonzero(np.isnan(xy[:, 0]) & ~io)[0]
        if len(bad):
            raise ValueError("pin %s of macro %s is not in the LEF" % (d.pin_names[d.pin_name[bad[0]]], names[comp_macro[comp[bad[0]]]]))
        pin_cell = np.where(io, self.num_mov, mov_id[comp])
        pin_off = (xy - size_dbu[comp] / 2) / u                       # movable: from the cell centre
        pin_off[io] = (d.io_xy[-1 - d.pin_comp[io]] - origin) / u
        for k in np.nonzero(~io & (pin_cell == self.num_mov))[0]:     # pins of fixed components: placed position
            c = comp[k]
            ox, oy = orient_point(xy[k, 0], xy[k, 1], size_dbu[c, 0], size_dbu[c, 1], d.comp_orient[c])
            pin_off[k] = (d.comp_xy[c] + (ox, oy) - origin) / u
        keep = ~np.isnan(pin_off[:, 0])                               # a design pin without a location is dropped
        pin_net = np.repeat(np.arange(len(d.net_name)), np.diff(d.net_start))
        self.unplaced_io_pins = int((~keep).sum())
        self.pin_cell, self.pin_off, self.pin_net = pin_cell[keep], pin_off[keep], pin_net[keep]
        self.num_nets = len(d.net_name)
        if not self.num_mov or not self.num_nets:
            raise ValueError("nothing to place: %d movable cells, %d nets (truncated DEF?)" % (self.num_mov, self.num_nets))
        self.num_pins = len(self.pin_cell)
        self.cell_pins = np.bincount(self.pin_cell, minlength=self.num_mov + 1)[:self.num_mov].astype(np.float64)

    # ------------------------------------------------------------------ free area
    def fixed_fraction(self, nbx, nby):
        """[nbx, nby] fraction of every bin that is NOT placeable: outside the rows, or under a fixed cell / blockage.

        Exact: the union of the rows minus the union of the blockers, on the grid of all rectangle edges.
        """
        x0, y0, x1, y1 = self.region
        rects = np.concatenate([self.row_rects, self.block_rects])
        xs = np.unique(np.clip(np.concatenate([rects[:, 0], rects[:, 2], [x0, x1]]), x0, x1))
        ys = np.unique(np.clip(np.concatenate([rects[:, 1], rects[:, 3], [y0, y1]]), y0, y1))

        def covered(r):
            cnt = np.zeros((len(xs), len(ys)), dtype=np.int32)
            if len(r):
                i0 = np.searchsorted(xs, np.clip(r[:, 0], x0, x1))
                i1 = np.searchsorted(xs, np.clip(r[:, 2], x0, x1))
                j0 = np.searchsorted(ys, np.clip(r[:, 1], y0, y1))
                j1 = np.searchsorted(ys, np.clip(r[:, 3], y0, y1))
                np.add.at(cnt, (i0, j0), 1)
                np.add.at(cnt, (i1, j1), 1)
                np.add.at(cnt, (i0, j1), -1)
                np.add.at(cnt, (i1, j0), -1)
            return (cnt.cumsum(0).cumsum(1) > 0)[:-1, :-1]

        free = (covered(self.row_rects) & ~covered(self.block_rects)).astype(np.float64)

        def overlap(edges, lo, hi, nb):                               # [intervals, bins] overlap length
            b = lo + (hi - lo) * np.arange(nb + 1) / float(nb)
            return np.clip(np.minimum(edges[1:, None], b[None, 1:]) - np.maximum(edges[:-1, None], b[None, :-1]), 0, None)

        cap = overlap(xs, x0, x1, nbx).T.dot(free).dot(overlap(ys, y0, y1, nby))
        bin_area = (x1 - x0) / float(nbx) * (y1 - y0) / float(nby)
        return np.clip(1.0 - cap / bin_area, 0.0, 1.0)

    # ------------------------------------------------------------------ positions <-> DEF
    def to_def_xy(self, pos):
        """Cell centres (placer units) -> lower-left corners in database units (rounded)."""
        ll = (np.asarray(pos, dtype=np.float64) - self.size / 2) * self.unit_dbu
        return np.rint(ll + np.array(self.region[:2], dtype=np.float64)).astype(np.int64)

    def placed_pos(self, placed=None):
        """Cell centres (placer units) of this design's movable cells as placed in a DEF (bounding-box centre).

        `placed`: another parsed DEF of the same netlist (default: the design's own DEF). The movable set is always
        this design's, whatever status the other DEF gives the components."""
        d, i = placed or self.d, self.mov_index
        if d.comp_name != self.d.comp_name:
            raise ValueError("the placed DEF has other components than the design")
        if not d.comp_placed[i].all():
            raise ValueError("%d movable cells of the DEF have no location" % int((~d.comp_placed[i]).sum()))
        swap = np.array([d.comp_orient[k] in ("W", "E", "FW", "FE") for k in i], dtype=bool)
        size = np.where(swap[:, None], self.size[:, ::-1], self.size)
        return (d.comp_xy[i] - np.array(self.region[:2])) / float(self.unit_dbu) + size / 2
