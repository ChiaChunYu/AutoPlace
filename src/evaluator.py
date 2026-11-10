"""The numbers a placement is judged by: exact HPWL (every net) and overflow (real cell sizes, no fillers)."""


def evaluate_placement(pos, wirelength, density, data):
    """pos: [>= num_mov, 2] cell centres on the device -> dict."""
    hpwl = wirelength.hpwl(pos).item()
    return {"hpwl_sites": hpwl, "hpwl_um": hpwl * data.unit_dbu / float(data.d.dbu),
            "overflow": density.exact_overflow(pos[:data.num_mov]).item()}
