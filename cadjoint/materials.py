"""A small catalogue of real engineering materials, in SI units.

Every entry is a factory returning a fresh :class:`~cadjoint.render.material.Material`
carrying both a plausible appearance and the physical properties the FEM layer
solves with — density, thermal conductivity, specific heat, Young's modulus,
Poisson ratio, linear thermal expansion and yield strength.  Factories (rather
than module-level singletons) so a scene can mark one instance's properties
``free`` without that leaking into every other scene in the process::

    from cadjoint.materials import aluminium_6061, copper_c11000

    heatsink = Box(size=[40e-3, 40e-3, 10e-3], material=aluminium_6061())
    slug = Cylinder(radius=6e-3, height=12e-3, material=copper_c11000())

Values are room-temperature (≈293–300 K) handbook figures for the stated
temper/grade, quoted to the precision the sources give.  They are engineering
reference values, not a certification: for a real design, substitute the
figures from your own supplier's datasheet.  Each factory's docstring cites
where its numbers come from.

Anisotropy caveat: FR-4 and single-crystal silicon are strongly anisotropic and
the solver is isotropic, so those entries pick one documented direction — read
the docstring before trusting a number.

Names use the British spelling ``aluminium``; ``aluminum_6061`` is provided as
an alias.
"""

from __future__ import annotations

from typing import Callable

from cadjoint.render.material import Material

__all__ = [
    "CATALOGUE",
    "aluminium_6061",
    "aluminum_6061",
    "catalogue",
    "copper_c11000",
    "fr4",
    "pla",
    "silicon",
    "steel_1018",
    "thermal_pad",
    "titanium_ti6al4v",
]


def aluminium_6061() -> Material:
    """Aluminium alloy 6061-T6 — the default structural/heatsink aluminium.

    Sources: ASM Aerospace Specification Metals Inc. datasheet for 6061-T6;
    MatWeb entry "Aluminum 6061-T6; 6061-T651".  Specific heat and thermal
    conductivity are the 25 °C values.

    Returns:
        A fresh Material with a brushed light-metal appearance.
    """
    return Material(
        name="aluminium_6061",
        color=[0.83, 0.84, 0.85],
        roughness=0.38,
        metallic=0.9,
        density=2700.0,
        conductivity=167.0,
        specific_heat=896.0,
        youngs_modulus=68.9e9,
        poisson_ratio=0.33,
        thermal_expansion=23.6e-6,
        yield_strength=276e6,
    )


#: US-spelling alias for :func:`aluminium_6061`.
aluminum_6061 = aluminium_6061


def copper_c11000() -> Material:
    """Copper C11000 (electrolytic tough pitch), annealed — heat spreaders.

    Sources: MatWeb entry "Copper, UNS C11000 (Electrolytic tough pitch),
    annealed"; CRC Handbook of Chemistry and Physics, 97th ed., thermal and
    calorific properties of the elements.  The yield strength is the soft
    (annealed, O60) temper; hard-drawn C11000 reaches ~310 MPa.

    Returns:
        A fresh Material with a polished copper appearance.
    """
    return Material(
        name="copper_c11000",
        color=[0.72, 0.45, 0.20],
        roughness=0.30,
        metallic=1.0,
        density=8940.0,
        conductivity=391.0,
        specific_heat=385.0,
        youngs_modulus=117e9,
        poisson_ratio=0.34,
        thermal_expansion=17.0e-6,
        yield_strength=69e6,
    )


def steel_1018() -> Material:
    """AISI 1018 mild carbon steel, cold drawn — general structural steel.

    Sources: MatWeb entry "AISI 1018 Steel, cold drawn"; ASM Handbook Vol. 1
    (Properties and Selection: Irons, Steels, and High-Performance Alloys).
    Hot-rolled 1018 yields nearer 220 MPa; the cold-drawn figure is quoted
    here because that is what bar stock ships as.

    Returns:
        A fresh Material with a dark machined-steel appearance.
    """
    return Material(
        name="steel_1018",
        color=[0.56, 0.57, 0.60],
        roughness=0.42,
        metallic=0.95,
        density=7870.0,
        conductivity=51.9,
        specific_heat=486.0,
        youngs_modulus=205e9,
        poisson_ratio=0.29,
        thermal_expansion=11.5e-6,
        yield_strength=370e6,
    )


def titanium_ti6al4v() -> Material:
    """Ti-6Al-4V (Grade 5), annealed — the standard aerospace titanium.

    Sources: MatWeb entry "Titanium Ti-6Al-4V (Grade 5), Annealed"; ASM
    Handbook Vol. 2 (Properties and Selection: Nonferrous Alloys).

    Returns:
        A fresh Material with a matte grey-metal appearance.
    """
    return Material(
        name="titanium_ti6al4v",
        color=[0.62, 0.61, 0.59],
        roughness=0.45,
        metallic=0.9,
        density=4430.0,
        conductivity=6.7,
        specific_heat=526.3,
        youngs_modulus=113.8e9,
        poisson_ratio=0.342,
        thermal_expansion=8.6e-6,
        yield_strength=880e6,
    )


def fr4() -> Material:
    """FR-4 woven-glass/epoxy laminate — the standard PCB substrate.

    FR-4 is orthotropic and the solver is isotropic, so this entry takes the
    **through-thickness** thermal conductivity (0.29 W/(m*K)) — the number that
    matters for getting heat out of a board — together with the **in-plane**
    elastic constants and in-plane CTE, which is what board-level warpage and
    stiffness depend on.  In-plane conductivity is roughly 0.8–1.0 W/(m*K) and
    the out-of-plane CTE above Tg is several times the in-plane value; model
    those explicitly if they drive your result.  ``yield_strength`` holds the
    ultimate tensile strength: FR-4 is brittle and has no yield plateau, so a
    "safety factor" against it is a factor against fracture.

    Sources: NEMA LI-1 / IPC-4101 FR-4 specifications; Isola 370HR and Rogers
    laminate datasheets; Sarvar, Poole & Witting, J. Electron. Mater. 19 (1990)
    on PCB thermal conductivity.

    Returns:
        A fresh Material with the familiar solder-mask green appearance.
    """
    return Material(
        name="fr4",
        color=[0.10, 0.35, 0.16],
        roughness=0.55,
        metallic=0.0,
        density=1850.0,
        conductivity=0.29,
        specific_heat=1100.0,
        youngs_modulus=24e9,
        poisson_ratio=0.136,
        thermal_expansion=14e-6,
        yield_strength=310e6,
    )


def silicon() -> Material:
    """Single-crystal silicon, intrinsic, at 300 K — dies and MEMS.

    Silicon is elastically anisotropic; this entry uses the ⟨100⟩ direction
    (E = 130 GPa, ν = 0.28), which is the wafer-normal direction for standard
    (100) wafers.  ⟨110⟩ and ⟨111⟩ reach 169 GPa and 188 GPa respectively.
    ``yield_strength`` holds a *practical fracture* strength: silicon is brittle
    with no yield point, and measured fracture strength is dominated by surface
    and edge damage, ranging from ~100 MPa for a sawn edge to several GPa for a
    pristine etched surface — 165 MPa is a conservative die-level figure.

    Sources: Hopcroft, Nix & Kenny, "What is the Young's Modulus of Silicon?",
    J. Microelectromech. Syst. 19(2):229–238, 2010; CRC Handbook of Chemistry
    and Physics, 97th ed. (density, conductivity, specific heat, CTE).

    Returns:
        A fresh Material with a dark specular die appearance.
    """
    return Material(
        name="silicon",
        color=[0.22, 0.23, 0.26],
        roughness=0.18,
        metallic=0.4,
        density=2329.0,
        conductivity=148.0,
        specific_heat=700.0,
        youngs_modulus=130e9,
        poisson_ratio=0.28,
        thermal_expansion=2.6e-6,
        yield_strength=165e6,
    )


def thermal_pad() -> Material:
    """A filled-silicone thermal interface pad (gap filler).

    Representative of a 3 W/(m*K) gap pad such as Henkel Bergquist Gap Pad
    3000S30.  Note ``poisson_ratio = 0.49``: elastomers are very nearly
    incompressible, and linear HEX8/TET4 elements *lock* volumetrically that
    close to 0.5 — a pad modelled with this value on a coarse linear mesh will
    read far too stiff.  Use TET10, or drop the ratio to ~0.45 deliberately, if
    the pad's compliance is what you are solving for; for a purely thermal
    study the ratio is irrelevant.  The yield figure is a nominal compressive
    proof stress, not a metallic yield point.

    Sources: Henkel Bergquist Gap Pad 3000S30 datasheet; Parker Chomerics
    THERM-A-GAP product data (mechanical figures).

    Returns:
        A fresh Material with a soft matte pink-grey appearance.
    """
    return Material(
        name="thermal_pad",
        color=[0.78, 0.55, 0.60],
        roughness=0.85,
        metallic=0.0,
        density=2500.0,
        conductivity=3.0,
        specific_heat=1000.0,
        youngs_modulus=1.0e6,
        poisson_ratio=0.49,
        thermal_expansion=200e-6,
        yield_strength=0.5e6,
    )


def pla() -> Material:
    """PLA (polylactic acid) — the default FDM printing plastic.

    Bulk / injection-moulded values.  A printed FDM part is anisotropic and
    porous: expect roughly 50–70 % of the quoted modulus and strength across
    layer lines, and lower still for sparse infill.

    Sources: NatureWorks Ingeo 3D850 and 4043D filament datasheets; Farah,
    Anderson & Langer, "Physical and mechanical properties of PLA, and their
    functions in widespread applications", Adv. Drug Deliv. Rev. 107:367–392,
    2016.

    Returns:
        A fresh Material with a matte off-white plastic appearance.
    """
    return Material(
        name="pla",
        color=[0.88, 0.87, 0.83],
        roughness=0.70,
        metallic=0.0,
        density=1240.0,
        conductivity=0.13,
        specific_heat=1800.0,
        youngs_modulus=3.5e9,
        poisson_ratio=0.36,
        thermal_expansion=68e-6,
        yield_strength=50e6,
    )


#: Name → factory for every catalogue entry (the US alias is not repeated).
CATALOGUE: dict[str, Callable[[], Material]] = {
    "aluminium_6061": aluminium_6061,
    "copper_c11000": copper_c11000,
    "steel_1018": steel_1018,
    "titanium_ti6al4v": titanium_ti6al4v,
    "fr4": fr4,
    "silicon": silicon,
    "thermal_pad": thermal_pad,
    "pla": pla,
}


def catalogue() -> list[Material]:
    """Every catalogue material, freshly constructed.

    Useful as the reference set for quantizing a blended property field onto
    named materials (see
    :func:`cadjoint.fem.properties.quantize_to_materials`).

    Returns:
        One new Material per catalogue entry, in declaration order.
    """
    return [factory() for factory in CATALOGUE.values()]
