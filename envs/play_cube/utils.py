from __future__ import annotations
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg, FileCfg
import carb
import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd
from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.sim.utils import (
    bind_visual_material,
    clone,
)
from omni.isaac.orbit.utils.assets import check_file_path


def _my_spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: FileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # check file path exists
    if not check_file_path(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage
        prim_utils.create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        carb.log_warn(f"A prim already exists at prim path: '{prim_path}'.")

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.define_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def my_spawn_from_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # spawn asset from the given usd file
    return _my_spawn_from_usd_file(
        prim_path, cfg.usd_path, cfg, translation, orientation
    )
