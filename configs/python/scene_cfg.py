import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import DeformableObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from utils.config import config


@configclass
class SceneCfg(InteractiveSceneCfg):
    """
    Scene configuration
    """
    # Ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Lighting
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=config["scene"]["light"]["intensity"],
            color=tuple(config["scene"]["light"]["color"]),
        ),
    )
    # Robot config
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0,)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path,
            activate_contact_sensors=True,
        ),
    )
    # Conact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_hand",
        update_period=0.0,
        history_length=5,
        debug_vis=True,
    )
    # Cube
    # Object to grasp
    cuboid: sim_utils.MeshCuboidCfg = sim_utils.DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=tuple(config["scene"]["cube"]["size"]),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=config["scene"]["cube"]["rest_offset"],
                contact_offset=config["scene"]["cube"]["contact_offset"],
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=config["scene"]["cube"]["poissons_ratio"],
                youngs_modulus=config["scene"]["cube"]["youngs_modulus"],
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(config["scene"]["cube"]["diffuse_color"]),
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=config["scene"]["cube"]["pos"],
        ),
    )