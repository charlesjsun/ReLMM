from .dual_nav_grasp_envs import (
    LocobotNavigationGraspingDualPerturbationEnv)

try: 
    from .real_dual_nav_grasp_envs import (
        RealLocobotNavigationGraspingDualPerturbationEnv
    )
except ModuleNotFoundError:
    print("No Real Env")

