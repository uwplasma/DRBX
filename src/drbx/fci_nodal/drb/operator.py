"""Traceable state and options for the six-field owner operator."""
from ..options import WALL_LAW_NAMES
from dataclasses import dataclass
from typing import NamedTuple
import jax
from ..operators.perp.rhs_options import NodalWallData


@dataclass(frozen=True)
class Switches:
    em: str = 'off'
    flutter: str = 'off'
    wall: str = 'manufactured'
    limiter: str = 'off'

    def require_implemented(self):
        if self.limiter not in ('off','afc'):
            raise ValueError('limiter must be off or afc')
        if self.em not in ('off','on') or self.flutter not in ('off','on') or self.wall not in ('manufactured', *WALL_LAW_NAMES):
            raise NotImplementedError('Need em/flutter=off/on and wall=manufactured/conducting_wall/floating_wall')
        if self.flutter=='on' and self.em!='on':
            raise ValueError('flutter=on requires em=on')


@dataclass(frozen=True)
class Plan:
    ctx: object
    params: object
    q: dict
    B: object
    BT: object
    LN: object
    LD: object
    maps: dict
    chi: object
    mass_ratio: object
    dirichlet_mass_correction: object
    nu: object
    beta_hat: object
    opts: object
    qparams: tuple
    wall_ops: object = None
    afc: object = None
    runtime_options: object = None


jax.tree_util.register_dataclass(Plan, data_fields=[
    'ctx', 'params', 'q', 'B', 'BT', 'LN', 'LD', 'maps', 'chi', 'mass_ratio',
    'dirichlet_mass_correction', 'nu', 'beta_hat', 'wall_ops', 'afc'],
    meta_fields=['opts', 'qparams', 'runtime_options'])


class WallData(NamedTuple):
    p: NodalWallData
    g: object
    load_N: object
    load_D: object
    q_neumann: object
    polarization_source: object = None
    closure: object = None
    ampere_source: object = None
