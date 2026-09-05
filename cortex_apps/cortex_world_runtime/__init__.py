"""
Cortex World Runtime: Persistent Multi-Agent Substrate, Skills & Real-Time Engine.
==================================================================================
Exposes:
  - SkillRegistry, SkillDefinition, SkillInvocationEvent, SkillSelector
  - FastWorldSubstrate (Clock 1, Clock 2, Clock 3)
  - WorldSnapshot U_v = <S_v, G_v, Z, H_v>
"""

import importlib


_LAZY_EXPORTS = {
    "SkillDefinition": ("skill_registry", "SkillDefinition"),
    "SkillInvocationEvent": ("skill_registry", "SkillInvocationEvent"),
    "SkillRegistry": ("skill_registry", "SkillRegistry"),
    "SkillSelector": ("skill_registry", "SkillSelector"),
    "FastWorldSubstrate": ("fast_world_substrate", "FastWorldSubstrate"),
    "WorldSnapshot": ("fast_world_substrate", "WorldSnapshot"),
    "EntityNode": ("fast_world_substrate", "EntityNode"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "SkillDefinition",
    "SkillInvocationEvent",
    "SkillRegistry",
    "SkillSelector",
    "FastWorldSubstrate",
    "WorldSnapshot",
    "EntityNode",
]
