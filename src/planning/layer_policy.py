"""
Layer policy resolution for scene-adaptive execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


LAYER_POLICY_VERSION = "2026-05-07"
CANONICAL_LAYER_ORDER: Tuple[str, ...] = ("l1_macro", "l2_topo", "l3_urban")

SUPPORTED_SCENE_PROFILES = MappingProxyType({
    "urban_flat": ("l1_macro", "l3_urban"),
    "mountain_rural": ("l1_macro", "l2_topo"),
    "suburban_mixed": ("l1_macro", "l2_topo", "l3_urban"),
    "plain_sparse": ("l1_macro",),
})


class LayerPolicyError(ValueError):
    """Base error for layer policy resolution failures."""


class UnknownSceneProfileError(LayerPolicyError):
    """Raised when a scene profile is not supported."""


class MissingSceneProfileError(LayerPolicyError):
    """Raised when a scene profile is required but absent."""


class MissingRequiredInputError(LayerPolicyError):
    """Raised when a required input is unavailable in strict modes."""


@dataclass(frozen=True)
class DisabledLayerReason:
    """Structured explanation for why a layer was disabled."""

    reason_type: str
    detail: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "reason_type": self.reason_type,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LayerPolicy:
    """Resolved layer policy for a single scene profile."""

    scene_profile: str
    enabled_layers: Tuple[str, ...]
    disabled_layers: Tuple[Tuple[str, DisabledLayerReason], ...]
    policy_version: str = LAYER_POLICY_VERSION
    strict: bool = False
    benchmark: bool = False

    def is_enabled(self, layer_name: str) -> bool:
        return layer_name in self.enabled_layers

    def disabled_layers_dict(self) -> Dict[str, Dict[str, str]]:
        return {layer: reason.to_dict() for layer, reason in self.disabled_layers}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_profile": self.scene_profile,
            "enabled_layers": list(self.enabled_layers),
            "disabled_layers": self.disabled_layers_dict(),
            "policy_version": self.policy_version,
            "strict": self.strict,
            "benchmark": self.benchmark,
        }


class LayerPolicyResolver:
    """Resolve scene profiles into deterministic layer execution decisions."""

    def __init__(
        self,
        scene_defaults: Optional[Mapping[str, Sequence[str]]] = None,
        layer_order: Sequence[str] = CANONICAL_LAYER_ORDER,
        policy_version: str = LAYER_POLICY_VERSION,
        default_scene_profile: Optional[str] = None,
    ) -> None:
        self._scene_defaults = dict(scene_defaults or SUPPORTED_SCENE_PROFILES)
        self._layer_order = tuple(layer_order)
        self._policy_version = policy_version
        self._default_scene_profile = default_scene_profile

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LayerPolicyResolver":
        return cls(default_scene_profile=cls._extract_scene_profile_static(config))

    def resolve(
        self,
        scene_profile: Optional[str] = None,
        *,
        strict: bool = False,
        benchmark: bool = False,
        user_overrides: Optional[Mapping[str, bool]] = None,
        input_availability: Optional[Mapping[str, bool]] = None,
    ) -> LayerPolicy:
        profile = scene_profile or self._default_scene_profile
        if not profile:
            raise MissingSceneProfileError("scene.profile is required for layer policy resolution.")

        if profile not in self._scene_defaults:
            raise UnknownSceneProfileError(f"Unsupported scene profile: {profile}")

        default_enabled = set(self._scene_defaults[profile])
        overrides = dict(user_overrides or {})
        availability = dict(input_availability or {})

        enabled_layers = []
        disabled_layers = []

        for layer_name in self._layer_order:
            enabled = layer_name in default_enabled
            if layer_name in overrides:
                enabled = bool(overrides[layer_name])
                if not enabled:
                    disabled_layers.append(
                        (layer_name, DisabledLayerReason("user_override", "disabled by explicit override"))
                    )
                    continue

            if enabled:
                if layer_name in availability and not bool(availability[layer_name]):
                    if strict or benchmark:
                        raise MissingRequiredInputError(
                            f"Required input for {layer_name} is unavailable in strict policy resolution."
                        )
                    disabled_layers.append(
                        (layer_name, DisabledLayerReason("missing_input", "required input is unavailable"))
                    )
                    continue
                enabled_layers.append(layer_name)
            else:
                disabled_layers.append(
                    (layer_name, DisabledLayerReason("scene_policy", "disabled by scene profile"))
                )

        return LayerPolicy(
            scene_profile=profile,
            enabled_layers=tuple(enabled_layers),
            disabled_layers=tuple(disabled_layers),
            policy_version=self._policy_version,
            strict=strict,
            benchmark=benchmark,
        )

    @staticmethod
    def _extract_scene_profile_static(config: Mapping[str, Any]) -> str:
        scene = config.get("scene", {})
        if isinstance(scene, Mapping):
            profile = scene.get("profile")
            if profile is not None:
                return str(profile)
        return ""


def configured_layer_overrides(config: Mapping[str, Any]) -> Dict[str, bool]:
    """Extract explicit layer enable flags from config as user overrides."""
    layers = config.get("layers", {})
    if not isinstance(layers, Mapping):
        return {}

    overrides: Dict[str, bool] = {}
    for layer_name in CANONICAL_LAYER_ORDER:
        layer_cfg = layers.get(layer_name, {})
        if isinstance(layer_cfg, Mapping) and layer_cfg.get("enabled") is False:
            overrides[layer_name] = False
    return overrides


def required_input_availability(config: Mapping[str, Any]) -> Dict[str, bool]:
    """Report availability for blocking inputs used by the runtime policy."""
    layers = config.get("layers", {})
    if not isinstance(layers, Mapping):
        return {}

    def _path_exists(path_value: Optional[str]) -> Optional[bool]:
        if not path_value:
            return None
        return Path(str(path_value)).exists()

    availability: Dict[str, bool] = {}

    l1_cfg = layers.get("l1_macro", {})
    if isinstance(l1_cfg, Mapping):
        tle_cfg = l1_cfg.get("tle", {})
        tle_path = None
        if isinstance(tle_cfg, Mapping):
            tle_path = tle_cfg.get("file")
        availability_value = _path_exists(tle_path or l1_cfg.get("tle_file"))
        if availability_value is not None:
            availability["l1_macro"] = availability_value

    l2_cfg = layers.get("l2_topo", {})
    if isinstance(l2_cfg, Mapping):
        availability_value = _path_exists(l2_cfg.get("dem_file"))
        if availability_value is not None:
            availability["l2_topo"] = availability_value

    l3_cfg = layers.get("l3_urban", {})
    if isinstance(l3_cfg, Mapping):
        availability_value = _path_exists(
            l3_cfg.get("tile_cache_root") or l3_cfg.get("data_dir") or l3_cfg.get("tiles_dir")
        )
        if availability_value is not None:
            availability["l3_urban"] = availability_value

    return availability


def enabled_layer_config(config: Mapping[str, Any], enabled_layers: Sequence[str]) -> Dict[str, Any]:
    """Return a shallow config copy containing only enabled layer configs."""
    out = dict(config)
    layers = config.get("layers", {})
    if not isinstance(layers, Mapping):
        return out

    out["layers"] = {
        layer_name: layers[layer_name]
        for layer_name in enabled_layers
        if layer_name in layers
    }
    return out


def infer_scene_profile(config: Mapping[str, Any]) -> Optional[str]:
    """Infer a legacy scene profile from layer enablement when no explicit profile exists."""
    explicit_profile = LayerPolicyResolver._extract_scene_profile_static(config)
    if explicit_profile:
        return explicit_profile

    layers = config.get("layers", {})
    if not isinstance(layers, Mapping) or not layers:
        enabled_layers = CANONICAL_LAYER_ORDER
    else:
        enabled_layers = tuple(
            layer_name
            for layer_name in CANONICAL_LAYER_ORDER
            if not (isinstance(layers.get(layer_name), Mapping) and layers.get(layer_name, {}).get("enabled") is False)
        )

    for profile_name, default_layers in SUPPORTED_SCENE_PROFILES.items():
        if tuple(default_layers) == tuple(enabled_layers):
            return profile_name
    return None


def resolve_layer_policy(
    config: Mapping[str, Any],
    *,
    strict: bool = False,
    benchmark: bool = False,
    input_availability: Optional[Mapping[str, bool]] = None,
) -> LayerPolicy:
    """Resolve layer policy from a runtime config dict."""
    explicit_profile = LayerPolicyResolver._extract_scene_profile_static(config)
    if explicit_profile:
        scene_profile = explicit_profile
    elif strict:
        scene_profile = None
    else:
        scene_profile = infer_scene_profile(config)

    resolver = LayerPolicyResolver.from_config(config)
    overrides = configured_layer_overrides(config)
    return resolver.resolve(
        scene_profile=scene_profile,
        strict=strict,
        benchmark=benchmark,
        user_overrides=overrides or None,
        input_availability=input_availability or required_input_availability(config),
    )


def layer_policy_metadata(policy: LayerPolicy) -> Dict[str, Any]:
    """Serialize the resolved layer policy into manifest-friendly metadata."""
    return {
        "scene_profile": policy.scene_profile,
        "enabled_layers": list(policy.enabled_layers),
        "disabled_layers": policy.disabled_layers_dict(),
        "layer_policy_version": policy.policy_version,
        "layer_policy": policy.to_dict(),
    }
