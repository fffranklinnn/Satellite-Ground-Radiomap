from __future__ import annotations

import pytest

from src.planning.layer_policy import (
    LAYER_POLICY_VERSION,
    LayerPolicyResolver,
    MissingRequiredInputError,
    MissingSceneProfileError,
    SUPPORTED_SCENE_PROFILES,
    resolve_layer_policy,
    UnknownSceneProfileError,
)


def test_supported_profiles_are_explicit():
    assert SUPPORTED_SCENE_PROFILES["urban_flat"] == ("l1_macro", "l3_urban")
    assert SUPPORTED_SCENE_PROFILES["mountain_rural"] == ("l1_macro", "l2_topo")
    assert SUPPORTED_SCENE_PROFILES["suburban_mixed"] == ("l1_macro", "l2_topo", "l3_urban")
    assert SUPPORTED_SCENE_PROFILES["plain_sparse"] == ("l1_macro",)


def test_unknown_profile_raises():
    resolver = LayerPolicyResolver()
    with pytest.raises(UnknownSceneProfileError):
        resolver.resolve(scene_profile="unknown_scene")


def test_missing_profile_raises():
    resolver = LayerPolicyResolver()
    with pytest.raises(MissingSceneProfileError):
        resolver.resolve()


def test_from_config_preserves_scene_profile():
    resolver = LayerPolicyResolver.from_config({"scene": {"profile": "plain_sparse"}})
    policy = resolver.resolve()
    assert policy.scene_profile == "plain_sparse"
    assert policy.enabled_layers == ("l1_macro",)


def test_non_strict_legacy_config_infers_scene_profile():
    policy = resolve_layer_policy({"layers": {}})
    assert policy.scene_profile == "suburban_mixed"
    assert policy.enabled_layers == ("l1_macro", "l2_topo", "l3_urban")


def test_strict_legacy_config_requires_explicit_profile():
    with pytest.raises(MissingSceneProfileError):
        resolve_layer_policy({"layers": {}}, strict=True)


def test_deterministic_output_and_policy_version():
    resolver = LayerPolicyResolver()
    policy1 = resolver.resolve(scene_profile="urban_flat")
    policy2 = resolver.resolve(scene_profile="urban_flat")

    assert policy1.to_dict() == policy2.to_dict()
    assert policy1.policy_version == LAYER_POLICY_VERSION
    assert policy1.enabled_layers == ("l1_macro", "l3_urban")
    assert policy1.disabled_layers_dict()["l2_topo"]["reason_type"] == "scene_policy"


def test_user_override_and_missing_input_reasons():
    resolver = LayerPolicyResolver()

    policy = resolver.resolve(
        scene_profile="suburban_mixed",
        user_overrides={"l2_topo": False},
    )
    assert policy.disabled_layers_dict()["l2_topo"] == {
        "reason_type": "user_override",
        "detail": "disabled by explicit override",
    }

    policy = resolver.resolve(
        scene_profile="urban_flat",
        input_availability={"l3_urban": False},
    )
    assert policy.disabled_layers_dict()["l3_urban"]["reason_type"] == "missing_input"


def test_strict_missing_input_raises():
    resolver = LayerPolicyResolver()
    with pytest.raises(MissingRequiredInputError):
        resolver.resolve(
            scene_profile="urban_flat",
            strict=True,
            input_availability={"l3_urban": False},
        )
