"""
GR1 posttrain modality config — derived from the pretrained checkpoint's processor_config.json.

The GR1 pretrain config uses 5 state/action groups (no legs/neck — arms-only manipulation),
sin/cos encoding for all state keys, and relative actions for arms+hands.
"""
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

gr1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view_bg_crop_pad_res256_freq20"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
        ],
        sin_cos_embedding_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
        ],
        action_configs=[
            # left_arm — relative
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm — relative
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand — relative
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand — relative
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist — absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Register directly under the "gr1" key so --embodiment-tag GR1 works
MODALITY_CONFIGS[EmbodimentTag.GR1.value] = gr1_config
