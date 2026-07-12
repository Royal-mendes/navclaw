"""Small AerialClaw-inspired skill registry for ground navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class SkillRegistry:
    """Build the action allowlist for one observation."""

    def for_observation(
        self,
        candidate_ids: Sequence[str],
        allow_stop: bool = False,
        allow_look: bool = True,
    ) -> List[SkillDefinition]:
        skills: List[SkillDefinition] = []
        if candidate_ids:
            skills.append(
                SkillDefinition(
                    name="select_waypoint",
                    description=(
                        "Select exactly one waypoint ID visible in the current annotated image. "
                        "The deterministic controller maps the ID to a safe path."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": list(candidate_ids),
                            }
                        },
                        "required": ["candidate_id"],
                        "additionalProperties": False,
                    },
                )
            )
        if allow_look:
            skills.extend(
                [
                    SkillDefinition(
                        name="look_left_60",
                        description="Rotate left by 60 degrees to obtain a new observation.",
                        input_schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    ),
                    SkillDefinition(
                        name="look_right_60",
                        description="Rotate right by 60 degrees to obtain a new observation.",
                        input_schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    ),
                ]
            )
        if allow_stop:
            skills.append(
                SkillDefinition(
                    name="stop",
                    description=(
                        "Request episode termination only when the current RGB clearly shows "
                        "the task target. This skill is exposed only after lower-layer target "
                        "category, confidence, distance, view-angle, and repeated-observation "
                        "checks have passed. STOP is a request for final environment handling, "
                        "not an unsupported claim of success."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                )
            )
        return skills

    @staticmethod
    def prompt_catalog(skills: Sequence[SkillDefinition]) -> List[Dict[str, Any]]:
        return [skill.to_prompt_dict() for skill in skills]
