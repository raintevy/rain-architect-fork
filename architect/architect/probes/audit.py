"""Static AST audit for generalization risks in robot programs.

Flags patterns that *correlate with* poor cross-scene generalization, so the
probe runner (and, in P5, the graduation gate) can demote programs that look
like they'd overfit before even running counterfactual probes:

  * ``hardcoded_pose``         — ``move_ee_to_pose({...numeric literal...})``
  * ``hardcoded_joint_config`` — joint-replay calls (``move_arm_to_config``,
                                 ``set_arm_joints``, ``rotate_wrist_to_config``)
                                 with all-numeric dict args
  * ``index_subscript``        — ``markers[0]`` / ``detected[1]`` etc., which
                                 break if detection order changes
  * ``absolute_after_relative``— ``move_ee_to_pose`` after a prior
                                 ``move_ee_to_rel_pose``, which can invalidate
                                 the relative offset's intent

The audit is intentionally conservative: it raises *suspicions*, not errors.
A flag should be read as "this line will probably regress under a perturbed
scene unless something else compensates." The probe runner combines audit
flags with empirical perturbation pass-rate to produce a final risk score.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"


# Functions whose return values represent *perceptual* observations of the
# scene. Variables assigned from these are tracked so the audit can
# distinguish "value derived from perception" from "literal".
_PERCEPTION_FUNCS = frozenset({
    "detect_markers",
    "detect_objects",
    "get_vqa_response",
    "verify_grasp",
    "get_current_ee_pose",
    "get_current_joints",
    "get_current_gripper_width",
    "get_current_odom",
    "query_robot_state",
    "get_scene_description",
})

# Functions whose first arg is interpreted as joint-space configuration.
# A purely numeric dict argument here is a demo replay.
_JOINT_CONFIG_FUNCS = frozenset({
    "move_arm_to_config",
    "set_arm_joints",
    "rotate_wrist_to_config",
})

# Names that programmatically suggest a *collection of detections* — these
# are the ones whose ``[INT]`` access is order-fragile. Coordinate-style
# names (``translation``, ``position``, ``xyz``) are deliberately excluded:
# indexing into a 3-vector by 0/1/2 is fine and common in the API.
_COLLECTION_HINTS = ("markers", "detections", "results", "grasps", "objects")


# ---------------------------------------------------------------------------
# RiskFlag
# ---------------------------------------------------------------------------


@dataclass
class RiskFlag:
    line: int
    kind: str
    severity: str
    message: str
    snippet: str  # truncated source for context


def severity_summary(flags: list[RiskFlag]) -> dict[str, int]:
    out = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 0, SEVERITY_LOW: 0}
    for f in flags:
        out[f.severity] = out.get(f.severity, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_program(code: str) -> list[RiskFlag]:
    """Static-audit ``code`` for generalization risks.

    Tolerates ``SyntaxError`` by returning a single high-severity flag — a
    program that doesn't parse can't generalize either.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            RiskFlag(
                line=exc.lineno or 0,
                kind="syntax_error",
                severity=SEVERITY_HIGH,
                message=f"Program does not parse: {exc.msg}",
                snippet="",
            )
        ]
    visitor = _AuditVisitor()
    visitor.visit(tree)
    return visitor.flags


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_name(call: ast.Call) -> str | None:
    """Return the simple name of the callee, or None if it's not ``Name(...)``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _is_numeric_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _signed_numeric(node: ast.AST) -> bool:
    """Match ``+0.5`` / ``-0.5`` / ``42`` — i.e., a numeric constant possibly
    wrapped in a unary +/- (which the parser keeps as ``UnaryOp``)."""
    if _is_numeric_constant(node):
        return True
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and _is_numeric_constant(node.operand)
    ):
        return True
    return False


def _is_numeric_dict(node: ast.AST) -> bool:
    """``{..: <numeric>, ..}`` — every value is a numeric constant.

    Empty dicts return False (they're not "all numeric" in a meaningful sense).
    """
    if not isinstance(node, ast.Dict) or not node.values:
        return False
    return all(_signed_numeric(v) for v in node.values)


def _is_hardcoded_pose(node: ast.AST) -> bool:
    """Match the canonical hardcoded-pose shape::

        {"position":    {"x": <num>, "y": <num>, "z": <num>},
         "orientation": {"x": <num>, "y": <num>, "z": <num>, "w": <num>}}

    or the simpler::

        {"x": <num>, "y": <num>, "z": <num>}

    or any dict whose values are all numeric / numeric-dicts.
    Variable references (``grasp["translation_wrt_base"]`` etc.) are *not*
    matched — those are perception-derived and pass the audit.
    """
    if not isinstance(node, ast.Dict) or not node.values:
        return False
    for v in node.values:
        if _signed_numeric(v):
            continue
        if _is_numeric_dict(v):
            continue
        return False
    return True


def _looks_like_collection(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in _COLLECTION_HINTS)


def _truncate(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class _AuditVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.flags: list[RiskFlag] = []
        self._seen_relative_move = False
        # Variables assigned from perception calls (or from chained subscripts
        # / attribute access of such variables). These are "safe" — using
        # values derived from them does not trigger audit flags.
        self._perception_vars: set[str] = set()
        # Variables holding fully-hardcoded pose dicts. Catches the common
        # form ``target = {"position": {...numeric...}, ...}; move_ee_to_pose(target)``
        # which would otherwise slip past the call-site check.
        self._hardcoded_pose_vars: set[str] = set()

    # ------------------------------------------------------------------
    # Tracking perception-derived variables
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_perception_value(node.value):
            for target in node.targets:
                self._record_target(target, self._perception_vars)
        elif _is_hardcoded_pose(node.value):
            for target in node.targets:
                self._record_target(target, self._hardcoded_pose_vars)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            self.generic_visit(node)
            return
        if self._is_perception_value(node.value):
            self._record_target(node.target, self._perception_vars)
        elif _is_hardcoded_pose(node.value):
            self._record_target(node.target, self._hardcoded_pose_vars)
        self.generic_visit(node)

    def _record_target(self, target: ast.AST, bucket: set[str]) -> None:
        if isinstance(target, ast.Name):
            bucket.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._record_target(elt, bucket)

    def _is_perception_value(self, value: ast.AST) -> bool:
        """Recursively decide whether ``value`` traces back to a perception call."""
        if isinstance(value, ast.Call):
            name = _call_name(value)
            return name in _PERCEPTION_FUNCS
        if isinstance(value, ast.Subscript):
            return self._is_perception_value(value.value)
        if isinstance(value, ast.Attribute):
            return self._is_perception_value(value.value)
        if isinstance(value, ast.Name):
            return value.id in self._perception_vars
        return False

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)

        if name == "move_ee_to_rel_pose":
            self._seen_relative_move = True

        if name == "move_ee_to_pose":
            self._check_move_ee_to_pose(node)

        if name in _JOINT_CONFIG_FUNCS:
            self._check_joint_config(name, node)

        self.generic_visit(node)

    def _check_move_ee_to_pose(self, node: ast.Call) -> None:
        if node.args and self._is_hardcoded_arg(node.args[0]):
            self.flags.append(RiskFlag(
                line=node.lineno,
                kind="hardcoded_pose",
                severity=SEVERITY_HIGH,
                message=(
                    "move_ee_to_pose called with a hardcoded pose — won't "
                    "generalize to scene shifts. Route through detect_objects() "
                    "/ detect_markers() instead, or use move_ee_to_rel_pose "
                    "for fixed offsets."
                ),
                snippet=_truncate(ast.unparse(node)),
            ))
        if self._seen_relative_move:
            self.flags.append(RiskFlag(
                line=node.lineno,
                kind="absolute_after_relative",
                severity=SEVERITY_MEDIUM,
                message=(
                    "Absolute move_ee_to_pose after a prior move_ee_to_rel_pose — "
                    "the relative offset's intent may be invalidated."
                ),
                snippet=_truncate(ast.unparse(node)),
            ))

    def _check_joint_config(self, name: str, node: ast.Call) -> None:
        if not node.args:
            return
        arg = node.args[0]
        if _is_numeric_dict(arg):
            self.flags.append(RiskFlag(
                line=node.lineno,
                kind="hardcoded_joint_config",
                severity=SEVERITY_HIGH,
                message=(
                    f"{name} called with hardcoded joint values — replays the "
                    "demo scene, will not adapt to perturbed object positions."
                ),
                snippet=_truncate(ast.unparse(node)),
            ))

    def _is_hardcoded_arg(self, arg: ast.AST) -> bool:
        """Direct dict literal *or* a Name bound to one earlier in the file."""
        if _is_hardcoded_pose(arg):
            return True
        if isinstance(arg, ast.Name) and arg.id in self._hardcoded_pose_vars:
            return True
        return False

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_int_index(node) and self._suggests_perception(node.value):
            self.flags.append(RiskFlag(
                line=node.lineno,
                kind="index_subscript",
                severity=SEVERITY_MEDIUM,
                message=(
                    f"{ast.unparse(node)} uses a position index into a "
                    "detection/marker collection — fragile to detection-order "
                    "changes. Filter by id or attribute instead."
                ),
                snippet=_truncate(ast.unparse(node)),
            ))
        self.generic_visit(node)

    @staticmethod
    def _is_int_index(node: ast.Subscript) -> bool:
        s = node.slice
        return isinstance(s, ast.Constant) and isinstance(s.value, int)

    def _suggests_perception(self, value: ast.AST) -> bool:
        """True iff ``value`` looks like a *collection of detections* — the
        kind whose position-based index access is order-fragile.

        Coordinate-style accesses (``translation_wrt_base[0]``) are
        deliberately excluded: indexing a 3-vector by 0/1/2 is part of the
        normal API contract, not a generalization risk."""
        if isinstance(value, ast.Name):
            return _looks_like_collection(value.id)
        if isinstance(value, ast.Subscript):
            # ``foo["markers"][0]`` — recurse into the inner subscript's value
            # so the "looks like a collection" check reaches the string key
            # via the parent path.
            return self._suggests_perception(value.value)
        if isinstance(value, ast.Attribute):
            return _looks_like_collection(value.attr)
        if isinstance(value, ast.Call):
            return _call_name(value) == "detect_markers"  # only the explicit collection-returning call
        return False
