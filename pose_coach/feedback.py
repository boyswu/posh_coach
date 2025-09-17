# Author: wujiahang
from __future__ import annotations
from typing import Dict, List, Tuple, Any

PART_CN = {"arms": "上肢", "legs": "下肢", "torso": "躯干"}
JOINT_CN = {
    "LElbow": "左肘", "RElbow": "右肘", "LKnee": "左膝", "RKnee": "右膝",
    "LHip": "左髋", "RHip": "右髋", "LShoulder": "左肩", "RShoulder": "右肩",
    "LAnkle": "左踝", "RAnkle": "右踝", "Neck": "颈部"
}


def make_advice(x: Any, mode: str = "rule", **kwargs) -> str:
    if isinstance(x, (int, float)):
        return _legacy_rule(float(x))
    if not isinstance(x, dict):
        return "未获得有效的分析结果，请重试或更换更清晰的视频。"
    if mode == "ml":
        clf = kwargs.get("clf")
        return _ml_advice(x, clf)
    if mode == "llm":
        llm_client = kwargs.get("llm_client")
        return _llm_advice(x, llm_client)
    return _rule_advice(x)


def suggest_recipes(main_groups: List[Tuple[str, float]]) -> List[str]:
    if not main_groups:
        return []
    recipes = {
        "legs": [
            "徒手深蹲 3×12：注意膝盖不过脚尖，避免内扣",
            "弓步蹲 3×10：保持髋部稳定、脚尖朝前"
        ],
        "arms": [
            "弹力带划船 3×12：肩胛后缩、肘部靠近躯干",
            "俯卧撑 3×10：核心收紧，肘部约45°"
        ],
        "torso": [
            "死虫 3×10/侧：保持腰椎中立，配合呼吸",
            "平板支撑 3×30秒：骨盆中立，不塌腰"
        ]
    }
    out = []
    for key, _ in sorted(main_groups, key=lambda x: x[1], reverse=True)[:3]:
        cn = PART_CN.get(key, key)
        drill = "；".join(recipes.get(key, []))
        if drill:
            out.append(f"{cn}：{drill}")
    return out


def _legacy_rule(score: float) -> str:
    if score >= 85:
        return "整体动作完成度很高，保持节奏与稳定性即可。"
    if score >= 70:
        return "动作质量良好，注意节奏稳定，适度加大关键关节的幅度以贴近模板。"
    if score >= 50:
        return "有一定差距，建议先拆分动作练习，聚焦膝/肘/肩等关键关节角度控制。"
    return "与模板差异较大，建议重拍更清晰的全身画面，并按模板节拍逐段对齐练习。"


def _fmt_pair_list(pairs: List[Tuple[str, float]]) -> str:
    if not pairs:
        return ""
    pairs = [f"{JOINT_CN.get(j, j)}" for j, _ in pairs[:2]]
    return "、".join(pairs)


def _rule_advice(struct: Dict[str, Any]) -> str:
    score = float(struct.get("overall_score", 0.0))
    main_groups = list(struct.get("main_groups", []))
    details = struct.get("group_details", {}) or {}
    base = _legacy_rule(score)
    if not main_groups:
        return base
    tops = []
    for key, val in sorted(main_groups, key=lambda x: x[1], reverse=True)[:3]:
        name = PART_CN.get(key, key)
        joints = _fmt_pair_list(details.get(key, []))
        seg = f"{name}（重点关注：{joints}）" if joints else f"{name}"
        tops.append(seg)
    tail = "；".join(tops)
    if tail:
        return f"{base} 主要问题集中在：{tail}。"
    return base


def _ml_advice(struct: Dict[str, Any], clf) -> str:
    try:
        x = [
            struct.get("overall_score", 0.0),
            *(struct.get("group_scores", {}).get(k, 0.0) for k in ("arms", "legs", "torso")),
        ]
        y = clf.predict([x])[0]  # type: ignore
        if int(y) == 2:
            return "模型评估：整体较好；建议在节奏一致性上再做巩固。"
        if int(y) == 1:
            return "模型评估：中等；建议针对主要问题部位做分段练习与对齐校准。"
        return "模型评估：偏弱；请检查入镜、光照与站位，并拆分动作逐段模仿。"
    except Exception:
        return _rule_advice(struct)


def _llm_advice(struct: Dict[str, Any], llm_client) -> str:
    import json
    prompt = (
            "你是一名专业动作教练。请阅读以下 JSON 结构化评估结果，"
            "用不超过120字的中文给出直接可执行的改进建议，少客套，突出要点：\n"
            + json.dumps(struct, ensure_ascii=False)
    )
    try:
        if callable(llm_client):
            resp = llm_client(prompt)
            return str(resp).strip()
        if hasattr(llm_client, "generate"):
            resp = llm_client.generate(text=prompt)
            return str(getattr(resp, "text", resp)).strip()
        if hasattr(llm_client, "chat"):
            resp = llm_client.chat(prompt=prompt)
            return str(getattr(resp, "text", resp)).strip()
    except Exception:
        pass
    return _rule_advice(struct)