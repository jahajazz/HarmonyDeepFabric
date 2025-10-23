import json
from typing import Any, Dict, TextIO, Optional


def flatten_analysis(o: Any) -> str:
    if isinstance(o, str):
        return o.strip()
    if isinstance(o, dict):
        parts = []
        sm = o.get('symbolic_meaning')
        if sm:
            parts.append(f"Symbolic reasoning:\n{sm}")
        ap = o.get('archetypal_patterns')
        if ap:
            parts.append("Archetypal patterns: " + ", ".join(map(str, ap)))
        tt = o.get('theological_themes')
        if tt:
            parts.append("Theological themes: " + ", ".join(map(str, tt)))
        return "\n\n".join(parts).strip()
    return ""


def flatten_final(o: Any, fallback: Optional[str] = None) -> str:
    if isinstance(o, str):
        return o.strip()
    if isinstance(o, dict):
        for k in ('summary', 'symbolic_summary', 'final'):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return (fallback or "").strip()


def write_harmony_record(
    out_fp: TextIO,
    system_prompt: str,
    user_content: str,
    analysis_obj: Any,
    final_obj: Any,
    metadata: Optional[Dict[str, Any]],
) -> None:
    analysis_txt = flatten_analysis(analysis_obj)
    final_txt = flatten_final(final_obj)
    if not analysis_txt or not final_txt:
        raise ValueError("Empty analysis/final")
    rec = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "analysis": analysis_txt,
                "final": final_txt,
            },
        ],
        "metadata": metadata or {},
    }
    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
