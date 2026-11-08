"""GDP-style shipment report (HTML and markdown, optional PDF) built from an Analysis."""

from __future__ import annotations

import base64
import math
from datetime import UTC, datetime
from pathlib import Path

import jinja2

from ccee import __version__
from ccee.analysis import Analysis
from ccee.figures import report_chart_png

MAX_AUDIT_ROWS = 40


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.PackageLoader("ccee", "templates"),
        autoescape=jinja2.select_autoescape(enabled_extensions=("html.j2",), default=False),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_context(an: Analysis, shipment_id: str, header: list[tuple[str, str]], data_label: str, reviewer: str | None = None) -> dict:
    mon = an.monitor
    p = an.profile
    events = []
    for e, where in zip(mon.events, an.locations, strict=True):
        notes = []
        if e.open_ended:
            notes.append("still out of range when record ended")
        if e.onset_unobserved:
            notes.append("onset after a data gap")
        if e.spans_gap:
            notes.append("spans a data gap")
        events.append(
            {
                "direction": e.direction,
                "start": f"{e.start:%Y-%m-%d %H:%M}",
                "end": f"{e.end:%Y-%m-%d %H:%M}" if e.end is not None else "",
                "minutes": f"{e.minutes:.0f}",
                "peak": f"{e.peak_c:.1f}",
                "dm": f"{e.degree_minutes:.0f}",
                "where": where,
                "notes": "; ".join(notes),
            }
        )
    bands = []
    for b, used in zip(p.bands, mon.band_minutes, strict=True):
        bands.append(
            {
                "label": f"{b.lo:g} to {b.hi:g}",
                "used": f"{used:.0f}",
                "allowed": f"{b.max_minutes:g}",
                "frac": f"{100 * used / b.max_minutes:.0f} %" if b.max_minutes else "n/a",
            }
        )
    used_total = mon.budget_consumed()
    if math.isinf(used_total):
        budget_text = "exceeded (time spent where no allowance is defined)" if mon.unbudgeted_minutes else "exceeded"
    else:
        budget_text = f"{100 * used_total:.0f} % of the illustrative allowance"
    needs_review = bool(mon.events) or any(f.severity == "critical" for f in an.flags)
    rows = an.audit.entries
    ok, msg = an.audit.verify()
    return {
        "shipment_id": shipment_id,
        "version": __version__,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "data_label": data_label,
        "header": header,
        "profile": p,
        "stats": an.stats,
        "mkt_c": an.mkt_c,
        "events": events,
        "bands": bands,
        "budget_text": budget_text,
        "alarms": [
            {"ts": f"{a.ts:%Y-%m-%d %H:%M}", "rule": a.rule, "severity": a.severity, "message": a.message}
            for a in mon.alarms
        ],
        "flags": an.flags,
        "needs_review": needs_review,
        "chart_b64": base64.b64encode(report_chart_png(an.df, p, mon)).decode("ascii"),
        "audit_ok": ok,
        "audit_msg": msg,
        "audit_head": rows[-1]["hash"] if rows else "",
        "audit_rows": rows[:MAX_AUDIT_ROWS],
        "audit_more": max(0, len(rows) - MAX_AUDIT_ROWS),
        "reviewer": reviewer,
    }


def render(an: Analysis, shipment_id: str, header: list[tuple[str, str]], data_label: str, reviewer: str | None = None) -> tuple[str, str]:
    """Return (html, markdown). Records the report in the audit trail first so the printed chain covers it."""
    an.audit.append("report_generated", "ccee", {"shipment_id": shipment_id, "format": "html+md"})
    ctx = build_context(an, shipment_id, header, data_label, reviewer)
    env = _env()
    return env.get_template("report.html.j2").render(r=ctx), env.get_template("report.md.j2").render(r=ctx)


def write_report(
    an: Analysis,
    out_dir: str | Path,
    shipment_id: str,
    header: list[tuple[str, str]],
    data_label: str,
    reviewer: str | None = None,
    pdf: bool = False,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html, md = render(an, shipment_id, header, data_label, reviewer)
    paths = {
        "html": out / f"{shipment_id}.html",
        "md": out / f"{shipment_id}.md",
        "audit": out / f"{shipment_id}.audit.jsonl",
    }
    paths["html"].write_text(html, encoding="utf-8")
    paths["md"].write_text(md, encoding="utf-8")
    an.audit.save(paths["audit"])
    if pdf:
        try:
            from weasyprint import HTML  # optional extra: pip install '.[pdf]'
        except ImportError as exc:
            raise RuntimeError("PDF output needs the optional 'pdf' extra (weasyprint)") from exc
        paths["pdf"] = out / f"{shipment_id}.pdf"
        HTML(string=html).write_pdf(str(paths["pdf"]))
    return paths
