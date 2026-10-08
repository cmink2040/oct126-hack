"""Shared look for Clinic Copilot and the patient portal: palette, charts and small components.

Charts follow one set of rules: color does one job (identity, magnitude or status), categorical hues come from
the validated reference palette in fixed order, marks are thin with rounded data-ends and 2px lines, grids are
recessive, every chart has a hover tooltip, and text stays in text colors, never series colors.
"""
from __future__ import annotations

import pathlib

import altair as alt
import pandas as pd
import streamlit as st

# Reference categorical palette, light and dark steps (validated: CVD-adjacent dE >= 8.4, normal-vision >= 19.6).
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}
# One-hue ramp (blue) for "same thing, different state" - e.g. booked now vs still expected.
TINT = {"light": "#9ec5f4", "dark": "#184f95"}
MUTED = {"light": "#77756e", "dark": "#9b9a93"}

# Urgency is status, not identity: reserved status colors, always with an icon and a label.
URGENCY = {"emergency": ("red", ":material/emergency:"), "urgent": ("orange", ":material/priority_high:"),
           "soon": ("yellow", ":material/schedule:"), "routine": ("gray", ":material/inbox:")}


def mode() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def series(i: int = 0) -> str:
    return SERIES[mode()][i]


# ---------------------------------------------------------------- components
def logo(name: str) -> None:
    """Brand mark + clinic name at the top of the sidebar (generated SVG, so the name follows settings)."""
    import html
    import tempfile
    ink = "#f1f0ea" if mode() == "dark" else "#1c1c1a"
    accent = "#3cc3b2" if mode() == "dark" else "#0f766e"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="260" height="36" viewBox="0 0 260 36">
<rect x="1" y="3" width="30" height="30" rx="8" fill="{accent}"/>
<path d="M16 10v16M8 18h16" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
<text x="40" y="24" font-family="Source Sans Pro, Helvetica, Arial, sans-serif" font-size="17" font-weight="700"
 fill="{ink}">{html.escape(name)}</text>
</svg>"""
    path = pathlib.Path(tempfile.gettempdir()) / f"chiro_logo_{mode()}_{abs(hash(name))}.svg"
    if not path.exists():
        path.write_text(svg)
    st.logo(str(path), size="large")



def header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def esc(text) -> str:
    """Make free text (agent drafts, rationales, patient messages) safe for st.markdown: '$' starts LaTeX there,
    so '$78 -> $81' would render as math."""
    return str(text or "").replace("$", "\\$")


def tag(text: str, color: str = "gray", icon: str | None = None) -> str:
    """Inline badge markup; several in one st.markdown line wrap naturally instead of truncating in columns."""
    text = esc(text).replace("[", "(").replace("]", ")")
    return f":{color}-badge[{icon + ' ' if icon else ''}{text}]"


def urgency_tag(tier: str | None) -> str:
    color, icon = URGENCY.get(tier or "", ("gray", ":material/help:"))
    return tag(tier or "untriaged", color, icon)


def badges(*tags: str) -> None:
    st.markdown(" ".join(t for t in tags if t))


def badge(text: str, color: str = "gray", icon: str | None = None) -> None:
    badges(tag(text, color, icon))


def kpi(col, label: str, value, help: str | None = None, delta=None, trend: list | None = None,
        delta_color: str = "normal") -> None:
    col.metric(label, value, delta=delta, help=help, border=True, chart_data=trend, chart_type="area",
               delta_color=delta_color)


def empty(message: str, icon: str = ":material/check_circle:") -> None:
    st.info(message, icon=icon)


def money(x) -> str:
    return "-" if x is None else f"${x:,.0f}"


def pct(x, digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}%}"


# ---------------------------------------------------------------- charts
def _show(chart: alt.Chart, height: int) -> None:
    st.altair_chart(chart.properties(height=height).configure_view(strokeWidth=0)
                    .configure_axis(grid=True, gridOpacity=0.35, domain=False, ticks=False, labelPadding=6,
                                    titleFontWeight="normal", titlePadding=8),
                    theme="streamlit", width="stretch")


def line(df: pd.DataFrame, x: str, y: str, *, y_title: str, y_format: str = ",.0f", x_title: str = "",
         x_type: str = "T", height: int = 240, zero: bool = True, axis_title: bool = False) -> None:
    """One series over time: a 2px line, a crosshair rule and a point + tooltip on hover."""
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(df).encode(x=alt.X(f"{x}:{x_type}", title=x_title, axis=alt.Axis(labelOverlap=True)))
    ln = base.mark_line(strokeWidth=2, color=series(0)).encode(
        y=alt.Y(f"{y}:Q", title=y_title if axis_title else None, axis=alt.Axis(format=y_format, tickCount=5),
                scale=alt.Scale(zero=zero)))
    pts = base.mark_point(size=70, filled=True, color=series(0)).encode(
        y=f"{y}:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:{x_type}", title=x_title or x.replace("_", " ")),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=y_format)]).add_params(hover)
    rule = base.mark_rule(color=MUTED[mode()], strokeDash=[3, 3]).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0))).transform_filter(hover)
    _show(alt.layer(ln, rule, pts), height)


def hbar(df: pd.DataFrame, cat: str, val: str, *, val_title: str, val_format: str = ",.0f", height: int | None = None,
         sort: str = "-x", label: bool = True) -> None:
    """One series across categories: horizontal, thin, rounded ends at the value, direct labels, tooltip."""
    h = height or max(120, 30 * len(df) + 30)
    base = alt.Chart(df).encode(y=alt.Y(f"{cat}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=180)),
                                x=alt.X(f"{val}:Q", title=None if label else val_title,
                                        axis=alt.Axis(format=val_format, tickCount=4)))
    bars = base.mark_bar(size=14, cornerRadiusEnd=4, color=series(0)).encode(
        tooltip=[alt.Tooltip(f"{cat}:N", title=cat.replace("_", " ")),
                 alt.Tooltip(f"{val}:Q", title=val_title, format=val_format)])
    layers = [bars]
    if label:
        layers.append(base.mark_text(align="left", dx=5, fontSize=11).encode(
            text=alt.Text(f"{val}:Q", format=val_format), color=alt.value(MUTED[mode()])))
    _show(alt.layer(*layers), h)


def vbar(df: pd.DataFrame, cat: str, val: str, *, val_title: str, val_format: str = ",.0f", height: int = 220,
         order: list | None = None) -> None:
    """One series across ordered categories (e.g. weekdays): vertical, thin, rounded tops, tooltip."""
    chart = alt.Chart(df).mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=series(0)).encode(
        x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{val}:Q", title=val_title, axis=alt.Axis(format=val_format, tickCount=4)),
        tooltip=[alt.Tooltip(f"{cat}:N", title=cat.replace("_", " ")),
                 alt.Tooltip(f"{val}:Q", title=val_title, format=val_format)])
    _show(chart, height)


def forecast(df: pd.DataFrame, height: int = 260) -> None:
    """Forward days: booked now (solid) inside expected final bookings (tint of the same hue), with effective
    capacity as a muted tick. Same entity in two states, so one hue in two steps, plus a legend."""
    long = df.melt(id_vars=["day", "weekday", "effective_capacity", "projected_idle_slots"],
                   value_vars=["expected_final_bookings", "booked_now"], var_name="state", value_name="slots")
    long["state"] = long["state"].map({"expected_final_bookings": "Expected by the day",
                                       "booked_now": "Booked now"})
    color = alt.Color("state:N", title=None, legend=alt.Legend(orient="top", direction="horizontal"),
                      scale=alt.Scale(domain=["Booked now", "Expected by the day"],
                                      range=[series(0), TINT[mode()]]))
    tip = [alt.Tooltip("day:T", title="day", format="%a %b %d"), alt.Tooltip("state:N", title=" "),
           alt.Tooltip("slots:Q", title="slots", format=".0f"),
           alt.Tooltip("effective_capacity:Q", title="capacity"),
           alt.Tooltip("projected_idle_slots:Q", title="projected idle", format=".0f")]
    x = alt.X("day:T", title=None, axis=alt.Axis(format="%a %d", labelAngle=0, tickCount=14))
    bars = alt.Chart(long).mark_bar(size=16, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x, y=alt.Y("slots:Q", title="slots", stack=None), color=color,
        order=alt.Order("state:N", sort="descending"), tooltip=tip)
    cap = alt.Chart(df).mark_tick(thickness=2, size=22, color=MUTED[mode()]).encode(
        x=x, y="effective_capacity:Q", tooltip=[alt.Tooltip("effective_capacity:Q", title="capacity")])
    _show(alt.layer(bars, cap), height)
