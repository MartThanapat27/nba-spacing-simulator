"""
eda_dashboard.py -- Interactive dashboard with a sidebar-navigated menu of
two independent pages (see the `st.sidebar.radio` "Navigate" menu, which is
the dashboard's main index -- always visible, not buried in a tab):

  1. Single-Game Play-by-Play Viewer (default/first page) -- runs one
     narrated single game through the FastAPI backend's `POST /api/simulate`
     (mode="cpu", which already returns the full possession-by-possession
     text -- see backend/api_simulation.py's SimulateResponse.play_by_play),
     parses it into a chronological event table, and shows the score
     progression alongside the GPU batch's aggregate win probabilities (raw
     and, optionally, Safety-Brake-blended) for context.
  2. Player Stats EDA -- quick EDA for nba_db's player_stats_raw table
     (requires Postgres populated via `python load_player_stats_raw.py`).

The two pages are fully independent: the EDA page needs Postgres, the
Play-by-Play page needs only the FastAPI backend (http://127.0.0.1:8000 by
default) -- one being down does not block the other (see the two
independent `if page == ...:` blocks below, gated on the sidebar radio's
single selection; a DB failure in the EDA page reports an error in-place
rather than calling st.stop(), which would have also killed the
Play-by-Play page's rendering on the same script pass).

Run with:
    streamlit run eda_dashboard.py
"""

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine

DB_URL = os.environ.get(
    "NBA_DB_URL", "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/nba_db"
)
API_BASE_URL = os.environ.get("NBA_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NBA Simulation Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Primary navigation -- a sidebar menu, not a buried tab, so a new section
# (like Play-by-Play below) is always visible in the dashboard's index
# instead of requiring a user to already know a second tab exists. The
# Play-by-Play Viewer is listed AND selected first by default since it's
# the newest primary feature.
# ---------------------------------------------------------------------------
PBP_LABEL = "🏀 Single-Game Play-by-Play Viewer"
EDA_LABEL = "📊 Player Stats EDA"

st.sidebar.title("NBA Simulation Dashboard")
page = st.sidebar.radio("Navigate", [PBP_LABEL, EDA_LABEL], index=0)
st.sidebar.divider()


# ---------------------------------------------------------------------------
# Page 1: Player Stats EDA (pre-existing content, unchanged in substance --
# only gated on the sidebar nav selection and switched from st.stop() to
# st.error() so a DB outage doesn't also block the independent Play-by-Play
# page).
# ---------------------------------------------------------------------------
if page == EDA_LABEL:
    st.title("NBA Player Stats EDA")

    @st.cache_resource
    def get_engine():
        return create_engine(DB_URL)

    @st.cache_data(ttl=600)
    def load_data() -> pd.DataFrame:
        query = """
            SELECT player_name, team_abbreviation, gp, min, fg3a, fg3_pct, pts, plus_minus
            FROM player_stats_raw
        """
        return pd.read_sql(query, get_engine())

    try:
        df = load_data()
    except Exception as exc:  # noqa: BLE001 -- surface connection/missing-table errors plainly
        df = None
        st.error(
            f"Could not read `player_stats_raw` from nba_db: {exc}\n\n"
            "Make sure Postgres is running (`docker compose up -d`) and the table is "
            "populated (`python load_player_stats_raw.py`)."
        )

    if df is not None:
        st.caption(f"{len(df)} players loaded from `player_stats_raw`")

        min_gp = int(df["gp"].min())
        max_gp = int(df["gp"].max())
        gp_floor = st.sidebar.slider("Minimum games played (GP)", min_gp, max_gp, min_gp)
        df = df[df["gp"] >= gp_floor]

        st.header("1. Distribution shape: FG3_PCT and PLUS_MINUS")
        col1, col2 = st.columns(2)
        with col1:
            fg3_pct_clean = df["fg3_pct"].dropna()
            fig = px.histogram(fg3_pct_clean, x="fg3_pct", nbins=30, title="FG3_PCT distribution")
            fig.add_annotation(
                text=f"skew = {fg3_pct_clean.skew():.2f}",
                xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            plus_minus_clean = df["plus_minus"].dropna()
            fig = px.histogram(plus_minus_clean, x="plus_minus", nbins=30, title="PLUS_MINUS distribution")
            fig.add_annotation(
                text=f"skew = {plus_minus_clean.skew():.2f}",
                xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.header("2. FG3A vs PTS correlation")
        corr = df["fg3a"].corr(df["pts"])
        st.metric("Pearson correlation (FG3A vs PTS)", f"{corr:.2f}")
        fig = px.scatter(
            df, x="fg3a", y="pts", hover_name="player_name", hover_data=["team_abbreviation", "gp"],
            labels={"fg3a": "3-Point Attempts", "pts": "Points"},
            title="3-Point Attempts vs Points",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("3. Outliers: high shooting % on very low minutes/games")
        low_sample = st.sidebar.slider("Low-sample threshold: MIN below", 1, int(df["min"].max()), 50)
        outlier_df = df.dropna(subset=["fg3_pct"]).copy()
        outlier_df["low_sample_hot_shooter"] = (outlier_df["min"] < low_sample) & (outlier_df["fg3_pct"] >= 0.4)
        fig = px.scatter(
            outlier_df, x="min", y="fg3_pct", color="low_sample_hot_shooter",
            size="gp", hover_name="player_name", hover_data=["team_abbreviation", "gp", "fg3a"],
            labels={"min": "Minutes", "fg3_pct": "3-Point %"},
            title="3-Point % vs Minutes (flagging small-sample hot shooters)",
            color_discrete_map={True: "crimson", False: "steelblue"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            outlier_df[outlier_df["low_sample_hot_shooter"]]
            .sort_values("fg3_pct", ascending=False)[["player_name", "team_abbreviation", "gp", "min", "fg3a", "fg3_pct"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page 2: Single-Game Play-by-Play Viewer
# ---------------------------------------------------------------------------

# Real NBA team abbreviations -- a fixed, static list (not DB-dependent, so
# this tab works even if Postgres/player_stats_raw is unavailable; the
# actual roster for whichever two are picked still comes live from the
# FastAPI backend's /api/players at simulation time).
NBA_TEAMS = sorted([
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
])

# Matches every possession's own bracketed timestamp line, e.g.
# " [Q4 | 46:31] LAL: LeBron James (guarded by X) handles the ball. ..."
# or " [OT1 | 48:16] ... [CLUTCH PLAY]: ...". Each such line is already a
# COMPLETE, self-contained event in the engine's own stdout (the touch and
# its resolution -- make/miss/foul/turnover -- print on one physical line;
# see cpp_engine/main.cpp's PossessionEngine::simulate_possession, which
# only calls std::endl once per event) -- see this module's docstring.
_EVENT_LINE_RE = re.compile(
    r"^\s*\[(?P<period>Q\d+|OT\d+)\s*\|\s*(?P<min>\d+):(?P<sec>\d+)\]\s*"
    r"(?P<team>\S+)\s*(?:\[CLUTCH PLAY\]\s*)?:\s*(?P<text>.*)$"
)
_POINTS_RE = re.compile(r"\[\+(\d+)\s*pts\]")
# Free throws never get a "[+N pts]" marker (see PossessionEngine's shooting-foul
# and bonus-foul branches in main.cpp) -- each make is worth exactly 1 point, and
# the made count is printed as "<made>/<attempted> makes.".
_FT_MAKES_RE = re.compile(r"goes to the line,\s*(\d+)/(\d+)\s*makes\.")
_REBOUND_LINE_RE = re.compile(r"^\s*->\s*(?P<text>.+)$")
_OVERTIME_RE = re.compile(r"\*\*\*\s*(OVERTIME\s+\d+)")
_SUB_LINE_RE = re.compile(r"^\s*\[Min\s+(?P<minute>\d+)\]\s*(?P<team>\S+)\s+SUB:\s*(?P<text>.+)$")


def parse_play_by_play(raw_text: str, team_a: str, team_b: str) -> pd.DataFrame:
    """Turns the CPU engine's free-text play-by-play (SimulateResponse.play_by_play)
    into a structured, chronological event table: one row per possession
    (plus substitutions and overtime-period markers), with a running score
    reconstructed from each event's own "[+N pts]" marker -- the raw text
    never prints a cumulative score itself, only per-event point deltas.

    Deliberately tolerant, not a strict grammar: a line this doesn't
    recognize is simply skipped (folded into the raw text still shown
    separately in the UI), so a future narrative-text tweak in the C++
    engine degrades this table gracefully instead of crashing the
    dashboard.
    """
    rows: list[dict] = []
    score_a, score_b = 0, 0
    seq = 0

    for line in raw_text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue

        ot_match = _OVERTIME_RE.search(line)
        if ot_match:
            seq += 1
            rows.append({
                "seq": seq, "period": ot_match.group(1), "clock": "", "team": "",
                "event_type": "period", "description": line.strip(),
                "points": 0, "score_a": score_a, "score_b": score_b,
            })
            continue

        sub_match = _SUB_LINE_RE.match(line)
        if sub_match:
            seq += 1
            rows.append({
                "seq": seq, "period": "", "clock": f"Min {sub_match.group('minute')}",
                "team": sub_match.group("team"), "event_type": "substitution",
                "description": sub_match.group("text").strip(),
                "points": 0, "score_a": score_a, "score_b": score_b,
            })
            continue

        event_match = _EVENT_LINE_RE.match(line)
        if event_match:
            team = event_match.group("team")
            text = event_match.group("text").strip()
            points_match = _POINTS_RE.search(text)
            ft_match = _FT_MAKES_RE.search(text)
            if points_match:
                points = int(points_match.group(1))
            elif ft_match:
                points = int(ft_match.group(1))
            else:
                points = 0
            if points:
                if team == team_a:
                    score_a += points
                elif team == team_b:
                    score_b += points
            seq += 1
            rows.append({
                "seq": seq,
                "period": event_match.group("period"),
                "clock": f"{event_match.group('min')}:{event_match.group('sec')}",
                "team": team, "event_type": "possession", "description": text,
                "points": points, "score_a": score_a, "score_b": score_b,
            })
            continue

        reb_match = _REBOUND_LINE_RE.match(line)
        if reb_match and rows:
            # A supplementary note on the immediately preceding possession
            # (an offensive/defensive rebound, or an AND-1 free throw after
            # a made shot) -- appended to that row's description rather than
            # given its own row, since it isn't a new possession/time.
            note = reb_match.group("text").strip()
            if "AND-1" in note and "converts the free throw" in note:
                prev = rows[-1]
                prev["points"] += 1
                if prev["team"] == team_a:
                    score_a += 1
                elif prev["team"] == team_b:
                    score_b += 1
                prev["score_a"], prev["score_b"] = score_a, score_b
            rows[-1]["description"] += " " + note
            continue
        # Any other line (headers, dividers, "FINAL SCORE", etc.) is
        # intentionally skipped here -- still visible in the raw-text
        # expander the UI renders alongside this table.

    return pd.DataFrame(rows)


def render_score_progression_chart(events: pd.DataFrame, team_a: str, team_b: str) -> go.Figure:
    """Score-progression line chart over the scoring events only (a flat
    stretch between two scores is real -- it means neither team scored in
    that stretch, not a gap in the data) -- plus vertical markers for
    every real lead change, the "momentum shift" signal this feature is
    meant to surface at a glance.
    """
    scoring = events[events["event_type"] == "possession"].copy()
    if scoring.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scoring["seq"], y=scoring["score_a"], mode="lines",
                              name=team_a, line=dict(color="#1f77b4", shape="hv")))
    fig.add_trace(go.Scatter(x=scoring["seq"], y=scoring["score_b"], mode="lines",
                              name=team_b, line=dict(color="#d62728", shape="hv")))

    # Real lead changes: where sign(score_a - score_b) flips (0 = tied,
    # not counted as a "side" of its own).
    diff = scoring["score_a"] - scoring["score_b"]
    sign = diff.apply(lambda d: 1 if d > 0 else (-1 if d < 0 else 0))
    nonzero = sign[sign != 0]
    lead_changes = nonzero[nonzero != nonzero.shift(1)].index[1:]  # skip the first real lead itself
    for idx in lead_changes:
        fig.add_vline(x=scoring.loc[idx, "seq"], line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(
        title=f"Score progression -- {team_a} vs {team_b} (dotted lines = lead changes)",
        xaxis_title="Possession sequence", yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def call_simulate(mode: str, team_a: str, team_b: str, home_team: Optional[str],
                   team_a_b2b: bool, team_b_b2b: bool, enable_shrinkage: bool,
                   timeout: float = 90.0) -> dict:
    payload = {
        "mode": mode, "roster_type": "real", "team_a": team_a, "team_b": team_b,
    }
    if home_team is not None:
        payload["home_team"] = home_team
    if team_a_b2b or team_b_b2b:
        payload["enable_fatigue"] = True
        if team_a_b2b:
            payload["team_a_rest_days"] = 0
        if team_b_b2b:
            payload["team_b_rest_days"] = 0
    if mode == "gpu" and enable_shrinkage:
        payload["enable_shrinkage"] = True

    resp = requests.post(f"{API_BASE_URL}/api/simulate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


if page == PBP_LABEL:
    st.title("Single-Game Play-by-Play Viewer")
    st.caption(
        "Runs one narrated single game through the CPU engine (simulator.exe) for a full "
        "possession-by-possession log, plus a 100,000-game GPU batch for win-probability context."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A (home, if Home Court below is set to Team A)", NBA_TEAMS, index=NBA_TEAMS.index("BOS"))
    with col_b:
        team_b = st.selectbox("Team B", NBA_TEAMS, index=NBA_TEAMS.index("LAL"))

    st.subheader("Toggles")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        home_choice = st.radio("Home Court", ["Neutral", f"Team A", f"Team B"], index=0)
        home_team = {"Neutral": None, "Team A": "a", "Team B": "b"}[home_choice]
    with tcol2:
        st.markdown("**Fatigue (back-to-back)**")
        team_a_b2b = st.checkbox(f"{team_a} is on a back-to-back", value=False)
        team_b_b2b = st.checkbox(f"{team_b} is on a back-to-back", value=False)
    with tcol3:
        enable_shrinkage = st.checkbox("Safety Brake (Prevent Overconfident Simulation)", value=False)
        st.caption(
            "The brake blends the GPU batch's AGGREGATE win probability across real "
            "NET_RATING context -- it has no effect on a single narrated game's own "
            "play-by-play, only on the win-probability context shown alongside it below."
        )

    if team_a == team_b:
        st.warning("Team A and Team B must be different.")

    run_clicked = st.button("Run Simulation", type="primary", disabled=(team_a == team_b))

    if run_clicked:
        with st.spinner(f"Simulating {team_a} vs {team_b} (single narrated game)..."):
            try:
                cpu_result = call_simulate("cpu", team_a, team_b, home_team, team_a_b2b, team_b_b2b, False)
            except requests.RequestException as e:
                cpu_result = None
                st.error(f"CPU simulation request failed: {e}")

        with st.spinner("Running 100,000-game GPU batch for win-probability context..."):
            try:
                gpu_result = call_simulate("gpu", team_a, team_b, home_team, team_a_b2b, team_b_b2b, enable_shrinkage)
            except requests.RequestException as e:
                gpu_result = None
                st.error(f"GPU batch request failed: {e}")

        if cpu_result is not None:
            st.session_state["pbp_cpu_result"] = cpu_result
            st.session_state["pbp_gpu_result"] = gpu_result
            st.session_state["pbp_team_a"] = team_a
            st.session_state["pbp_team_b"] = team_b

    if "pbp_cpu_result" in st.session_state:
        cpu_result = st.session_state["pbp_cpu_result"]
        gpu_result = st.session_state.get("pbp_gpu_result")
        ta = st.session_state["pbp_team_a"]
        tb = st.session_state["pbp_team_b"]

        if cpu_result.get("play_by_play"):
            st.divider()
            st.subheader("Final Result")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{ta} final score", cpu_result.get("final_score_a", "?"))
            m2.metric(f"{tb} final score", cpu_result.get("final_score_b", "?"))
            m3.metric("Winner (this single game)", cpu_result.get("winner") or "Tie")
            if gpu_result is not None and gpu_result.get("win_probability_a") is not None:
                win_prob_label = f"{gpu_result['win_probability_a'] * 100:.1f}%"
                if gpu_result.get("shrinkage_applied") and gpu_result.get("raw_win_probability_a") is not None:
                    win_prob_label += f" (raw: {gpu_result['raw_win_probability_a'] * 100:.1f}%)"
                m4.metric(f"{ta} win probability (100k-game GPU batch)", win_prob_label)

            if gpu_result is not None:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.metric(f"{ta} avg score (GPU batch)", f"{gpu_result.get('average_score_a', 0):.1f}")
                with gcol2:
                    st.metric(f"{tb} avg score (GPU batch)", f"{gpu_result.get('average_score_b', 0):.1f}")
                if gpu_result.get("shrinkage_applied"):
                    st.caption(
                        f"Safety Brake applied -- dynamic raw-sim weight "
                        f"{gpu_result['parameters_applied'].get('shrinkage_weight', float('nan')):.3f} "
                        f"(delta NET_RATING {gpu_result['parameters_applied'].get('delta_net_rating', float('nan')):.1f})."
                    )
                elif enable_shrinkage:
                    st.caption("Safety Brake requested but not applied (see warnings below) -- showing pure raw output.")
                if gpu_result.get("warnings"):
                    for w in gpu_result["warnings"]:
                        st.warning(w)

            events = parse_play_by_play(cpu_result["play_by_play"], ta, tb)

            if not events.empty:
                st.subheader("Score Progression & Momentum Shifts")
                st.plotly_chart(render_score_progression_chart(events, ta, tb), use_container_width=True)

                st.subheader("Quarter-by-Quarter Breakdown")
                possession_events = events[events["event_type"] == "possession"]
                if not possession_events.empty:
                    q_summary = (
                        possession_events.groupby("period")
                        .agg(final_score_a=("score_a", "last"), final_score_b=("score_b", "last"),
                             possessions=("seq", "count"))
                        .reset_index()
                    )
                    st.dataframe(q_summary, use_container_width=True, hide_index=True)

                st.subheader("Chronological Play-by-Play Log")
                event_type_filter = st.multiselect(
                    "Show event types", options=sorted(events["event_type"].unique()),
                    default=list(events["event_type"].unique()),
                )
                display_cols = ["period", "clock", "team", "event_type", "description", "points", "score_a", "score_b"]
                st.dataframe(
                    events[events["event_type"].isin(event_type_filter)][display_cols]
                    .rename(columns={"score_a": f"{ta} score", "score_b": f"{tb} score"}),
                    use_container_width=True, hide_index=True, height=500,
                )
            else:
                st.info("Could not parse any structured events from the play-by-play text -- showing raw log only.")

            with st.expander("Raw play-by-play text (unparsed, always authoritative)"):
                st.text(cpu_result["play_by_play"])
        else:
            st.warning("No play-by-play text was returned for this matchup.")
