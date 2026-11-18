"""RateMyForm - Streamlit frontend.

Run with:  streamlit run app.py

Three stages (from our app-screen planning):
    1. Exercise selection + camera instructions
    2. Video upload + body measurements (validated)
    3. Results: scores, reps, charts, skeleton overlay, personalization

How the frontend talks to the analysis code: this file only collects input,
validates it with ratemyform.validation, calls ONE function
(ratemyform.pipeline.analyze_video, or analyze_sequence for the synthetic
demo), and renders the AnalysisResult it gets back. All analysis logic lives
in the ratemyform package, so it can be tested without the UI.

Privacy: the uploaded video is written to a temporary folder on THIS computer,
analyzed locally, and the folder is deleted right after analysis. Nothing is
sent to an external service.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import tempfile

import streamlit as st

from ratemyform import charts
from ratemyform.config import DEFAULT_CONFIG, DISCLAIMER, PLANNED_EXERCISES
from ratemyform.pipeline import AnalysisResult, analyze_sequence, analyze_video
from ratemyform.pose_detection import ModelNotFoundError, VideoReadError, probe_video
from ratemyform.synthetic import default_measurements_for, generate_squat, scenarios
from ratemyform.validation import (
    CM_PER_INCH,
    FIELD_LABELS,
    MEASUREMENT_FIELDS,
    safe_temp_video_path,
    validate_measurements,
    validate_upload,
    validate_video_properties,
)

log = logging.getLogger("ratemyform.app")

st.set_page_config(page_title="RateMyForm", page_icon=":material/fitness_center:", layout="wide")

STAGES = {1: "Choose exercise", 2: "Video & measurements", 3: "Results"}
MEASURE_HELP = {
    "torso": "Top of the shoulder (the bony point) down to the side of the hip joint.",
    "femur": "Side of the hip joint down to the middle of the outside of the knee.",
    "lower_leg": "Middle of the outside of the knee down to the ankle bone.",
    "arm": "Top of the shoulder down to the wrist, with the arm straight.",
}
DEMO_SCENARIOS = ["clean_5_reps", "shallow_reps", "forward_lean_long_femur", "forward_lean_short_femur", "chest_drop", "occluded_shoulder"]


# --------------------------------------------------------------------- state
def init_state() -> None:
    defaults = {"stage": 1, "exercise": "squat", "result": None, "unit": "cm", "error": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    for name in MEASUREMENT_FIELDS:
        st.session_state.setdefault(f"m_{name}", None)


def go(stage: int) -> None:
    st.session_state.stage = stage
    st.session_state.error = None


def convert_units() -> None:
    """Keep entered lengths physically the same when the unit is switched."""
    new_unit = st.session_state.unit_choice
    old_unit = st.session_state.unit
    if new_unit == old_unit:
        return
    factor = 1 / CM_PER_INCH if new_unit == "in" else CM_PER_INCH
    for name in MEASUREMENT_FIELDS:
        value = st.session_state.get(f"m_{name}")
        if value is not None:
            st.session_state[f"m_{name}"] = round(value * factor, 1)
    st.session_state.unit = new_unit


# ------------------------------------------------------------------- sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### RateMyForm")
        st.caption("CPSC 490 prototype - CSUF")
        for number, name in STAGES.items():
            marker = ":material/radio_button_checked:" if number == st.session_state.stage else ":material/radio_button_unchecked:"
            st.markdown(f"{marker} **{number}. {name}**" if number == st.session_state.stage else f"{marker} {number}. {name}")
        st.divider()
        st.caption(
            "**Privacy:** your video is analyzed on this computer and its temporary copy is deleted "
            "right after analysis. Nothing is uploaded to an external service."
        )
        st.caption(DISCLAIMER)
        if st.session_state.stage > 1:
            st.button("Start over", on_click=reset, width="stretch")


def reset() -> None:
    st.session_state.stage = 1
    st.session_state.result = None
    st.session_state.error = None


# ------------------------------------------------------------------- stage 1
def stage_exercise() -> None:
    st.title("RateMyForm")
    st.markdown("#### Personalized squat feedback from a smartphone video")
    st.write(
        "Upload a side-view video of your squats and a few body measurements. RateMyForm tracks your "
        "joints with computer vision, counts your reps, and scores each rep against reference ranges "
        "that take your body proportions into account."
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("1. Choose an exercise")
        st.radio("Exercise", ["Squat"], index=0, key="exercise_choice", help="Squats are the only exercise supported in this prototype.")
        st.caption("Planned, not yet supported: " + ", ".join(PLANNED_EXERCISES) + ".")
    with right:
        st.subheader("How to record")
        st.markdown(
            "- **Side view:** place the phone to your left or right, not in front of you.\n"
            "- **Whole body in frame:** head to feet, with a little space around you.\n"
            "- **Distance:** about 2-4 m (6-13 ft) away; the pose model is not designed for people farther "
            "than about 4 m.\n"
            "- **Height:** phone at about hip height, upright and level (lean is measured from vertical).\n"
            "- **Lighting & clothing:** bright, even light; fitted clothing helps the joints show.\n"
            "- **Reps:** start standing tall, do 3-10 reps, and finish standing.\n"
            "- **Only you in view:** avoid recording other people (bystanders' privacy, and the model "
            "tracks only one person)."
        )
    st.button("Continue", type="primary", on_click=go, args=(2,), icon=":material/arrow_forward:")


# ------------------------------------------------------------------- stage 2
def measurement_inputs():
    st.subheader("Body measurements")
    st.radio(
        "Units",
        ["cm", "in"],
        index=0 if st.session_state.unit == "cm" else 1,
        horizontal=True,
        key="unit_choice",
        on_change=convert_units,
    )
    unit = st.session_state.unit
    cols = st.columns(2)
    for k, name in enumerate(MEASUREMENT_FIELDS):
        with cols[k % 2]:
            st.number_input(
                f"{FIELD_LABELS[name]} length ({unit})",
                min_value=0.0,
                max_value=500.0,
                step=0.5,
                format="%.1f",
                key=f"m_{name}",
                help=MEASURE_HELP[name],
            )
    with st.expander("How to measure"):
        st.markdown(
            "Measure on one side of the body with a flexible tape while standing straight. "
            "Ask someone to help if you can.\n\n"
            + "\n".join(f"- **{FIELD_LABELS[n]}:** {MEASURE_HELP[n]}" for n in MEASUREMENT_FIELDS)
            + "\n\nThe app compares these with proportions estimated from the video and warns you if they "
            "look inconsistent. Arm length is collected (per our proposal) but is not used in squat scoring."
        )
    values = {name: st.session_state.get(f"m_{name}") for name in MEASUREMENT_FIELDS}
    if all(v is None for v in values.values()):
        st.info("Enter all four measurements to continue.")
        return None
    check = validate_measurements(values, unit, DEFAULT_CONFIG.limits)
    for e in check.errors:
        st.error(e, icon=":material/error:")
    for w in check.warnings:
        st.warning(w, icon=":material/warning:")
    if check.ok and check.measurements:
        r = check.measurements.ratios()
        st.caption(f"Femur/torso ratio {r.femur_to_torso:.2f} - lower-leg/femur ratio {r.lower_leg_to_femur:.2f}")
    return check.measurements if check.ok else None


def run_video_analysis(uploaded, measurements) -> None:
    progress = st.progress(0.0, text="Starting analysis...")
    try:
        with tempfile.TemporaryDirectory(prefix="ratemyform_") as tmp:
            path = safe_temp_video_path(tmp, uploaded.name)
            path.write_bytes(uploaded.getvalue())
            info = probe_video(path)
            props = validate_video_properties(info.fps, info.frame_count, info.width, info.height, DEFAULT_CONFIG.limits)
            if not props.ok:
                st.session_state.error = " ".join(props.errors)
                return
            result = analyze_video(
                path,
                measurements,
                DEFAULT_CONFIG,
                progress=lambda f, msg: progress.progress(min(1.0, max(0.0, f)), text=msg),
            )
            result.warnings = props.warnings + result.warnings
        # The temporary folder (and the video copy) is deleted here.
        st.session_state.result = result
        st.session_state.stage = 3
    except ModelNotFoundError as exc:
        st.session_state.error = str(exc)
    except VideoReadError as exc:
        st.session_state.error = f"Could not read the video: {exc}"
    except Exception:  # noqa: BLE001 - show a safe message, log details locally
        log.exception("Unexpected analysis failure")
        st.session_state.error = "Something went wrong while analyzing the video. Please try a different file."
    finally:
        progress.empty()


def run_demo(name: str) -> None:
    spec, _desc = scenarios()[name]
    syn = generate_squat(spec)
    result = analyze_sequence(syn.sequence, default_measurements_for(spec), DEFAULT_CONFIG)
    result.processing["synthetic_scenario"] = name
    from ratemyform.overlay import render_synthetic_frame

    for rep in result.rep_detection.reps:
        result.key_frames[rep.rep_number] = render_synthetic_frame(
            syn.sequence, result.frame_metrics, rep.bottom_index, rep.rep_number - 1, DEFAULT_CONFIG
        )
    st.session_state.result = result
    st.session_state.stage = 3
    st.session_state.error = None


def stage_inputs() -> None:
    st.title("Upload your squat video")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Video")
        limits = DEFAULT_CONFIG.limits
        uploaded = st.file_uploader(
            "Side-view squat video",
            type=[e.lstrip(".") for e in limits.allowed_video_extensions],
            help=f"Up to {limits.max_upload_mb:.0f} MB and {limits.max_duration_s:.0f} seconds.",
        )
        upload_ok = False
        if uploaded is not None:
            check = validate_upload(uploaded.name, uploaded.size, limits)
            for e in check.errors:
                st.error(e, icon=":material/error:")
            upload_ok = check.ok
            if upload_ok:
                st.caption(f"Ready: {uploaded.size / 1e6:.1f} MB")
    with right:
        measurements = measurement_inputs()

    ready = upload_ok and measurements is not None
    if st.button("Analyze my squats", type="primary", disabled=not ready, icon=":material/play_arrow:"):
        run_video_analysis(uploaded, measurements)
        if st.session_state.stage == 3:
            st.rerun()
    if st.session_state.error:
        st.error(st.session_state.error, icon=":material/error:")

    st.divider()
    with st.expander("No video yet? Try a SYNTHETIC demo"):
        st.write(
            "These are computer-generated stick-figure squats with known answers, used to test the "
            "analysis logic. They are **not** real people or real measurements. The demo uses the "
            "synthetic body's own proportions instead of the values entered above."
        )
        all_scenarios = scenarios()
        choice = st.selectbox(
            "Synthetic scenario",
            DEMO_SCENARIOS,
            format_func=lambda n: f"{n} - {all_scenarios[n][1]}",
        )
        if st.button("Run synthetic demo", icon=":material/science:"):
            run_demo(choice)
            st.rerun()
    st.button("Back", on_click=go, args=(1,), icon=":material/arrow_back:")


# ------------------------------------------------------------------- stage 3
def fmt_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.1f} / 10"


def rows_to_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def show_rep_details(result: AnalysisResult) -> None:
    rows = result.reps_table()
    if not rows:
        st.info("No complete reps were counted, so there are no per-rep results.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Angles are read at each rep's lowest point (smallest knee angle). Knee and hip angles come from "
        "MediaPipe's 3D landmarks; torso lean, shin lean, and depth come from the 2D image. "
        "'n/a' or a low-confidence flag means a value could not be measured reliably."
    )
    for m, b, p in zip(result.rep_measurements, result.baseline_scores, result.personalized_scores):
        title = f"Rep {m.rep_number}: personalized {fmt_score(p.score)} - baseline {fmt_score(b.score)}"
        if m.low_confidence:
            title += " (low confidence)"
        with st.expander(title):
            c1, c2 = st.columns([1, 1])
            with c1:
                if m.rep_number in result.key_frames:
                    st.image(result.key_frames[m.rep_number], caption=f"Rep {m.rep_number} at its lowest point", width="stretch")
            with c2:
                def show(label, value, unit="deg", key=None):
                    ok = result_reliable(m, key)
                    text = "not measured" if value is None else f"{value:.0f} {unit}" if unit else f"{value:.2f}"
                    st.markdown(f"- **{label}:** {text}" + ("" if ok else " *(not scored: unreliable)*"))

                show("Knee angle", m.knee_angle_deg, key="knee_angle")
                show("Hip angle", m.hip_angle_deg, key="hip_angle")
                show("Torso lean", m.torso_lean_deg, key="torso_lean")
                show("Shin lean", m.shin_lean_deg, key="shin_lean")
                st.markdown(f"- **Depth:** {m.depth_class} (hip joint vs. knee joint height)")
                show("Chest drop while standing up", m.chest_drop_deg, key="chest_drop")
                st.markdown("**Personalized deductions**")
                if p.deductions:
                    for d in p.deductions:
                        st.markdown(f"- {d.message}")
                elif p.score is None:
                    st.markdown("- Not scored (no reliable measurements).")
                else:
                    st.markdown("- None")
                if b.deductions != p.deductions:
                    st.markdown("**Baseline deductions (same reference for everyone)**")
                    for d in b.deductions or []:
                        st.markdown(f"- {d.message}")
                    if not b.deductions:
                        st.markdown("- None")
                for note in m.reliability_notes:
                    st.caption(note)


def result_reliable(m, key) -> bool:
    return True if key is None else bool(m.reliable.get(key))


def show_personalization(result: AnalysisResult) -> None:
    meas = result.measurements
    r = meas.ratios()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Your measurements**")
        st.dataframe(
            [
                {"measurement": "Torso", "cm": round(meas.torso_cm, 1)},
                {"measurement": "Femur", "cm": round(meas.femur_cm, 1)},
                {"measurement": "Lower leg", "cm": round(meas.lower_leg_cm, 1)},
                {"measurement": "Arm", "cm": round(meas.arm_cm, 1)},
            ],
            hide_index=True,
        )
    with c2:
        st.markdown("**Ratios: entered vs. estimated from the video**")
        st.dataframe(
            [
                {
                    "ratio": c.name.replace("_", " "),
                    "entered": c.entered,
                    "video estimate": c.video_estimate,
                    "difference": None if c.relative_difference is None else f"{c.relative_difference:.0%}",
                    "consistent": {True: "yes", False: "check", None: "n/a"}[c.consistent],
                }
                for c in result.ratio_checks
            ],
            hide_index=True,
        )
        st.caption(
            "Video estimates use MediaPipe's 3D joint positions while you stand, which do not match tape "
            "measurements exactly. A mismatch is a prompt to double-check, not proof of an error."
        )
    st.markdown("**Reference values used**")
    st.dataframe(
        [
            {
                "reference": "Depth target (knee angle at bottom, max)",
                "baseline": f"{result.baseline.depth_target_knee_angle_deg:.0f} deg",
                "personalized": f"{result.personalized.depth_target_knee_angle_deg:.0f} deg",
            },
            {
                "reference": "Torso lean at bottom (max)",
                "baseline": f"{result.baseline.torso_lean_limit_deg:.0f} deg",
                "personalized": f"{result.personalized.torso_lean_limit_deg:.1f} deg",
            },
            {
                "reference": "Chest drop while standing up (max)",
                "baseline": f"{result.baseline.chest_drop_threshold_deg:.0f} deg",
                "personalized": f"{result.personalized.chest_drop_threshold_deg:.0f} deg",
            },
        ],
        hide_index=True,
    )
    st.markdown("**Why the personalized values differ**")
    for line in result.personalized.rationale:
        st.markdown(f"- {line}")
    st.caption(f"Femur/torso {r.femur_to_torso:.2f}, lower-leg/femur {r.lower_leg_to_femur:.2f}, arm/torso {r.arm_to_torso:.2f}.")


def stage_results() -> None:
    result: AnalysisResult | None = st.session_state.result
    if result is None:
        go(2)
        st.rerun()
        return
    st.title("Your results")
    if result.source == "synthetic":
        st.warning(
            "SYNTHETIC DEMO DATA: these results come from a computer-generated stick figure, not a real "
            f"person or video (scenario: {result.processing.get('synthetic_scenario')}).",
            icon=":material/science:",
        )
    st.caption(DISCLAIMER)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Personalized score", fmt_score(result.overall_personalized), help="Average of the scored reps, using reference ranges adjusted for your proportions.")
    c2.metric("Baseline score", fmt_score(result.overall_baseline), help="Same reference ranges for everyone, for comparison.")
    c3.metric("Reps counted", result.rep_count)
    c4.metric("Camera check", result.orientation.label.replace("_", " "), help=result.orientation.message)
    if result.overall_personalized is not None and result.overall_baseline is not None:
        change = round(result.overall_personalized - result.overall_baseline, 1)
        if change:
            st.caption(f"Personalization changed the overall score by {change:+.1f} points compared with the baseline.")

    st.subheader("Feedback")
    for tip in result.feedback:
        st.markdown(f"- {tip}")
    if result.warnings:
        st.subheader("Data quality and limitations")
        for w in result.warnings:
            st.warning(w, icon=":material/info:")

    tabs = st.tabs(["Reps", "Charts", "Skeleton overlay", "Personalization", "Export"])
    with tabs[0]:
        show_rep_details(result)
    with tabs[1]:
        if result.frame_metrics.times_s.size:
            st.image(charts.to_png(charts.knee_angle_timeline(result)), width="stretch")
            st.caption("A rep is counted when the knee angle drops below 140 deg and then rises above 160 deg (shaded areas).")
        if result.comparison:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(charts.to_png(charts.score_comparison(result)), width="stretch")
            with col_b:
                st.image(charts.to_png(charts.bottom_metrics_chart(result)), width="stretch")
    with tabs[2]:
        if result.overlay_video:
            st.video(result.overlay_video, format=result.overlay_mime)
            st.caption("Highlighted joints are the side of the body being analyzed. Grey parts were not clearly visible.")
        elif result.source == "synthetic":
            st.info("Synthetic demos have no video. Rep key frames are drawn on a blank canvas in the Reps tab.")
        else:
            st.info("The overlay video is not available.")
        if result.key_frames:
            cols = st.columns(min(4, len(result.key_frames)))
            for k, (rep_number, png) in enumerate(sorted(result.key_frames.items())):
                cols[k % len(cols)].image(png, caption=f"Rep {rep_number} bottom", width="stretch")
    with tabs[3]:
        show_personalization(result)
    with tabs[4]:
        report = json.dumps(result.to_dict(), indent=2)
        st.download_button("Download full report (JSON)", report, file_name="ratemyform_report.json", mime="application/json")
        st.download_button("Download rep table (CSV)", rows_to_csv(result.reps_table()), file_name="ratemyform_reps.csv", mime="text/csv")
        st.download_button(
            "Download frame-by-frame angles (CSV)",
            rows_to_csv(result.timeseries_rows()),
            file_name="ratemyform_timeseries.csv",
            mime="text/csv",
        )
        st.caption("Reports contain your measurements, angles, and scores. They do not contain your video.")
        with st.expander("Processing details"):
            st.json({"video": result.video_info, "processing": result.processing})

    b1, b2 = st.columns([1, 5])
    b1.button("Analyze another video", on_click=go, args=(2,), icon=":material/replay:")


def main() -> None:
    init_state()
    sidebar()
    stage = st.session_state.stage
    if stage == 1:
        stage_exercise()
    elif stage == 2:
        stage_inputs()
    else:
        stage_results()


main()
