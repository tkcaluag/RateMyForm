# Future development roadmap

Ordered by what would most improve the credibility of the project.

## Next 2-3 weeks (before the final presentation, Dec 7-18 per our schedule)

1. **Real-video evaluation (highest priority).** Record consenting teammates using
   the protocol in [data/real_videos/README.md](../data/real_videos/README.md): side,
   45 deg, and front views; 2/3/4 m; bright/dim; fitted/loose. Count reps by hand.
   Run `scripts/evaluate_videos.py`. Report exact-count rate, mean rep error,
   detection rate, and ms per frame.
2. **Merge with the team repository** (steps in [AUDIT.md](AUDIT.md#merging-with-the-team-repository)).
3. **Replace PROTOTYPE reference values** with Billy's researched guidelines, and
   cite each source in `config.py`.
4. **Define the ">75% form-ranking accuracy" goal.** Decide what is labeled (per-rep
   "good/needs work"?), who labels it (a certified trainer?), how many videos, and
   how agreement is measured.

## Next semester (CPSC 491)

- Validate the personalization heuristic: compare predicted vs. measured torso lean
  for participants with different femur/torso ratios.
- Personalize depth using ankle mobility (measured shin angle) instead of only body
  ratios.
- Add front-view analysis for knee valgus and left/right symmetry (two-video or
  two-angle mode).
- Heel-rise detection from the heel and foot-index landmarks.
- More exercises (deadlift, push-up) as new rule sets on the same pipeline.
- An interactive calibration step (live camera check before recording).
- Session history with explicit consent, a retention period, and deletion.

## Generative AI feedback (researched option, not built)

Status Report 2 lists "Check whether AI could help explain exercise mistakes, and
whether simpler rules would work better."

| | Rules-based (current) | LLM-generated |
|---|---|---|
| Traceability | Every sentence maps to a measured number and a reference | Can state things the video does not show (Tracy's challenge: "AI might give advice that does not match what the video actually shows") |
| Consistency | Identical input → identical text | Varies between runs unless constrained |
| Privacy / cost | Fully local, free | Usually a cloud API (sends data out, needs keys, costs money); local models need more hardware |
| Language quality | Template-like | More natural, can answer follow-up questions |
| Safety | No medical claims by construction | Needs guardrails against diagnostic or "safe/unsafe" language |

A reasonable hybrid for later: keep rules for **what** is wrong, and optionally let a
model **rephrase** only the structured deductions (numbers in, wording out), with the
original rule text shown alongside. It must stay optional, off by default, and
documented in the privacy notes.

## Scaling (only if the project continues past the course)

Run pose detection on the phone or in the browser, so videos never leave the device.
A server would then only store opt-in summaries. Add background workers and a job
queue for server-side analysis, plus rate limits and size limits. See
[ARCHITECTURE.md](ARCHITECTURE.md#scaling-notes-hypothetical-not-tested).
