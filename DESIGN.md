# DESIGN.md — UI Design Guidelines

This document defines the visual and interaction language for the platform. It exists so every screen, whoever builds it, feels like one product.

## 1. Design Principle

The product's core tension: **an autonomous agent is doing complex, multi-step work the user can't directly see, and the UI's entire job is making that legible and trustworthy.** Every design decision should be evaluated against: *does this make the system's reasoning more visible, or does it hide it behind a prettier surface?*

This is not a marketing site. Avoid template SaaS-dashboard defaults (generic card grids, decorative gradients, stat-tile hero sections) unless they're actually the clearest way to show something real. Favor clarity and information density appropriate to a technical/analytical tool over visual flourish.

## 2. Visual Identity

### 2.1 Palette
Working system — treat as a starting point to refine once real screens exist, not a locked spec:

- `--bg-base`: #0F1115 (near-black, not pure black — reduces eye strain for long analytical sessions)
- `--bg-surface`: #1A1D24 (cards, panels)
- `--bg-surface-raised`: #23262E (modals, active/focused panels)
- `--text-primary`: #E8E9ED
- `--text-secondary`: #9298A5
- `--border-subtle`: #2C303A
- `--accent-primary`: #4F8CFF (used for active/running states, primary actions — a working blue, not decorative)
- `--accent-success`: #3FBF7F (completed stages, good metrics)
- `--accent-warning`: #E8A33D (caveats, flagged issues, needs-review states)
- `--accent-danger`: #E8544F (errors, failed runs)

Rationale: a dark, low-saturation base keeps focus on data and status color-coding, which is doing real semantic work (running/done/warning/error) rather than decoration. Avoid warm-cream-plus-terracotta and pure-black-plus-neon-accent defaults — this is a working tool, not a landing page.

### 2.2 Typography
- **Display/headings**: a grounded, slightly condensed sans (e.g., Inter Tight or Söhne) — used with restraint, mainly for section titles and report headlines, not everywhere.
- **Body/UI**: Inter or system-ui stack for maximum legibility at small sizes across dense screens.
- **Data/mono**: JetBrains Mono or similar for column names, code exports, raw logs, metric values — anything that is literally data should look like data, distinct from prose.
- Type scale: establish a clear hierarchy (e.g., 12/14/16/20/28/36px) and use it consistently; don't introduce one-off sizes per screen.

### 2.3 Iconography & Imagery
- No decorative illustration or stock imagery. This product's "hero" content is real data and real pipeline state — lead with that, not an illustrated abstraction of "AI."
- Icons: a single consistent set (e.g., Lucide/Phosphor), used only where they add scannability (status icons, action icons), never purely decorative.

## 3. Layout System

- 8px base spacing unit; all margins/padding as multiples of 8 (4px allowed for tight inline spacing only).
- Max content width for report/profile views: ~1200px, centered, with generous side margins on larger screens — this is a reading/analysis tool, not an edge-to-edge dashboard.
- Live pipeline view is the exception: full-width, since the graph/stage visualization benefits from horizontal space.
- Border-radius: small and consistent (6–8px) — enough to soften panels without looking playful; avoid 0px (too severe/broadsheet for this product) and avoid large pill-shaped radii (too casual for a technical tool).

## 4. Signature Element

**The pipeline stage tracker** is the one element this product should be recognized by: a horizontal (desktop) or vertical (mobile) sequence of named stages, each showing live status via color and a short plain-language annotation of what's happening, with a subtle animated pulse only on the currently-running stage (motion is reserved for this one place — not used elsewhere as ambient decoration). This directly embodies the product's core promise: an opaque agentic process made visible and step-legible. Every other screen should feel calm and static by comparison, so this element reads as meaningful rather than as generic loading chrome.

## 5. Component Guidelines

### 5.1 Status & Confidence Indicators
Three distinct visual treatments, never conflated:
- **Detected** (data-driven fact): neutral badge, e.g., gray background, checkmark icon.
- **Inferred** (LLM judgment call needing possible confirmation): accent-warning colored badge, distinct icon (e.g., a small "?" or sparkle), always paired with an edit affordance.
- **Flagged issue** (leakage risk, quality problem): accent-warning/danger badge, always paired with a one-line explanation on hover/click — never a bare icon with no context.

### 5.2 Data Display
- Tables: monospace for values, sans for headers, zebra striping only if row count is high enough to need it (>15 rows visible).
- Never display more than the backend-capped sample size in any raw data table — pair with a persistent, quiet label like "12 of 340,000 rows shown."
- Charts (feature importance, metric comparisons): consistent color mapping across the whole product (e.g., the winning model is always accent-primary, others always neutral gray) so users build pattern recognition across sessions.

### 5.3 Forms & Inputs
- The use-case intake text input is a first-class, prominent element — treat it closer to a search bar than a generic form field (large, centered, inviting).
- Inline editable fields (target column, metric) use a clear "click to edit" affordance, not a separate edit mode/modal — keeps correction low-friction.

### 5.4 Buttons & Actions
- Primary action: solid accent-primary fill, one per screen/section max.
- Destructive actions (cancel run, delete dataset): accent-danger, always with a confirmation step if irreversible.
- Button copy is always the action itself in active voice ("Run pipeline," "Cancel run," "Download model") — never vague verbs like "Submit" or "Confirm" alone.

### 5.5 Motion
- Reserve orchestrated motion for the pipeline stage tracker (Section 4) and for state transitions that carry real meaning (a stage completing, a checkpoint appearing).
- No decorative page-load animations, no parallax, no ambient background motion. Respect `prefers-reduced-motion` everywhere.

## 6. Content & Voice (applies to all UI copy)

- Write from the user's side of the screen: name things by what people control, not by internal system concepts. A user reviews a "flagged column," not a "PII detection heuristic output."
- Active voice, consistent verb-to-outcome mapping: a button that says "Run pipeline" leads to a status that says "Running," and a completion state that says "Completed" — never a mismatched vocabulary across the same action.
- Errors state what happened and what to do, in the product's voice, never apologetic or vague ("Training failed: target column contains 40% missing values. Choose a different column or review imputation settings.")
- Empty states are invitations to act, not dead space ("No pipelines yet — upload a dataset to get started.")
- Caveats and limitations are stated plainly and are never softened into marketing language — this is where the product's credibility lives.

## 7. Quality Floor (non-negotiable for every screen)

- Responsive down to tablet width at minimum; graceful degradation to mobile for complex views (see PRODUCT.md 4.5).
- Visible keyboard focus states on every interactive element.
- Color is never the only signal for status — always pair with an icon or text label (accessibility, and also useful for colorblind-safe status reading of running/done/error states).
- `prefers-reduced-motion` respected throughout.
- Every screen reviewed against the question in Section 1 before shipping: does this make the system's reasoning more visible, or just prettier?