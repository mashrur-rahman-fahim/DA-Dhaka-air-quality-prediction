# Paper draft — ICCIT 2026

`main.tex` is a complete IEEE conference paper built from the results in
`../dhaka_pm25.ipynb`. It compiles standalone: the two figures are drawn with
`pgfplots` from the real numbers, so there are no image files to supply.

## How to use it

1. Open the IEEE conference template on Overleaf
2. Replace its `main.tex` with this one
3. Compile with pdfLaTeX

## Before you submit — four things

**1. Verify the references.** Entries marked `% [VERIFY]` come from search
results, not from the publisher record. Author lists, volumes, issues and page
numbers must be checked. The claim in Section I that a study reports
$R^2 = 0.998$ is attributed to `\cite{dhaka_compare}` — **confirm that
attribution is correct before submission.** Misattributing a number to the wrong
paper is worse than omitting it.

**2. It is over six pages.** ICCIT allows six, including figures and references.
Expect roughly seven or eight as written. Trim in this order, which loses the
least:

- Section II-A can lose two sentences
- Section III-C: shorten the five bullets to three lines of prose
- Table `tab:cities` can merge into `tab:cities_res` — the Region column is
  decorative and Mean PM2.5 appears only once
- Section VI-C: the limitations can drop to one paragraph
- Fig. `fig:acf` is the most expendable figure; the autocorrelation numbers are
  already stated in the text

**3. Keep it anonymous.** ICCIT reviews double-blind and states that a
manuscript carrying author names, affiliations, addresses or e-mail addresses is
rejected immediately. The author block is anonymised and the code repository
link is deliberately absent. Restore both only for camera-ready.

**4. Check the arithmetic in Table `tab:leak`.** Condition A in that table is
35.02, while Section V-A quotes 34.37 for the same model. They differ because
they use different test sets — the shared 1,567-hour intersection versus the
full test period. The text says so, but a reviewer may still ask.

## Where each number comes from

| Paper | Notebook |
|---|---|
| Section III quality figures | Part 1, Q1–Q5 |
| Table `tab:horizon` | Part 4 Step 37, Part 5 Step 43 |
| Table `tab:leak` | Part 6 Steps 48–50 |
| Table `tab:cities_res`, Fig. `fig:cities` | Part 7 Steps 52–54 |
| Fig. `fig:acf` | Part 1, Q11 |
| Seed and fold stability | Part 5 Step 44, Part 4 Step 38 |

Every figure in the paper was checked against the committed notebook output.

## What a reviewer is most likely to ask for

- **Per-city significance tests.** Dhaka has them; the twelve-city replication
  reports a sign test only. About ten minutes of compute.
- **A fitted SARIMA baseline.** Named as a limitation rather than run.
- **Meteorology.** The obvious next study, not a fixable gap in this one.
