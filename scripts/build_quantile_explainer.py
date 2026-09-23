"""Build a vector poster from the saved quantile confirmation results."""

import csv
import html
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A2
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/quantile_confirmation_20260923"
OUTPUT = ROOT / "output/pdf/quantile_landscape_explainer.pdf"
WIDTH, HEIGHT = A2
BLUE = "#167db5"
GREEN = "#279d29"
INK = "#172127"
GREY = "#56636a"
PALE = "#edf5f8"
PROTEINS = ["DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV"]
PROSE = []


def rows(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A2)
    pdf.setTitle("From single mutants to a calibrated landscape")
    pdf.setAuthor("Mutascape")

    def text(x, top, value, size=17, font="Times-Roman", colour=INK):
        pdf.setFillColor(colors.HexColor(colour))
        pdf.setFont(font, size)
        pdf.drawString(x, HEIGHT - top - size, value)
        PROSE.append(value)

    def para(x, top, value, width, size=17, colour=INK):
        style = ParagraphStyle(
            "body",
            fontName="Times-Roman",
            fontSize=size,
            leading=size * 1.27,
            textColor=colors.HexColor(colour),
            spaceAfter=0,
        )
        paragraph = Paragraph(value, style)
        _, height = paragraph.wrap(width, 1000)
        paragraph.drawOn(pdf, x, HEIGHT - top - height)
        PROSE.append(value)
        return top + height

    def line(x1, top1, x2, top2, colour="#c5d2d8", weight=0.7):
        pdf.setStrokeColor(colors.HexColor(colour))
        pdf.setLineWidth(weight)
        pdf.line(x1, HEIGHT - top1, x2, HEIGHT - top2)

    def box(x, top, width, height, fill=PALE):
        pdf.setFillColor(colors.HexColor(fill))
        pdf.rect(x, HEIGHT - top - height, width, height, fill=1, stroke=0)

    def arrow(x1, x2, top):
        line(x1, top, x2, top, GREY, 1)
        line(x2 - 7, top - 4, x2, top, GREY, 1)
        line(x2 - 7, top + 4, x2, top, GREY, 1)

    margin = 48
    usable = WIDTH - 2 * margin
    column = (usable - 42) / 2
    right = margin + column + 42

    text(margin, 30, "MUTASCAPE  /  MODEL EXPLAINER  /  23 SEPTEMBER 2026", 12, "Helvetica", GREY)
    text(margin, 57, "From single mutants to a calibrated landscape", 37, "Times-Bold")
    text(
        margin,
        110,
        "Better fitness distributions. Similar search rankings.",
        25,
        "Times-Roman",
        BLUE,
    )
    para(
        margin,
        151,
        "<b>Singles and a FASTA file are insufficient.</b> This model also needs measured"
        " combinations from the same assay.",
        usable,
        19,
    )
    line(margin, 191, WIDTH - margin, 191)

    text(margin, 210, "1  Define a small mutation library", 24, "Times-Bold")
    para(
        margin,
        250,
        "The FASTA supplies residue identities and position numbers. The assay supplies "
        "wild-type fitness and single-mutant fitness. Use one-based positions.",
        column,
        18,
    )
    para(
        margin,
        329,
        "Choose the positions and allowed residues first. Each allowed substitution needs"
        " a measured single. A full-protein DMS usually defines too many combinations.",
        column,
        17,
    )

    text(right, 210, "2  Measure random combinations", 24, "Times-Bold")
    para(
        right,
        250,
        "Sample uniformly from combinations with at least two substitutions. Measure them"
        " on the same fitness scale as the wild type and singles.",
        column,
        18,
    )
    para(
        right,
        329,
        "Our stronger evidence uses 192 combinations. A pilot selected for high fitness "
        "would bias the fitted distribution. These measurements are an extra experimental"
        " cost.",
        column,
        17,
    )

    text(margin, 414, "3  Fit the existing pilot model", 24, "Times-Bold")
    para(
        margin,
        454,
        "Singles set the individual mutation effects. The pilot fits two terms for "
        "mutation count and calibrates the strength of random pair effects.",
        column,
        18,
    )
    para(
        margin,
        533,
        "A sigmoid maps latent scores to fitness. Each seed supplies fixed pair effects. "
        "The fit does not learn which specific pairs interact.",
        column,
        17,
    )

    text(right, 414, "4  Match the pilot distribution", 24, "Times-Bold")
    para(
        right,
        454,
        "Each draw scores every supported combination. The new model maps each score's "
        "percentile to the same percentile of measured pilot fitness.",
        column,
        18,
    )
    para(
        right,
        533,
        "This changes fitness spacing. Combination order stays the same, except for added"
        " ties. Wild type and singles retain their measured values exactly.",
        column,
        17,
    )

    box(margin, 618, usable, 112)
    stages = [
        (margin + 25, "Generated score"),
        (margin + 409, "Percentile"),
        (margin + 746, "Pilot fitness quantile"),
    ]
    for x, label in stages:
        text(x, 632, label, 22, "Times-Bold")
    arrow(margin + 228, margin + 383, 650)
    arrow(margin + 538, margin + 719, 650)
    text(margin + 25, 677, "f_new(x) = Q_pilot( F_draw( f_pilot(x) ) )", 19, "Courier")
    text(margin + 645, 677, "Example: 80th percentile maps to 80th percentile.", 16)

    text(margin, 752, "What changed across five proteins?", 27, "Times-Bold")
    para(
        margin,
        791,
        "Both models use the same 192 calibration combinations. Classifier tests use "
        "1,000 separate genotypes per protein.",
        usable,
        17,
    )
    text(margin + 12, 827, "Current pilot", 16, colour=BLUE)
    line(margin, 839, margin + 8, 839, BLUE, 2)
    text(margin + 194, 827, "Quantile-calibrated pilot", 16, colour=GREEN)
    line(margin + 182, 839, margin + 190, 839, GREEN, 2)
    text(
        margin + 478,
        827,
        "Points show medians. Bars show the range across three draws.",
        16,
        colour=GREY,
    )

    metrics = rows("metrics.csv")
    structure = rows("structure.csv")
    panel_width = usable / 3
    specs = [
        ("C2ST using assay features", "auc", "False", 0.48, 1.0, [0.5, 0.7, 0.9, 1.0]),
        ("C2ST with sequence added", "auc", "True", 0.48, 1.0, [0.5, 0.7, 0.9, 1.0]),
        ("Walsh / ANOVA distance", "walsh_l1", None, 0, 0.85, [0.0, 0.2, 0.4, 0.6, 0.8]),
    ]
    for panel, (title, metric, sequence, low, high, ticks) in enumerate(specs):
        left = margin + panel * panel_width
        text(left, 868, title, 20, "Times-Bold")
        plot_left, plot_right = left + 90, left + panel_width - 24
        plot_top, plot_bottom = 917, 1116

        def xpos(value):
            return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

        line(plot_left, plot_bottom, plot_right, plot_bottom, INK)
        if sequence is not None:
            pdf.setDash(3, 4)
            line(xpos(0.5), plot_top - 12, xpos(0.5), plot_bottom, "#9aabb3")
            pdf.setDash()
        for tick in ticks:
            x = xpos(tick)
            line(x, plot_bottom, x, plot_bottom + 5, INK)
            text(x - 10, plot_bottom + 9, f"{tick:.1f}", 14, colour=GREY)
        for index, protein in enumerate(PROTEINS):
            y = plot_top + index * 42
            text(left, y - 9, protein, 16)
            data = structure if sequence is None else metrics
            for model, colour, offset in [("reference", BLUE, -5), ("quantile", GREEN, 5)]:
                selected = [
                    float(r[metric])
                    for r in data
                    if r["protein"] == protein
                    and r["pilot_size"] == "192"
                    and r["model"] == model
                    and (
                        sequence is None
                        or (r["classifier"] == "histogram" and r["sequence"] == sequence)
                    )
                ]
                if not selected:
                    if model == "reference":
                        text(plot_left + 27, y - 9, "Incomplete library", 14, colour=GREY)
                    continue
                centre = statistics.median(selected)
                yy = y + offset
                line(xpos(min(selected)), yy, xpos(max(selected)), yy, colour, 1)
                for end in [min(selected), max(selected)]:
                    line(xpos(end), yy - 4, xpos(end), yy + 4, colour, 1)
                pdf.setFillColor(colors.HexColor(colour))
                pdf.circle(xpos(centre), HEIGHT - yy, 3.4, fill=1, stroke=0)
        label = (
            "0.5 is chance discrimination"
            if sequence is not None
            else "0 is an identical order spectrum"
        )
        text(left + 47, 1155, label, 15, colour=GREY)

    para(
        margin,
        1191,
        "C2ST improves on three proteins. Values near 0.7 still show detectable mismatch."
        " AUC is not classification accuracy. There is no universal passing value.",
        column,
        17,
    )
    para(
        right,
        1191,
        "Spectral distance improves on three of four proteins. DHFR worsens from 0.181 to"
        " 0.688. A nonlinear fitness map can change interaction-order variance.",
        column,
        17,
    )
    line(margin, 1265, WIDTH - margin, 1265)

    text(margin, 1281, "Search rankings change little", 24, "Times-Bold")
    para(
        margin,
        1319,
        "We tested 12 methods, ten search seeds, and budgets of 48 or 96 queries. Both "
        "models use 192 extra calibration measurements.",
        column,
        17,
    )
    xvals = [margin, margin + 292, margin + 405]
    for x, label in zip(xvals, ["Measure", "Current pilot", "New model"]):
        text(x, 1381, label, 15, "Times-Bold")
    line(margin, 1408, margin + column, 1408)
    search = [r for r in rows("search_comparison.csv") if r["pilot_size"] == "192"]
    for y, field, label, percentage in [
        (1418, "winner_accuracy", "Agreement on clear method winners", True),
        (1451, "kendall_tau", "Kendall rank correlation", False),
    ]:
        text(margin, y, label, 16)
        for x, model, colour in [(xvals[1], "reference", BLUE), (xvals[2], "quantile", GREEN)]:
            value = statistics.mean(float(r[field]) for r in search if r["model"] == model)
            text(x, y, f"{value:.1%}" if percentage else f"{value:.3f}", 18, colour=colour)
    para(
        margin,
        1491,
        "Only method pairs with a clear measured difference count towards winner "
        "agreement. Means give each protein and query budget equal weight.",
        column,
        15,
    )

    text(right, 1281, "Use the model", 24, "Times-Bold")
    code = [
        "from mutascape.experimental import (",
        "    QuantileCalibratedLandscape,",
        ")",
        "model = QuantileCalibratedLandscape.fit(",
        "    sequence, singles, pilot)",
        "landscape = model.draw(seed=42)",
        "fitness = landscape.evaluate_batch(variants)",
    ]
    box(right, 1323, column, 163)
    for index, value in enumerate(code):
        text(right + 12, 1333 + index * 21, value, 14.5, "Courier")
    para(
        right,
        1497,
        "Both tables need mutations and fitness columns. Include wild type as {} in "
        "singles. Repeated queries return the same fitness.",
        column,
        15,
    )

    line(margin, 1558, WIDTH - margin, 1558)
    para(
        margin,
        1573,
        "Limits. One calibration set per protein. Draw ranges are not confidence "
        "intervals. Spectra include calibration genotypes. CreiLOV uses a binary "
        "projection. Exact enumeration defaults to one million variants.",
        usable,
        14,
    )
    para(
        margin,
        1617,
        "Sources. Lopez-Paz and Oquab, ICLR 2017, arxiv.org/abs/1610.06545. Model and "
        "data in mutascape.experimental.quantile and "
        "results/quantile_confirmation_20260923. These results compare two pilot models, "
        "not the earlier singles-only benchmark.",
        usable,
        13,
    )
    pdf.linkURL("https://arxiv.org/abs/1610.06545", (margin, 20, margin + 535, 54), relative=0)
    pdf.save()
    import re

    prose = "\n\n".join(re.sub("<[^>]+>", "", html.unescape(t)) for t in PROSE)
    (ROOT / "tmp/pdfs/quantile_poster_text.txt").write_text(prose)
    print(OUTPUT)


if __name__ == "__main__":
    main()
