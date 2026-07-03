# Create a one-slide poster layout as a high-resolution PNG (1920x1080) that can be inserted into PowerPoint.
# We'll draw a title band, three columns (Intro/Data; Methods; Results/Discussion), and a footer.
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Try to load a clean sans font; fall back to default if unavailable
def load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

W, H = 1920, 1080
bg = "white"
img = Image.new("RGB", (W, H), bg)
draw = ImageDraw.Draw(img)

# Fonts
title_f   = load_font(60)
subtitle_f= load_font(26)
h_f       = load_font(34)
body_f    = load_font(22)
small_f   = load_font(18)
ref_f     = load_font(16)

# Layout measurements
margin = 64
title_h = 150
footer_h = 100
gutter = 30

content_top = margin + title_h
content_bottom = H - footer_h - margin
content_height = content_bottom - content_top

# Column widths
col_w = (W - 2*margin - 2*gutter) // 3
col_x = [margin + i*(col_w + gutter) for i in range(3)]

# Helpers
def text_block(x, y, w, text, font, line_spacing=6):
    # Wrap text to fit width w
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        # greedy wrap by words with width measurement
        words = paragraph.split(" ")
        cur = ""
        for wword in words:
            test = (cur + " " + wword).strip()
            bbox = draw.textbbox((0,0), test, font=font)
            if bbox[2]-bbox[0] <= w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = wword
        if cur:
            lines.append(cur)
    # Draw
    y_cursor = y
    for ln in lines:
        draw.text((x, y_cursor), ln, fill="black", font=font)
        # compute line height
        bbox = draw.textbbox((0,0), ln if ln else " ", font=font)
        lh = bbox[3]-bbox[1]
        y_cursor += lh + line_spacing
    return y_cursor

def section_box(x, y, w, heading, body, y_pad=14, box_outline="#DDDDDD"):
    # heading
    draw.rectangle([x, y, x+w, y+8], fill="#000000")  # small accent rule
    y += 18
    draw.text((x, y), heading, fill="black", font=h_f)
    y += 44
    # body
    end_y = text_block(x, y, w, body, body_f, line_spacing=6)
    # bottom spacing
    end_y += y_pad
    # optional light outline
    draw.rectangle([x-10, y-62, x+w+10, end_y+8], outline=box_outline, width=1)
    return end_y

# Title band
title_text = "Protein-Based Survival Prediction with SuperLearner"
authors_text = "Your Name · Icahn School of Medicine at Mount Sinai"
takeaway_text = "Ensembling penalized Cox and RSF to predict 1- and 3-year overall survival from proteomic features"

# Draw title band
title_y = margin
draw.text((margin, title_y), title_text, fill="black", font=title_f)
title_y += 72
draw.text((margin, title_y), authors_text, fill="black", font=small_f)
title_y += 30
draw.text((margin, title_y), takeaway_text, fill="black", font=subtitle_f)

# Column 1: Introduction + Data
intro_body = (
    "• Proteomic profiles may encode prognostic signal in HNSCC.\n"
    "• SuperLearner (SL) combines complementary survival learners.\n"
    "• We assess discrimination and calibration at 1 and 3 years."
)
data_body = (
    "Cohort\n"
    "• Dataset: CPTAC HNSCC (train via 5-fold CV)\n"
    "• n = [ ], events = [ ], censoring = [ ]%\n"
    "• Predictors: ~9.4k proteins (log-normalized; batch-adjusted)\n"
    "• Outcome: OS_days, OS_event\n\n"
    "Preprocessing\n"
    "• Missingness: median/KNN imputation\n"
    "• Standardization: z-score per feature\n"
    "• Feature screen: top K (200–500) by univariate Cox / variance"
)

y_cursor = content_top
y_cursor = section_box(col_x[0], y_cursor, col_w, "Introduction", intro_body)
y_cursor = section_box(col_x[0], y_cursor+10, col_w, "Data & Preprocessing", data_body)

# Column 2: Methods
methods_body = (
    "Base learners\n"
    "• Cox-lasso / elastic net / ridge (glmnet)\n"
    "• Random Survival Forest (RSF)\n"
    "• (Avoid AFT gamma/ggamma for p ≫ n)\n\n"
    "Training\n"
    "• SuperLearner stacking with non-negative weights\n"
    "• 5-fold CV; seed = 11\n"
    "• Metric for weighting: Harrell C (fast) or Uno’s C (robust)\n"
    "• Optional penalty mask to force-in specific covariates\n\n"
    "Evaluation\n"
    "• Discrimination: C-index (Harrell or Uno)\n"
    "• Calibration: IPCW Brier @365d and @1095d; IBS (0–3y)\n"
    "• Interpretability: RSF VI; non-zero Cox coefficients"
)
section_box(col_x[1], content_top, col_w, "Methods", methods_body)

# Column 3: Results + Discussion (placeholders)
results_body = (
    "Key results (placeholders)\n"
    "• SL C-index: 0.•• (best single: 0.••)\n"
    "• Brier@1y: 0.•••; Brier@3y: 0.•••\n"
    "• Top-weighted learners: [Cox-lasso, RSF, …]\n"
    "• KM curves by risk tertiles: clear separation\n"
)
discussion_body = (
    "• Protein-only SL provides robust discrimination; calibration acceptable across horizons.\n"
    "• Proteomic signal complements staging; potential clinical utility.\n"
    "• Limitations: p ≫ n; cohort size; batch effects; need external validation.\n"
    "• Next: external validation; pathway enrichment; prospective evaluation."
)
yc = section_box(col_x[2], content_top, col_w, "Results (Summary)", results_body)
section_box(col_x[2], yc+10, col_w, "Discussion", discussion_body)

# Footer
footer_y = H - footer_h + 20
draw.line([(margin, footer_y-10), (W-margin, footer_y-10)], fill="#CCCCCC", width=2)
refs = ("References: Uno et al. (time-dependent concordance); Friedman et al. (glmnet); "
        "Ishwaran et al. (Random Survival Forests)")
draw.text((margin, footer_y), refs, fill="black", font=ref_f)

contact = "Contact: your.email@institution.edu   •   QR: link to repo/preprint"
draw.text((margin, footer_y+28), contact, fill="black", font=ref_f)

# QR placeholder box
qr_size = 120
qr_x = W - margin - qr_size
qr_y = footer_y - 10
draw.rectangle([qr_x, qr_y, qr_x+qr_size, qr_y+qr_size], outline="black", width=2)
draw.text((qr_x+12, qr_y+40), "QR code", fill="black", font=small_f)

# Save outputs
# png_path = "/mnt/data/survival_poster.png"
# pdf_path = "/mnt/data/survival_poster.pdf"
img.save("n.png", format="PNG")
img.save("n.ppng", "PDF")

# png_path, pdf_path
