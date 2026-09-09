"""
SIH26012 PowerPoint Presentation Generator.
Generates an official 16:9 widescreen SIH Idea Submission PPT matching the exact structure,
layout, colors, typography, tables, and callout boxes of the sample template.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path="SIH26012_Idea_Presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_slide_layout = prs.slide_layouts[6] # completely blank layout

    # Palette
    C_NAVY        = RGBColor(15, 34, 64)       # Deep SIH Navy
    C_BLUE        = RGBColor(28, 90, 160)      # Primary Header Blue
    C_BANNER_BLUE = RGBColor(0, 112, 192)      # Official SIH Banner Blue
    C_LIGHTBLUE   = RGBColor(230, 242, 255)    # Light accent background
    C_TEAL        = RGBColor(13, 148, 136)     # Cadastre Teal
    C_ORANGE      = RGBColor(234, 88, 12)      # Accent Orange / Bulb
    C_GREEN       = RGBColor(22, 163, 74)      # Success Green
    C_YELLOW_BG   = RGBColor(254, 243, 199)    # Highlight Box Gold
    C_YELLOW_BD   = RGBColor(245, 158, 11)     # Highlight Border Gold
    C_DARK        = RGBColor(30, 41, 59)       # Body text
    C_MUTED       = RGBColor(100, 116, 139)    # Subtext
    C_WHITE       = RGBColor(255, 255, 255)
    C_BORDER      = RGBColor(203, 213, 225)
    C_CARD_BG     = RGBColor(248, 250, 252)

    def add_header(slide, title_text, team_name="SkyGen"):
        # Top Team Pill (Left)
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(0.28), Inches(1.8), Inches(0.55))
        pill.fill.solid()
        pill.fill.fore_color.rgb = C_WHITE
        pill.line.color.rgb = C_BORDER
        pill.line.width = Pt(1.5)
        tf_pill = pill.text_frame
        tf_pill.word_wrap = True
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = team_name
        p_pill.font.name = "Arial"
        p_pill.font.size = Pt(14)
        p_pill.font.bold = True
        p_pill.font.color.rgb = C_NAVY
        p_pill.alignment = PP_ALIGN.CENTER

        # Center Title
        tb_title = slide.shapes.add_textbox(Inches(2.5), Inches(0.2), Inches(8.333), Inches(0.7))
        tf_title = tb_title.text_frame
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Georgia"
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY
        p_title.alignment = PP_ALIGN.CENTER

        # Right SIH Badge
        tb_sih = slide.shapes.add_textbox(Inches(11.0), Inches(0.18), Inches(2.0), Inches(0.75))
        tf_sih = tb_sih.text_frame
        p_s1 = tf_sih.paragraphs[0]
        p_s1.text = "SMART INDIA"
        p_s1.font.name = "Arial Black"
        p_s1.font.size = Pt(10)
        p_s1.font.color.rgb = C_ORANGE
        p_s1.alignment = PP_ALIGN.RIGHT
        p_s2 = tf_sih.add_paragraph()
        p_s2.text = "HACKATHON 2026"
        p_s2.font.name = "Arial Black"
        p_s2.font.size = Pt(10)
        p_s2.font.color.rgb = C_NAVY
        p_s2.alignment = PP_ALIGN.RIGHT

    def add_footer(slide, slide_num=None):
        # Footer banner
        shape_ft = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.05), Inches(13.333), Inches(0.45))
        shape_ft.fill.solid()
        shape_ft.fill.fore_color.rgb = C_BANNER_BLUE
        shape_ft.line.fill.background()
        tf = shape_ft.text_frame
        p = tf.paragraphs[0]
        p.text = "@SIH Idea submission- Template"
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p.alignment = PP_ALIGN.CENTER

        if slide_num:
            tb_num = slide.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.45))
            p_n = tb_num.text_frame.paragraphs[0]
            p_n.text = str(slide_num)
            p_n.font.name = "Arial"
            p_n.font.size = Pt(11)
            p_n.font.bold = True
            p_n.font.color.rgb = C_WHITE
            p_n.alignment = PP_ALIGN.RIGHT

    # =========================================================================
    # SLIDE 1: TITLE PAGE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_slide_layout)

    # Top Header
    tb_h1 = s1.shapes.add_textbox(Inches(1.0), Inches(0.35), Inches(9.5), Inches(0.6))
    p_h1 = tb_h1.text_frame.paragraphs[0]
    p_h1.text = "SMART INDIA HACKATHON 2026"
    p_h1.font.name = "Georgia"
    p_h1.font.size = Pt(28)
    p_h1.font.bold = True
    p_h1.font.color.rgb = C_BLUE
    p_h1.alignment = PP_ALIGN.CENTER

    # Right Logo placeholder
    tb_sih1 = s1.shapes.add_textbox(Inches(11.0), Inches(0.2), Inches(2.0), Inches(0.9))
    p1 = tb_sih1.text_frame.paragraphs[0]
    p1.text = "SMART INDIA"
    p1.font.name = "Arial Black"
    p1.font.size = Pt(11)
    p1.font.color.rgb = C_ORANGE
    p1.alignment = PP_ALIGN.RIGHT
    p2 = tb_sih1.text_frame.add_paragraph()
    p2.text = "HACKATHON 2026"
    p2.font.name = "Arial Black"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_NAVY
    p2.alignment = PP_ALIGN.RIGHT

    # "TITLE PAGE" center subhead
    tb_sub = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(9.5), Inches(0.5))
    p_sub = tb_sub.text_frame.paragraphs[0]
    p_sub.text = "TITLE PAGE"
    p_sub.font.name = "Arial Black"
    p_sub.font.size = Pt(22)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_NAVY
    p_sub.alignment = PP_ALIGN.CENTER

    # Left content box
    tb_details = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(7.5), Inches(5.0))
    tf_det = tb_details.text_frame
    tf_det.word_wrap = True

    items_s1 = [
        ("Problem Statement ID –", " SIH26012"),
        ("Problem Statement Title –", " Automated Cadastral Feature Extraction & Urban Parcel Mapping System"),
        ("Theme –", " Geospatial Technology / Smart Cities / Land Governance"),
        ("PS Category –", " Software"),
        ("Team ID –", " 69110"),
        ("Team Name :-", " SkyGen"),
        ("College Name :-", " [Your College / Institute Name Here]"),
    ]

    for idx, (label, val) in enumerate(items_s1):
        p = tf_det.paragraphs[0] if idx == 0 else tf_det.add_paragraph()
        p.space_after = Pt(14)
        run1 = p.add_run()
        run1.text = "• " + label
        run1.font.name = "Arial"
        run1.font.size = Pt(18)
        run1.font.bold = True
        run1.font.color.rgb = C_NAVY

        run2 = p.add_run()
        run2.text = val
        run2.font.name = "Arial"
        run2.font.size = Pt(18)
        run2.font.color.rgb = C_DARK

    # Right side graphic card (Bulb / AI Cadastre graphic representation)
    card_right = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.9), Inches(3.8), Inches(4.8))
    card_right.fill.solid()
    card_right.fill.fore_color.rgb = C_CARD_BG
    card_right.line.color.rgb = C_BORDER
    card_right.line.width = Pt(1.5)
    tf_cr = card_right.text_frame
    tf_cr.word_wrap = True
    
    p_cr1 = tf_cr.paragraphs[0]
    p_cr1.text = "CadastreVision-AI"
    p_cr1.font.name = "Arial Black"
    p_cr1.font.size = Pt(18)
    p_cr1.font.color.rgb = C_BLUE
    p_cr1.alignment = PP_ALIGN.CENTER

    p_cr2 = tf_cr.add_paragraph()
    p_cr2.space_before = Pt(10)
    p_cr2.text = "Automated Land Boundary &\nUrban Parcel Extraction"
    p_cr2.font.name = "Arial"
    p_cr2.font.size = Pt(13)
    p_cr2.font.bold = True
    p_cr2.font.color.rgb = C_NAVY
    p_cr2.alignment = PP_ALIGN.CENTER

    features_s1 = [
        "🧠 12.38M Full CadastreUNet",
        "🌐 Sub-Meter (0.25m) Aerial Imagery",
        "🗺️ Continuous Solid Cyan Lines",
        "📐 Closed Property Parcel Polygons",
        "⚖️ ISO 19107 Topology Audit Engine",
        "🚀 94.43% Pixel Accuracy Benchmark"
    ]
    for feat in features_s1:
        pf = tf_cr.add_paragraph()
        pf.space_before = Pt(8)
        pf.text = feat
        pf.font.name = "Arial"
        pf.font.size = Pt(11)
        pf.font.color.rgb = C_DARK
        pf.alignment = PP_ALIGN.LEFT

    # =========================================================================
    # SLIDE 2: IDEA TITLE
    # =========================================================================
    s2 = prs.slides.add_slide(blank_slide_layout)
    add_header(s2, "IDEA TITLE")
    add_footer(s2)

    # Sub-heading banner
    tb_s2_sub = s2.shapes.add_textbox(Inches(0.6), Inches(0.9), Inches(12.133), Inches(0.5))
    p_s2_sub = tb_s2_sub.text_frame.paragraphs[0]
    p_s2_sub.text = "CadastreVision — Automated Cadastral Feature Extraction & Urban Parcel Mapping System"
    p_s2_sub.font.name = "Arial"
    p_s2_sub.font.size = Pt(14)
    p_s2_sub.font.bold = True
    p_s2_sub.font.color.rgb = C_NAVY

    # Left Column: Proposed Solution (6.5 inches wide)
    tb_s2_left = s2.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(6.8), Inches(5.5))
    tf_s2_l = tb_s2_left.text_frame
    tf_s2_l.word_wrap = True

    p_prop = tf_s2_l.paragraphs[0]
    p_prop.text = "Proposed Solution :-"
    p_prop.font.name = "Arial Black"
    p_prop.font.size = Pt(15)
    p_prop.font.color.rgb = C_BLUE
    p_prop.space_after = Pt(4)

    p_subhead = tf_s2_l.add_paragraph()
    p_subhead.text = "A Deep Learning Geospatial Platform powered by PMG + MixStyle + Directional Affinity for Survey-Grade Parcel Mapping & Title Verification"
    p_subhead.font.name = "Arial"
    p_subhead.font.size = Pt(10.5)
    p_subhead.font.bold = True
    p_subhead.font.color.rgb = C_DARK
    p_subhead.space_after = Pt(8)

    sol_points = [
        ("Sub-Meter Boundary Extraction Hub:-", " Automated inference from 0.25m GSD high-resolution aerial imagery via CadastreUNetFull (12.38M params) capturing micro-fences and long borders."),
        ("Continuous Solid Cadastral Borders:-", " 8-way directional connectivity head & 1.8m endpoint snapping eliminate dotted/dashed artifacts, producing 100% continuous solid borders."),
        ("Closed Cadastral Parcel Polygonizer:-", " Interior contour polygonization automatically extracts closed residential property plots, calculating exact land area (m²) and perimeter (m)."),
        ("ISO 19107 Real-Time Topology Validator:-", " Spatial STRtree index audits vector lines against OGC standards, guaranteeing 0 self-intersections and 0 invalid shapes."),
        ("Human-in-the-Loop Surveyor Deck:-", " Interactive WebGIS enables revenue surveyors to inspect, edit, and click '✓ Approve Feature' before committing legal property boundaries."),
        ("One-Click GIS Vector Export:-", " Direct survey-grade GeoJSON export fully compatible with QGIS, ArcGIS, and national land registries (SVAMITVA, Bhoomi, BRK).")
    ]

    for title, desc in sol_points:
        p = tf_s2_l.add_paragraph()
        p.space_after = Pt(5)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.name = "Arial"
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_DARK

    # Right Column: Architecture Diagram Card (5.4 inches wide)
    card_arch = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.5), Inches(1.4), Inches(5.3), Inches(5.5))
    card_arch.fill.solid()
    card_arch.fill.fore_color.rgb = C_CARD_BG
    card_arch.line.color.rgb = C_BORDER
    card_arch.line.width = Pt(1.5)
    tf_ca = card_arch.text_frame
    tf_ca.word_wrap = True

    p_at = tf_ca.paragraphs[0]
    p_at.text = "SYSTEM ARCHITECTURE & FLOW"
    p_at.font.name = "Arial Black"
    p_at.font.size = Pt(11)
    p_at.font.color.rgb = C_BLUE
    p_at.alignment = PP_ALIGN.CENTER

    arch_layers = [
        ("👥 TARGET USERS", "Revenue Surveyors, Town Planners, Land Registries (SVAMITVA), Property Owners", RGBColor(59, 130, 246)),
        ("🖥️ UI LAYER (WebGIS)", "Leaflet 1.9.4 Map, Tailwind CSS Controls, Real-time Overlays, SVG Renderers", RGBColor(13, 148, 136)),
        ("⚡ BACKEND SERVICES", "FastAPI REST API, /api/predict, /api/vectorize, /api/topology-audit, Layer Engine", RGBColor(234, 88, 12)),
        ("🧠 AI / ML ENGINE", "CadastreUNetFull: PMG Multiscale (d=1,2,4) + MixStyle DG + 8-Way Connectivity Head", RGBColor(139, 92, 246)),
        ("📐 COMPUTATIONAL GIS", "Guo-Hall Thinning, NetworkX Graph Tracing, Douglas-Peucker (ε=0.35m), STRtree Audit", RGBColor(16, 185, 129)),
        ("💾 OUTPUT / STORAGE", "Survey-Grade GeoJSON, ESRI Shapefile, Closed Parcel Plots (m²), GeoPackage Catalog", RGBColor(15, 34, 64)),
    ]

    for title, desc, col in arch_layers:
        p_l = tf_ca.add_paragraph()
        p_l.space_before = Pt(6)
        r1 = p_l.add_run()
        r1.text = title + "\n"
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = col
        r2 = p_l.add_run()
        r2.text = desc
        r2.font.name = "Arial"
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_DARK

    # =========================================================================
    # SLIDE 3: TECHNICAL APPROACH
    # =========================================================================
    s3 = prs.slides.add_slide(blank_slide_layout)
    add_header(s3, "TECHNICAL APPROACH")
    add_footer(s3, slide_num=3)

    # Left Column: Technologies & Process Flow
    tb_s3_left = s3.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(7.2), Inches(5.8))
    tf_s3_l = tb_s3_left.text_frame
    tf_s3_l.word_wrap = True

    p_t1 = tf_s3_l.paragraphs[0]
    p_t1.text = "Technologies to be Used:-"
    p_t1.font.name = "Arial Black"
    p_t1.font.size = Pt(14)
    p_t1.font.color.rgb = C_BLUE
    p_t1.space_after = Pt(4)

    tech_specs = [
        ("Frontend:-", " Leaflet.js 1.9.4, Tailwind CSS, HTML5 Canvas, Lucide Icons"),
        ("Backend:-", " FastAPI, Uvicorn (ASGI Server), Pydantic v2, Python 3.10+"),
        ("GIS & Geometry:-", " GeoPandas, Shapely 2.0+, Rasterio, NetworkX, OpenCV, PyProj"),
        ("AI / Deep Learning:-", " PyTorch 2.0+, Torchvision, AdamW, Cosine Annealing"),
        ("CRS & Projections:-", " EPSG:28992 (Dutch Metric RD New) ↔ EPSG:4326 (WGS84)"),
        ("Cloud & Infra:-", " NVIDIA A100 GPU, Apple Silicon Metal (MPS), Docker, Google Colab Pro"),
        ("Integrations:-", " QGIS, ArcGIS, SVAMITVA / Bhoomi APIs, ESRI Satellite Basemap")
    ]

    for lbl, val in tech_specs:
        p = tf_s3_l.add_paragraph()
        p.space_after = Pt(2)
        r1 = p.add_run()
        r1.text = "• " + lbl
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        r2 = p.add_run()
        r2.text = val
        r2.font.name = "Arial"
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_DARK

    p_pf = tf_s3_l.add_paragraph()
    p_pf.space_before = Pt(8)
    p_pf.text = "Process Flow :-"
    p_pf.font.name = "Arial Black"
    p_pf.font.size = Pt(13)
    p_pf.font.color.rgb = C_BLUE

    # Chevron flow sequence representation
    flow_steps = [
        "1. Aerial Tile Acquisition (0.25m GSD)",
        "2. Windowed 512×512 Patch Extraction",
        "3. PMG Multiscale Dilated Encoding (d=1,2,4)",
        "4. MixStyle Domain Generalization (Beta 0.2)",
        "5. Dual Heads (Boundary + 8-Way Affinity)",
        "6. Medial-Axis Thinning & Graph Tracing",
        "7. Closed Parcel Polygonization (Area/Perim)",
        "8. ISO 19107 STRtree Topology Audit",
        "9. Interactive WebGIS & One-Click GeoJSON"
    ]
    for step in flow_steps:
        p = tf_s3_l.add_paragraph()
        p.text = "  → " + step
        p.font.name = "Arial"
        p.font.size = Pt(8.5)
        p.font.color.rgb = C_DARK

    # Project Links Box
    tb_links = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(5.75), Inches(7.2), Inches(1.15))
    tb_links.fill.solid()
    tb_links.fill.fore_color.rgb = C_LIGHTBLUE
    tb_links.line.color.rgb = C_BLUE
    tb_links.line.width = Pt(1.5)
    tf_lk = tb_links.text_frame
    tf_lk.word_wrap = True
    p_lk_t = tf_lk.paragraphs[0]
    p_lk_t.text = "• Project Links Demo:-"
    p_lk_t.font.name = "Arial Black"
    p_lk_t.font.size = Pt(11)
    p_lk_t.font.color.rgb = C_NAVY
    
    p_lk1 = tf_lk.add_paragraph()
    p_lk1.text = "• Github: https://github.com/SkyGen/CadastreVision"
    p_lk1.font.name = "Arial"
    p_lk1.font.size = Pt(10)
    p_lk1.font.bold = True
    p_lk1.font.color.rgb = C_BLUE

    p_lk2 = tf_lk.add_paragraph()
    p_lk2.text = "• Demo Live Prototype : https://cadastre.skygen.site"
    p_lk2.font.name = "Arial"
    p_lk2.font.size = Pt(10)
    p_lk2.font.bold = True
    p_lk2.font.color.rgb = C_TEAL

    # Right Column: Technology Architecture Badges Box (5.0 inches wide)
    card_tech = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.1), Inches(4.9), Inches(5.8))
    card_tech.fill.solid()
    card_tech.fill.fore_color.rgb = C_CARD_BG
    card_tech.line.color.rgb = C_BORDER
    card_tech.line.width = Pt(1.5)
    tf_tc = card_tech.text_frame
    tf_tc.word_wrap = True

    p_tc_h = tf_tc.paragraphs[0]
    p_tc_h.text = "CORE MODULE STACK"
    p_tc_h.font.name = "Arial Black"
    p_tc_h.font.size = Pt(12)
    p_tc_h.font.color.rgb = C_BLUE
    p_tc_h.alignment = PP_ALIGN.CENTER

    tech_badges = [
        ("FRONTEND", "Leaflet 1.9.4 | Tailwind CSS | HTML5 Canvas\nESRI World Imagery | Lucide GIS Icons", RGBColor(14, 165, 233)),
        ("BACKEND", "FastAPI (Python 3.10+) | Uvicorn ASGI\nPydantic v2 | In-Memory Tile Caching", RGBColor(16, 185, 129)),
        ("AI / ML MODELS", "CadastreUNetFull (12.38M Params)\nPMG Dilated Module (d=1,2,4) | MixStyle DG\n8-Way Connectivity Head | Balanced Focal+Dice", RGBColor(168, 85, 247)),
        ("COMPUTATIONAL GIS", "Shapely 2.0 STRtree | GeoPandas 0.13+\nNetworkX 3.0 Graph Tracing | Rasterio 1.3+\nGuo-Hall Thinning | Douglas-Peucker (ε=0.35m)", RGBColor(234, 88, 12)),
        ("CLOUD & INFRA", "NVIDIA A100 SXM4 (40GB VRAM)\nApple Silicon MPS (Metal) | Docker Container\nGoogle Colab Pro Standalone Bundle", RGBColor(59, 130, 246)),
        ("DATASET & BENCHMARKS", "CadastreVision-SIH26012 (900 km² Coverage)\n9 GeoTIFF Tiles (0.25m GSD, EPSG:28992)\n3,700 Patches (2500 Train, 600 Val, 600 Test)", RGBColor(15, 34, 64)),
    ]

    for title, desc, col in tech_badges:
        p = tf_tc.add_paragraph()
        p.space_before = Pt(6)
        r1 = p.add_run()
        r1.text = "▪ " + title + "\n"
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = col
        r2 = p.add_run()
        r2.text = desc
        r2.font.name = "Arial"
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_DARK

    # =========================================================================
    # SLIDE 4: FEASIBILITY AND VIABILITY
    # =========================================================================
    s4 = prs.slides.add_slide(blank_slide_layout)
    add_header(s4, "FEASIBILITY AND VIABILITY")
    add_footer(s4, slide_num=4)

    # Left Column: Feasibility, Challenges, Viability (6.2 inches wide)
    tb_s4_left = s4.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(6.3), Inches(5.8))
    tf_s4_l = tb_s4_left.text_frame
    tf_s4_l.word_wrap = True

    sections_s4_l = [
        ("Feasibility :-", [
            ("Technical:", " Uses proven PyTorch/FastAPI/Shapely stack → high scalability"),
            ("Economic:", " Zero proprietary software licensing → >80% survey cost reduction"),
            ("Operational:", " Lightweight browser UI, zero-install, field laptop compatible"),
            ("Social:", " Formalizes rural property titles, eliminates land litigation")
        ]),
        ("Potential Challenges :-", [
            ("", "Class imbalance with boundary lines < 2.1% of patch pixels"),
            ("", "Invisible/sub-surface property boundaries without physical markers"),
            ("", "Cross-geographic terrain variations (differing soil, roof textures)"),
            ("", "Large multi-gigabyte aerial GeoTIFF I/O bottleneck")
        ]),
        ("Mitigation Strategies :-", [
            ("", "Balanced Multi-Objective Loss (Focal α=0.75 + Dice + Conn w_pos=15.0)"),
            ("", "Human-in-the-loop review deck & supporting building/road layers"),
            ("", "MixStyle Domain Generalization perturbing feature stats (Beta 0.2)"),
            ("", "Windowed lazy raster reads via Rasterio & R-tree spatial indexing")
        ]),
        ("Viability & Business Potential :-", [
            ("", "Directly addresses India's SVAMITVA mission (6.6 lakh villages)"),
            ("", "SaaS / On-Premise government contracts with State Revenue Depts"),
            ("", "Municipal property tax assessment & urban corridor acquisition")
        ])
    ]

    is_first = True
    for sec_title, bullet_list in sections_s4_l:
        p_sec = tf_s4_l.paragraphs[0] if is_first else tf_s4_l.add_paragraph()
        is_first = False
        p_sec.space_before = Pt(4)
        p_sec.text = sec_title
        p_sec.font.name = "Arial Black"
        p_sec.font.size = Pt(11)
        p_sec.font.color.rgb = C_BLUE

        for lbl, val in bullet_list:
            p_b = tf_s4_l.add_paragraph()
            p_b.space_after = Pt(1)
            if lbl:
                r1 = p_b.add_run()
                r1.text = "• " + lbl
                r1.font.name = "Arial"
                r1.font.size = Pt(8.5)
                r1.font.bold = True
                r1.font.color.rgb = C_NAVY
            else:
                r1 = p_b.add_run()
                r1.text = "• "
                r1.font.name = "Arial"
                r1.font.size = Pt(8.5)
                r1.font.bold = True
                r1.font.color.rgb = C_NAVY
            r2 = p_b.add_run()
            r2.text = " " + val
            r2.font.name = "Arial"
            r2.font.size = Pt(8.0)
            r2.font.color.rgb = C_DARK

    # Right Column: Use Cases & Solutions (6.0 inches wide)
    tb_s4_right = s4.shapes.add_textbox(Inches(7.0), Inches(1.1), Inches(5.8), Inches(3.2))
    tf_s4_r = tb_s4_right.text_frame
    tf_s4_r.word_wrap = True

    sections_s4_r = [
        ("Use Cases :-", [
            "SVAMITVA Mission: Automated parcel extraction from drone orthoimagery",
            "Municipal Town Planning: Updating property tax cadastre & identifying vacant plots",
            "Agricultural Land Delineation: Farmland plot demarcation & crop subsidy claims",
            "Linear Infrastructure: Land acquisition corridor mapping for NHAI & Railways"
        ]),
        ("Key Solutions Delivered :-", [
            "Continuous Solid Cyan Lines: No dotted lines; 1.8m endpoint snapping",
            "Closed Land Parcels: Metric area (m²) & perimeter (m) calculated automatically",
            "Zero OGC Violations: Real-time ISO 19107 STRtree topology auditing",
            "One-Click CAD/GIS Export: Ready for direct import into QGIS & ArcGIS"
        ])
    ]

    is_first_r = True
    for sec_title, bullet_list in sections_s4_r:
        p_sec = tf_s4_r.paragraphs[0] if is_first_r else tf_s4_r.add_paragraph()
        is_first_r = False
        p_sec.space_before = Pt(4)
        p_sec.text = sec_title
        p_sec.font.name = "Arial Black"
        p_sec.font.size = Pt(11)
        p_sec.font.color.rgb = C_BLUE

        for val in bullet_list:
            p_b = tf_s4_r.add_paragraph()
            p_b.space_after = Pt(1)
            p_b.text = "• " + val
            p_b.font.name = "Arial"
            p_b.font.size = Pt(8.5)
            p_b.font.color.rgb = C_DARK

    # Gold Highlight Box: SUPPORTING FACTS FOR FEASIBILITY AND VIABILITY
    gold_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(4.5), Inches(5.8), Inches(2.4))
    gold_box.fill.solid()
    gold_box.fill.fore_color.rgb = C_YELLOW_BG
    gold_box.line.color.rgb = C_YELLOW_BD
    gold_box.line.width = Pt(1.5)
    tf_gb = gold_box.text_frame
    tf_gb.word_wrap = True

    p_gb_t = tf_gb.paragraphs[0]
    p_gb_t.text = "SUPPORTING FACTS FOR FEASIBILITY AND VIABILITY"
    p_gb_t.font.name = "Arial Black"
    p_gb_t.font.size = Pt(10.5)
    p_gb_t.font.color.rgb = RGBColor(180, 83, 9) # Amber 700
    p_gb_t.alignment = PP_ALIGN.CENTER

    facts = [
        "Over 65% of all civil court litigation in India is tied to property/boundary disputes.",
        "SVAMITVA drone mapping covers 6.6 lakh villages; automated AI cuts drafting time by >80%.",
        "Measured production accuracy on 600 held-out test patches: 94.43% pixel accuracy.",
        "Topological validation engine achieved 100% VALID geometries across all benchmark tiles.",
        "Inference latency: 42 ms on NVIDIA A100 GPU / 110 ms on Apple Silicon MPS."
    ]

    for fact in facts:
        pf = tf_gb.add_paragraph()
        pf.space_before = Pt(3)
        pf.text = "• " + fact
        pf.font.name = "Arial"
        pf.font.size = Pt(8.5)
        pf.font.color.rgb = RGBColor(120, 53, 15) # Amber 900

    # =========================================================================
    # SLIDE 5: IMPACT AND BENEFITS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_slide_layout)
    add_header(s5, "IMPACT AND BENEFITS")
    add_footer(s5, slide_num=5)

    # Top Left: Potential Impact
    tb_s5_tl = s5.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(6.3), Inches(2.3))
    tf_s5_tl = tb_s5_tl.text_frame
    tf_s5_tl.word_wrap = True

    p_pi = tf_s5_tl.paragraphs[0]
    p_pi.text = "Potential impact on the target audience:-"
    p_pi.font.name = "Arial Black"
    p_pi.font.size = Pt(12)
    p_pi.font.color.rgb = C_BLUE

    impact_points = [
        ("Land Revenue Departments:", " Eliminates manual digitization backlogs; maps villages in hours instead of months."),
        ("Certified Revenue Surveyors:", " AI-powered drafting co-pilot; focuses surveyor effort on verification rather than drawing."),
        ("Rural & Urban Citizens:", " Fast, dispute-free Property Cards (Aadhar for Land) enabling institutional bank loans."),
        ("Municipalities & Planners:", " Up-to-date cadastral base maps for property tax assessment and infrastructure corridors.")
    ]
    for lbl, val in impact_points:
        p = tf_s5_tl.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + lbl
        r1.font.name = "Arial"
        r1.font.size = Pt(9.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        r2 = p.add_run()
        r2.text = " " + val
        r2.font.name = "Arial"
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_DARK

    # Top Right: Unique Outcomes
    tb_s5_tr = s5.shapes.add_textbox(Inches(7.0), Inches(1.1), Inches(5.8), Inches(2.3))
    tf_s5_tr = tb_s5_tr.text_frame
    tf_s5_tr.word_wrap = True

    p_uo = tf_s5_tr.paragraphs[0]
    p_uo.text = "Unique Outcomes from Our Solution"
    p_uo.font.name = "Arial Black"
    p_uo.font.size = Pt(12)
    p_uo.font.color.rgb = C_BLUE

    outcomes = [
        ("Massive Speedup:", " Turnaround time reduced from weeks to < 150 ms per patch with 1-click execution."),
        ("Survey Cost Reduction:", " Lowers parcel mapping costs from ₹1,000+ to less than ₹5 per parcel."),
        ("Survey-Grade Topology:", " Zero self-intersections and zero duplicate lines (ISO 19107 compliant)."),
        ("Closed Parcel Geometry:", " Automated property plot polygonization with metric area (m²) and perimeter (m)."),
        ("Cross-Region Generalization:", " MixStyle feature statistics perturbation prevents regional overfitting.")
    ]
    for lbl, val in outcomes:
        p = tf_s5_tr.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + lbl
        r1.font.name = "Arial"
        r1.font.size = Pt(9.0)
        r1.font.bold = True
        r1.font.color.rgb = C_NAVY
        r2 = p.add_run()
        r2.text = " " + val
        r2.font.name = "Arial"
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_DARK

    # Bottom Half: Benefits Table
    tb_s5_bot = s5.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(12.333), Inches(0.4))
    p_tbl_t = tb_s5_bot.text_frame.paragraphs[0]
    p_tbl_t.text = "Benefits of the Solution (Social, Economic, Environmental)"
    p_tbl_t.font.name = "Arial Black"
    p_tbl_t.font.size = Pt(12)
    p_tbl_t.font.color.rgb = C_BLUE

    # Add 4-row, 3-column table
    rows, cols = 4, 3
    left, top, width, height = Inches(0.5), Inches(3.95), Inches(12.333), Inches(2.9)
    table_s5 = s5.shapes.add_table(rows, cols, left, top, width, height).table
    table_s5.columns[0].width = Inches(1.8)
    table_s5.columns[1].width = Inches(5.2)
    table_s5.columns[2].width = Inches(5.333)

    # Headers
    headers = ["Type", "Benefit", "Supporting Example / Measurable Impact"]
    for j, h in enumerate(headers):
        cell = table_s5.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial Black"
        p.font.size = Pt(11)
        p.font.color.rgb = C_WHITE
        p.alignment = PP_ALIGN.LEFT

    table_data = [
        ("Social", 
         "Mitigates property disputes & establishes secure land rights\nEmpowers rural landowners with formal credit collateral",
         "Land disputes constitute >65% of all civil litigation in India; clear boundaries resolve disputes before entering courts."),
        ("Economic",
         "Massive survey cost reduction & accelerated land administration\nEnables municipal corporations to broaden property tax bases",
         "Reduces manual digitizing costs from ₹500–₹2,000 to <₹5 per parcel; maps 900 km² dataset in under 1 hour of compute."),
        ("Environmental",
         "Accurate boundary delineation prevents encroachment into forest reserves\nOptimizes sustainable land use and infrastructure corridor planning",
         "Digitized cadastral boundaries establish clear legal buffer zones around floodplains, wetlands, and conservation reserves.")
    ]

    for i, row in enumerate(table_data, start=1):
        for j, val in enumerate(row):
            cell = table_s5.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_CARD_BG if i % 2 == 1 else C_WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(9.0)
            if j == 0:
                p.font.bold = True
                p.font.color.rgb = C_BLUE
            else:
                p.font.color.rgb = C_DARK

    # =========================================================================
    # SLIDE 6: RESEARCH AND REFERENCES
    # =========================================================================
    s6 = prs.slides.add_slide(blank_slide_layout)
    add_header(s6, "RESEARCH AND REFERENCES")
    add_footer(s6, slide_num=6)

    # Left Column: Research Papers & Platforms
    tb_s6_l = s6.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(6.0), Inches(4.5))
    tf_s6_l = tb_s6_l.text_frame
    tf_s6_l.word_wrap = True

    p_rp = tf_s6_l.paragraphs[0]
    p_rp.text = "• Research Papers:"
    p_rp.font.name = "Arial Black"
    p_rp.font.size = Pt(11.5)
    p_rp.font.color.rgb = C_BLUE

    papers = [
        "a. Enemark, S., et al., 'Fit-for-Purpose Land Administration: Guiding Principles for Country Implementation,' UN-Habitat / GLTN (2016).",
        "b. Crommelinck, S., Bennett, R., et al., 'Large-scale Cadastral Mapping using High-Resolution UAV Imagery: A Review,' ISPRS IJGI (2016).",
        "c. Zhou, K., Yang, Y., et al., 'Domain Generalization with MixStyle,' International Conference on Learning Representations (ICLR 2021).",
        "d. Zhou, L., Zhang, C., & Wu, M., 'D-LinkNet: LinkNet with Dilated Convolution for Satellite Imagery Road Extraction,' CVPRW (2018)."
    ]
    for p_txt in papers:
        p = tf_s6_l.add_paragraph()
        p.space_after = Pt(2)
        p.text = p_txt
        p.font.name = "Arial"
        p.font.size = Pt(7.8)
        p.font.color.rgb = C_DARK

    p_plat = tf_s6_l.add_paragraph()
    p_plat.space_before = Pt(4)
    p_plat.text = "• Cadastral & Remote Sensing Benchmarks:"
    p_plat.font.name = "Arial Black"
    p_plat.font.size = Pt(11)
    p_plat.font.color.rgb = C_BLUE

    platforms = [
        "a. Dutch National Land Registry (Kadaster BRK / PDOK): https://www.pdok.nl",
        "b. Survey of India (SVAMITVA Drone Scheme): https://svamitva.nic.in",
        "c. OpenStreetMap Cadastral Guidelines: https://wiki.openstreetmap.org"
    ]
    for plat in platforms:
        p = tf_s6_l.add_paragraph()
        p.space_after = Pt(1)
        p.text = plat
        p.font.name = "Arial"
        p.font.size = Pt(8.0)
        p.font.color.rgb = C_DARK

    # Left Bottom: Project Links Demo
    card_links_s6 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.35), Inches(6.0), Inches(1.3))
    card_links_s6.fill.solid()
    card_links_s6.fill.fore_color.rgb = C_LIGHTBLUE
    card_links_s6.line.color.rgb = C_BLUE
    card_links_s6.line.width = Pt(1.5)
    tf_cls = card_links_s6.text_frame
    tf_cls.word_wrap = True

    p_cl_t = tf_cls.paragraphs[0]
    p_cl_t.text = "• Project Links Demo:-"
    p_cl_t.font.name = "Arial Black"
    p_cl_t.font.size = Pt(11)
    p_cl_t.font.color.rgb = C_NAVY

    p_cl1 = tf_cls.add_paragraph()
    p_cl1.text = "• Github: https://github.com/SkyGen/CadastreVision"
    p_cl1.font.name = "Arial"
    p_cl1.font.size = Pt(9.5)
    p_cl1.font.bold = True
    p_cl1.font.color.rgb = C_BLUE

    p_cl2 = tf_cls.add_paragraph()
    p_cl2.text = "• Demo Live Prototype : https://cadastre.skygen.site"
    p_cl2.font.name = "Arial"
    p_cl2.font.size = Pt(9.5)
    p_cl2.font.bold = True
    p_cl2.font.color.rgb = C_TEAL

    # Right Column: Feature Comparison Matrix Table (6.3 inches wide)
    rows_s6, cols_s6 = 9, 5
    table_s6 = s6.shapes.add_table(rows_s6, cols_s6, Inches(6.7), Inches(1.1), Inches(6.1), Inches(4.55)).table
    table_s6.columns[0].width = Inches(2.3)
    table_s6.columns[1].width = Inches(0.95)
    table_s6.columns[2].width = Inches(0.95)
    table_s6.columns[3].width = Inches(0.95)
    table_s6.columns[4].width = Inches(0.95)

    comp_headers = ["Feature / Capability", "Our Platform", "Manual DGPS", "Generic AI", "ArcGIS Pro"]
    for j, h in enumerate(comp_headers):
        cell = table_s6.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial Black"
        p.font.size = Pt(8.0)
        p.font.color.rgb = C_WHITE
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    matrix_rows = [
        ("1. Sub-Meter Boundary Line Extraction", "✔", "✔", "✖", "⚠"),
        ("2. Continuous Solid Lines (No Dots)", "✔", "✔", "✖", "✖"),
        ("3. Closed Parcel Extraction (Area m²)", "✔", "✔", "✖", "⚠"),
        ("4. Automated ISO 19107 Topology Audit", "✔", "⚠", "✖", "⚠"),
        ("5. MixStyle Domain Generalization", "✔", "✖", "✖", "✖"),
        ("6. 8-Way Directional Connectivity Head", "✔", "✖", "✖", "✖"),
        ("7. Zero-Install WebGIS Interface", "✔", "✖", "✖", "✖"),
        ("8. Survey Turnaround per Patch", "<150ms", "Weeks", "Seconds*", "Minutes*")
    ]

    for i, row in enumerate(matrix_rows, start=1):
        for j, val in enumerate(row):
            cell = table_s6.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_CARD_BG if i % 2 == 1 else C_WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(8.0)
            if j == 0:
                p.font.bold = True
                p.font.color.rgb = C_DARK
                p.alignment = PP_ALIGN.LEFT
            elif j == 1:
                p.font.bold = True
                p.font.color.rgb = C_GREEN if val == "✔" or val == "<150ms" else C_BLUE
                p.alignment = PP_ALIGN.CENTER
            else:
                p.font.color.rgb = C_GREEN if val == "✔" else (RGBColor(220, 38, 38) if val == "✖" else C_MUTED)
                p.alignment = PP_ALIGN.CENTER

    # Bottom Full Width: Research Flow (Chevron Pipeline)
    tb_rf_title = s6.shapes.add_textbox(Inches(0.5), Inches(5.8), Inches(12.333), Inches(0.3))
    p_rft = tb_rf_title.text_frame.paragraphs[0]
    p_rft.text = "• Research Flow:"
    p_rft.font.name = "Arial Black"
    p_rft.font.size = Pt(10.5)
    p_rft.font.color.rgb = C_BLUE

    # 8 horizontal pipeline boxes
    r_flow_steps = [
        "Problem\nIdentification",
        "Literature &\nRemote Sensing",
        "Gap\nAnalysis",
        "Model & Loss\nExploration",
        "Validation &\nFeasibility",
        "Proposed\nSolution Design",
        "Final Research\nOutcome",
        "CadastreVision\nEcosystem"
    ]

    box_w = Inches(1.4)
    box_gap = Inches(0.12)
    box_top = Inches(6.15)
    box_h = Inches(0.7)

    for k, step_text in enumerate(r_flow_steps):
        b_x = Inches(0.5) + k * (box_w + box_gap)
        s_box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b_x, box_top, box_w, box_h)
        s_box.fill.solid()
        s_box.fill.fore_color.rgb = C_BLUE if k < 7 else C_TEAL
        s_box.line.color.rgb = C_WHITE
        s_box.line.width = Pt(1.0)
        tf_sb = s_box.text_frame
        tf_sb.word_wrap = True
        p_sb = tf_sb.paragraphs[0]
        p_sb.text = step_text
        p_sb.font.name = "Arial"
        p_sb.font.size = Pt(7.5)
        p_sb.font.bold = True
        p_sb.font.color.rgb = C_WHITE
        p_sb.alignment = PP_ALIGN.CENTER

    # Save presentation
    prs.save(output_path)
    print(f"[SUCCESS] Presentation generated successfully at: {output_path}")

if __name__ == "__main__":
    build_presentation()
