"""交付物导出：把生成的报告（Markdown）+ 指标，导出为 JSON / Markdown / PDF / PPTX。

- PDF 用 reportlab 内置 STSong-Light CID 字体，原生支持中文，无需额外字体文件；
- PPTX 用 python-pptx，按 `## ` 小节拆分幻灯片；
- 所有函数返回 (bytes, media_type, file_ext)，便于 Web 端点直接作为下载流。
"""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(s: str) -> str:
    """转 **加粗** 为 <b>，并转义 HTML 特殊字符。"""
    s = _escape(s)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


def export_report(report_md: str, metrics: Dict, fmt: str) -> Tuple[bytes, str, str]:
    fmt = (fmt or "md").lower()
    if fmt == "json":
        data = json.dumps({"report": report_md, "metrics": metrics}, ensure_ascii=False, indent=2)
        return data.encode("utf-8"), "application/json", "json"
    if fmt == "md":
        return report_md.encode("utf-8"), "text/markdown", "md"
    if fmt == "pdf":
        return _to_pdf(report_md, metrics), "application/pdf", "pdf"
    if fmt == "pptx":
        return _to_pptx(report_md, metrics), "application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"
    raise ValueError(f"不支持的导出格式: {fmt}")


# ---------- PDF ----------
def _to_pdf(report_md: str, metrics: Dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT = "STSong-Light"

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=FONT, fontSize=18, leading=24)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=FONT, fontSize=14, leading=20, spaceBefore=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=FONT, fontSize=12, leading=17, spaceBefore=8)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=FONT, fontSize=10, leading=15)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceBefore=2)

    import io

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="研究报告")
    flow = []
    for line in report_md.splitlines():
        line = line.rstrip()
        if not line.strip():
            flow.append(Spacer(1, 4))
            continue
        if line.startswith("## "):
            flow.append(Paragraph(_inline_md(line[3:]), h2))
        elif line.startswith("# "):
            flow.append(Paragraph(_inline_md(line[2:]), title))
        elif re.match(r"^[-*] ", line):
            flow.append(Paragraph("• " + _inline_md(line[2:]), bullet))
        else:
            flow.append(Paragraph(_inline_md(line), body))

    # 指标页
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("生成指标", h2))
    for k, v in (metrics or {}).items():
        if k in ("trace", "cost"):
            continue
        flow.append(Paragraph(f"• {k}: {v}", bullet))

    doc.build(flow)
    return buf.getvalue()


# ---------- PPTX ----------
def _to_pptx(report_md: str, metrics: Dict) -> bytes:
    from pptx import Presentation
    from pptx.util import Pt

    prs = Presentation()
    lines = report_md.splitlines()

    # 标题取首个 # 标题
    title_text = "研究报告"
    for ln in lines:
        if ln.startswith("# "):
            title_text = ln[2:].strip()
            break
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title_text
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = "多智能体协作生成 · 带引用深度研究报告"

    # 按 ## 拆分内容幻灯片
    sections = []
    cur_title, cur_body = None, []
    for ln in lines:
        if ln.startswith("## "):
            if cur_title is not None:
                sections.append((cur_title, cur_body))
            cur_title = ln[3:].strip()
            cur_body = []
        elif ln.strip():
            cur_body.append(ln.strip())
    if cur_title is not None:
        sections.append((cur_title, cur_body))

    if not sections:
        sections = [("报告内容", [ln.strip() for ln in lines if ln.strip()])]

    for sec_title, sec_body in sections:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = sec_title
        tf = s.placeholders[1].text_frame
        tf.word_wrap = True
        first = True
        for b in sec_body:
            b = re.sub(r"^[-*] ", "", b)
            b = re.sub(r"\*\*", "", b)
            b = re.sub(r"^#+\s*", "", b)
            if not b:
                continue
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            p.text = b
            p.font.size = Pt(16)
            first = False

    # 指标幻灯片
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "生成指标"
    tf = s.placeholders[1].text_frame
    tf.word_wrap = True
    first = True
    for k, v in (metrics or {}).items():
        if k in ("trace", "cost"):
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(16)
        first = False

    import io
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
