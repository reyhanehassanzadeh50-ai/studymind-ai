import hashlib
import html
import io
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

try:
    import fitz  # PyMuPDF, used only for extracting images
except ImportError:
    fitz = None

load_dotenv()
st.set_page_config(
    page_title="StudyMind AI · Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
ROOT = Path(__file__).resolve().parent
DB = ROOT / "studymind.db"
IMAGE_DIR = ROOT / "saved_figures"
IMAGE_DIR.mkdir(exist_ok=True)
MODEL = "gemini-3-flash-preview"  # change to another available model if your account needs it
LANGUAGES = {"Deutsch": "German", "English": "English", "فارسی": "Persian"}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:radial-gradient(ellipse at 70% 0%,#29214c 0%,#111321 48%,#0b1020 100%);color:#f2f3fc;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#171a32,#0e1324);border-right:1px solid #363553;}
.block-container {padding-top:2rem;max-width:1250px;}
.hero {padding:36px 36px 30px;border:1px solid #5e4d9b;border-radius:25px;background:linear-gradient(125deg,#29204c,#211c41 55%,#182a43);box-shadow:0 12px 45px #080a1890;animation:rise .65s ease-out;}
.hero h1 {font-size:clamp(2rem,4vw,3.3rem);margin:0;color:#fff;letter-spacing:-1.7px;}
.hero p {color:#c6c4e2;font-size:1.05rem;}
.kicker {color:#cbb7ff;letter-spacing:2px;font-size:12px;font-weight:800;}
.mini {border:1px solid #393b5e;border-radius:18px;padding:19px;background:#1a1d35;min-height:135px;animation:rise .75s ease-out;}
.mini strong {font-size:1.5rem;color:#f6f2ff;}
.mini p {color:#b7b9d1;margin-bottom:0;}
[data-testid="stMetric"] {border:1px solid #363b5d;background:#191d35;border-radius:15px;padding:14px;}
.stButton>button[kind="primary"], .stDownloadButton>button {background:linear-gradient(90deg,#7658ee,#a16be9);color:white;border:0;border-radius:12px;font-weight:700;}
.stButton>button {border-radius:12px;}
[data-testid="stTabs"] button {font-weight:700;}
@keyframes rise {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
</style>
""",
    unsafe_allow_html=True,
)


def connect():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS courses (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL)"
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY, course_id INTEGER NOT NULL, title TEXT NOT NULL,
        created_at TEXT NOT NULL, source_hash TEXT NOT NULL, source_text TEXT NOT NULL,
        payload TEXT NOT NULL, FOREIGN KEY(course_id) REFERENCES courses(id))""")
    conn.commit()
    return conn


def courses():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM courses ORDER BY name").fetchall()
        return [dict(row) for row in rows]


def add_course(name):
    name = name.strip()[:100]
    if not name:
        raise ValueError("Enter a course name.")
    with connect() as conn:
        conn.execute("INSERT OR IGNORE INTO courses(name) VALUES(?)", (name,))
        conn.commit()


def save_session(course_id, title, source_hash, source_text, payload):
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions(course_id,title,created_at,source_hash,source_text,payload) VALUES(?,?,?,?,?,?)",
            (
                course_id,
                title[:140],
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                source_hash,
                source_text,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cur.lastrowid


def history(course_id=None):
    with connect() as conn:
        sql = "SELECT s.*,c.name AS course FROM sessions s JOIN courses c ON c.id=s.course_id"
        if course_id is None:
            return conn.execute(sql + " ORDER BY s.id DESC").fetchall()
        return conn.execute(
            sql + " WHERE s.course_id=? ORDER BY s.id DESC", (course_id,)
        ).fetchall()


def get_session(session_id):
    with connect() as conn:
        return conn.execute(
            "SELECT s.*,c.name AS course FROM sessions s JOIN courses c ON c.id=s.course_id WHERE s.id=?",
            (session_id,),
        ).fetchone()


def delete_session(session_id):
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()


def extract_pdf(upload):
    raw = upload.getvalue()
    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")
    if len(reader.pages) > 80:
        raise ValueError("Please upload a PDF of at most 80 pages.")
    text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return text, len(reader.pages), raw


def extract_images(raw, limit=5):
    found = []
    if fitz is None:
        return found
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            for page in doc:
                for item in page.get_images(full=True):
                    image_data = doc.extract_image(item[0])
                    if len(image_data["image"]) > 20_000:
                        found.append(image_data["image"])
                    if len(found) >= limit:
                        return found
    except Exception:
        pass
    return found


def save_figures(raw, digest):
    folder = IMAGE_DIR / digest
    folder.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(extract_images(raw)):
        (folder / f"figure_{i}.bin").write_bytes(data)


def load_figures(digest):
    folder = IMAGE_DIR / digest
    if not folder.exists():
        return []
    return [p.read_bytes() for p in sorted(folder.glob("figure_*.bin"))]


def get_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        st.error(
            "GEMINI_API_KEY is missing. Add it to your .env file and restart the app."
        )
        return None
    return genai.Client(api_key=key)


def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    return json.loads(raw)


def generate_pack(text, lang, level, focus):
    client = get_client()
    if client is None:
        return None
    # Limit prompt size to control cost. Large files need chunking in a later version.
    excerpt = text[:42000]
    prompt = f"""You are an expert university tutor and instructional designer.
Use ONLY the source lecture notes below. Never invent unsupported facts. If information is missing, explicitly state that.
Write in {lang}. Level: {level}. Focus: {focus}.
Return ONLY a valid JSON object with EXACT keys:
"title": string,
"overview": string (3-5 sentences),
"sections": array of 4-8 objects each with "heading" (string), "explanation" (detailed string), "example" (string), "exam_tip" (string),
"key_terms": array of 5-12 objects with "term" and "definition" strings,
"flashcards": array of 8-12 objects with "question" and "answer" strings,
"quiz": array of 6-10 objects with "question" string, "options" array of EXACTLY 4 strings, "correct_index" integer 0-3, "explanation" string,
"mindmap": array of 4-8 objects with "topic" string and "children" array of 2-4 short strings,
"study_plan": array of 4 objects with "day" string and "task" string.
Include useful code examples in explanations where source supports them. Do not use markdown code fences inside JSON.
SOURCE NOTES:\n{excerpt}"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.35
                ),
            )
            result = parse_json(response.text or "")
            for key in (
                "title",
                "overview",
                "sections",
                "flashcards",
                "quiz",
                "mindmap",
                "key_terms",
                "study_plan",
            ):
                if key not in result:
                    raise ValueError(f"AI output missing {key}")
            return result
        except (errors.ServerError, errors.ClientError) as exc:
            if attempt == 2:
                st.error(f"Gemini request failed: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
        except (ValueError, json.JSONDecodeError, AttributeError) as exc:
            st.error(f"Could not parse AI response: {exc}. Please try again.")
            return None
    return None


def ask_tutor(question, source_text, lang):
    client = get_client()
    if client is None:
        return None
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Answer in {lang}. Use only the provided lecture notes. If not covered, say so clearly.\nNOTES:\n{source_text[:35000]}\nQUESTION: {question}",
    )
    return response.text or "No answer returned."


def pdf_font():
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    ]
    for normal, bold in candidates:
        if Path(normal).exists() and Path(bold).exists():
            if "StudyRegular" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("StudyRegular", normal))
                pdfmetrics.registerFont(TTFont("StudyBold", bold))
                pdfmetrics.registerFontFamily(
                    "StudyRegular", normal="StudyRegular", bold="StudyBold"
                )
            return "StudyRegular", "StudyBold"
    return "Helvetica", "Helvetica-Bold"


def safe(s):
    return html.escape(str(s or "")).replace("\n", "<br/>")


def make_pdf(pack, course, images=None):
    font, bold = pdf_font()
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    navy, violet = colors.HexColor("#17223B"), colors.HexColor("#6855C9")
    title = ParagraphStyle(
        "SMTitle",
        parent=styles["Title"],
        fontName=bold,
        fontSize=24,
        leading=31,
        textColor=navy,
        spaceAfter=18,
    )
    heading = ParagraphStyle(
        "SMHeading",
        fontName=bold,
        fontSize=14,
        leading=20,
        textColor=violet,
        spaceBefore=18,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "SMBody",
        fontName=font,
        fontSize=9.6,
        leading=16,
        textColor=navy,
        spaceAfter=9,
        wordWrap="CJK",
    )
    small = ParagraphStyle("SMSmall", parent=body, fontSize=8.3, leading=13)
    cover = ParagraphStyle(
        "SMCover",
        fontName=bold,
        fontSize=30,
        leading=38,
        textColor=navy,
        alignment=TA_CENTER,
    )

    def page_decor(canvas, doc):
        canvas.saveState()
        w, h = doc.pagesize
        canvas.setFillColor(violet)
        canvas.rect(0, h - 13, w, 13, stroke=0, fill=1)
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#777D96"))
        canvas.drawString(43, 30, "StudyMind AI  |  " + str(course)[:45])
        canvas.drawRightString(w - 43, 30, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        buf,
        pagesize=(595, 842),
        leftMargin=45,
        rightMargin=45,
        topMargin=53,
        bottomMargin=52,
    )
    frame = Frame(
        45, 52, 505, 735, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0
    )
    doc.addPageTemplates(PageTemplate(id="default", frames=[frame], onPage=page_decor))
    story = [
        Spacer(1, 75),
        Paragraph("STUDYMIND AI", cover),
        Spacer(1, 22),
        Paragraph(safe(pack.get("title", "Study notes")), title),
        Paragraph("Course: " + safe(course), body),
        Paragraph("AI-powered learning guide", body),
        Spacer(1, 28),
        Paragraph(safe(pack.get("overview", "")), body),
        PageBreak(),
    ]
    story.append(Paragraph("Deep-dive study guide", title))
    for i, section in enumerate(pack.get("sections", []), 1):
        story.append(Paragraph(f"{i}. " + safe(section.get("heading")), heading))
        story.append(Paragraph(safe(section.get("explanation")), body))
        if section.get("example"):
            story.append(Paragraph("<b>Example:</b> " + safe(section["example"]), body))
        if section.get("exam_tip"):
            box = Table(
                [
                    [
                        Paragraph(
                            "<b>EXAM TIP</b><br/>" + safe(section["exam_tip"]), small
                        )
                    ]
                ],
                colWidths=[495],
            )
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEAFE")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFC6FA")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(box)
    story.append(Paragraph("Key terms", title))
    for term in pack.get("key_terms", []):
        story.append(
            Paragraph(
                "<b>"
                + safe(term.get("term"))
                + "</b> — "
                + safe(term.get("definition")),
                body,
            )
        )
    story.append(Paragraph("Concept tree", title))
    for node in pack.get("mindmap", []):
        story.append(Paragraph("<b>" + safe(node.get("topic")) + "</b>", heading))
        for child in node.get("children", []):
            story.append(Paragraph("&nbsp;&nbsp;&nbsp;• " + safe(child), body))
    if images:
        story.append(Paragraph("Images from the uploaded lecture PDF", title))
        for raw in images[:3]:
            try:
                from PIL import Image as PILImage

                im = PILImage.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((470, 360))
                image_bytes = io.BytesIO()
                im.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                story.append(RLImage(image_bytes, width=im.width, height=im.height))
                story.append(Spacer(1, 12))
            except Exception:
                continue
    story.append(Paragraph("Flashcards", title))
    for i, card in enumerate(pack.get("flashcards", []), 1):
        story.append(
            Paragraph(
                f"<b>{i}. {safe(card.get('question'))}</b><br/>{safe(card.get('answer'))}",
                body,
            )
        )
    story.append(Paragraph("Practice quiz", title))
    for i, item in enumerate(pack.get("quiz", []), 1):
        story.append(Paragraph(f"<b>{i}. {safe(item.get('question'))}</b>", body))
        for letter, option in zip("ABCD", item.get("options", [])):
            story.append(Paragraph(letter + ". " + safe(option), small))
    story.append(Paragraph("Answer key", heading))
    for i, item in enumerate(pack.get("quiz", []), 1):
        idx = item.get("correct_index", 0)
        if not isinstance(idx, int) or idx not in range(4):
            idx = 0
        story.append(
            Paragraph(f"{i}. {chr(65 + idx)} — " + safe(item.get("explanation")), body)
        )
    story.append(Paragraph("Study plan", title))
    for item in pack.get("study_plan", []):
        story.append(
            Paragraph(
                "<b>" + safe(item.get("day")) + ":</b> " + safe(item.get("task")), body
            )
        )
    doc.build(story)
    return buf.getvalue()


def dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")[:100]


def mindmap_dot(pack):
    lines = [
        "digraph G {",
        'graph [bgcolor="transparent", rankdir="LR", pad="0.3", nodesep="0.3", ranksep="0.65"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", color="#6D5DF1", fillcolor="#27264C", fontcolor="white", margin="0.16"];',
        'edge [color="#8B7CEF", penwidth=1.5];',
        f'root [label="{dot_escape(pack.get("title", "Study notes"))}", fillcolor="#6D5DF1"];',
    ]
    for i, node in enumerate(pack.get("mindmap", [])):
        lines.append(
            f'n{i} [label="{dot_escape(node.get("topic", "Topic"))}"]; root -> n{i};'
        )
        for j, child in enumerate(node.get("children", [])):
            lines.append(
                f'n{i}_{j} [label="{dot_escape(child)}", fillcolor="#1A3450", color="#3C668D"]; n{i} -> n{i}_{j};'
            )
    return "\n".join(lines + ["}"])


def view_pack(pack, course, source_text, source_hash, images=None):
    if images is None:
        images = load_figures(source_hash)
    st.markdown("### 📖 " + str(pack.get("title", "Study guide")))
    st.info(pack.get("overview", ""))
    tabs = st.tabs(
        [
            "📘 Deep Dive",
            "🧠 Mind Map",
            "🃏 Flashcards",
            "🎯 Quiz",
            "💬 AI Tutor",
            "🗓️ Study Plan",
            "📥 Export",
        ]
    )
    with tabs[0]:
        for section in pack.get("sections", []):
            with st.expander("✨ " + section.get("heading", "Section"), expanded=True):
                st.write(section.get("explanation", ""))
                if section.get("example"):
                    st.markdown("**Example**")
                    st.code(section["example"], language=None)
                if section.get("exam_tip"):
                    st.warning("🎓 Exam tip: " + section["exam_tip"])
        st.markdown("#### 📚 Key terminology")
        for term in pack.get("key_terms", []):
            st.markdown(f"**{term.get('term','')}** — {term.get('definition','')}")
        if images:
            st.markdown("#### 🖼️ Figures from your uploaded notes")
            st.caption(
                "These are original images extracted from the PDF, not AI-generated illustrations."
            )
            for img in images:
                st.image(img, width=480)
        else:
            st.caption(
                "No embedded figures were found. The Mind Map tab provides a visual explanation."
            )
    with tabs[1]:
        st.markdown("#### Explore how the concepts connect")
        st.graphviz_chart(mindmap_dot(pack), use_container_width=True)
        st.caption(
            "AI-created concept tree based on your uploaded notes. Verify important details against the source."
        )
    with tabs[2]:
        cards = pack.get("flashcards", [])
        if cards:
            key = "flash_" + source_hash[:12]
            idx = st.session_state.get(key, 0) % len(cards)
            st.progress((idx + 1) / len(cards), text=f"Card {idx+1} of {len(cards)}")
            st.markdown("#### " + cards[idx].get("question", ""))
            if st.toggle(
                "Reveal answer", key="reveal_" + source_hash[:12] + "_" + str(idx)
            ):
                st.success(cards[idx].get("answer", ""))
            a, b = st.columns(2)
            if a.button("← Previous", key="prev_" + source_hash[:12]):
                st.session_state[key] = (idx - 1) % len(cards)
                st.rerun()
            if b.button("Next →", key="next_" + source_hash[:12]):
                st.session_state[key] = (idx + 1) % len(cards)
                st.rerun()
    with tabs[3]:
        quiz = pack.get("quiz", [])
        with st.form("quiz_form_" + source_hash[:12]):
            for i, q in enumerate(quiz):
                options = q.get("options", [])
                if len(options) != 4:
                    continue
                st.radio(
                    f"{i+1}. {q.get('question','')}",
                    ["Choose an answer"] + options,
                    key=f"q_{source_hash[:12]}_{i}",
                )
            submitted = st.form_submit_button("Check my answers", type="primary")
        if submitted:
            score, total = 0, 0
            for i, q in enumerate(quiz):
                options = q.get("options", [])
                if len(options) != 4:
                    continue
                total += 1
                chosen = st.session_state.get(f"q_{source_hash[:12]}_{i}")
                correct = q.get("correct_index", 0)
                if (
                    isinstance(correct, int)
                    and 0 <= correct < 4
                    and chosen == options[correct]
                ):
                    score += 1
                    st.success(f"Q{i+1}: Correct")
                else:
                    answer = (
                        options[correct]
                        if isinstance(correct, int) and 0 <= correct < 4
                        else "Unknown"
                    )
                    st.error(f"Q{i+1}: Correct answer: {answer}")
                st.caption(q.get("explanation", ""))
            st.metric("Your score", f"{score}/{total}")
    with tabs[4]:
        st.caption(
            "Ask questions about the uploaded lecture notes. The tutor is instructed not to invent facts beyond the document."
        )
        question = st.text_input(
            "What would you like to understand?", key="tutor_" + source_hash[:12]
        )
        if (
            st.button("Ask StudyMind", key="ask_" + source_hash[:12])
            and question.strip()
        ):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_tutor(
                        question,
                        source_text,
                        st.session_state.get("language", "German"),
                    )
                    if answer:
                        st.markdown(answer)
                except Exception as exc:
                    st.error(f"Tutor request failed: {exc}")
    with tabs[5]:
        for item in pack.get("study_plan", []):
            st.checkbox(
                f"{item.get('day','Day')}: {item.get('task','')}",
                key=f"plan_{source_hash[:12]}_{item.get('day','')}",
            )
    with tabs[6]:
        try:
            pdf = make_pdf(pack, course, images)
            st.download_button(
                "📥 Download professional PDF",
                data=pdf,
                file_name="StudyMind_Study_Guide.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")
        st.download_button(
            "📦 Export structured JSON",
            json.dumps(pack, ensure_ascii=False, indent=2),
            file_name="studymind_notes.json",
            mime="application/json",
        )


if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "generated" not in st.session_state:
    st.session_state.generated = None

with st.sidebar:
    st.markdown("## 🧠 StudyMind AI")
    st.caption("Your personal learning studio")
    page = st.radio(
        "Navigation",
        ["✨ Create study pack", "📚 My Library", "📊 Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Courses / Classes**")
    new_course = st.text_input("New course", placeholder="e.g. Programming I")
    if st.button("＋ Add course", use_container_width=True):
        try:
            add_course(new_course)
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))
    st.divider()
    st.caption("Local-first history · AI powered by Gemini")

if page == "✨ Create study pack":
    st.markdown(
        """<div class="hero"><div class="kicker">✦ YOUR PERSONAL AI LEARNING STUDIO</div><h1>Learn deeper. Remember longer.</h1><p>Turn lecture PDFs into beautiful study guides, interactive quizzes, concept trees, flashcards and more.</p></div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            '<div class="mini"><strong>📖 Deep explanations</strong><p>Clear concepts, examples and exam insights.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="mini"><strong>🧠 Visual learning</strong><p>Concept maps and original lecture figures.</p></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            '<div class="mini"><strong>🎯 Active recall</strong><p>Flashcards, quizzes and learning plans.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### 📂 Create your study workspace")
    available = courses()
    if not available:
        st.info("Create your first course in the sidebar to get started.")
    else:
        selected = st.selectbox(
            "Course / Class", available, format_func=lambda r: r["name"]
        )
        upload = st.file_uploader("Upload lecture PDF", type=["pdf"])
        left, right = st.columns(2)
        with left:
            lang_label = st.selectbox("Output language", list(LANGUAGES))
            st.session_state.language = LANGUAGES[lang_label]
        with right:
            level = st.selectbox(
                "Explanation depth",
                ["University / Detailed", "Beginner-friendly", "Exam-focused"],
            )
        focus = st.text_input(
            "Special focus (optional)",
            placeholder="e.g. Python loops and exam questions",
        )
        if upload:
            try:
                source, pages, raw = extract_pdf(upload)
                digest = hashlib.sha256(
                    raw + lang_label.encode() + level.encode() + focus.encode()
                ).hexdigest()
                st.success(f"PDF loaded: {pages} pages · {len(source):,} characters")
                with st.expander("Preview extracted text"):
                    st.text(source[:6000] or "No selectable text detected.")
                if not source.strip():
                    st.warning(
                        "This PDF appears to be scanned. OCR is not included in this version."
                    )
                elif st.button(
                    "✨ Generate my complete AI Study Pack",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "Creating detailed notes, flashcards, quiz, mind map and study plan..."
                    ):
                        result = generate_pack(
                            source, LANGUAGES[lang_label], level, focus
                        )
                    if result:
                        save_figures(raw, digest)
                        sid = save_session(
                            selected["id"],
                            result.get("title", upload.name),
                            digest,
                            source,
                            result,
                        )
                        st.session_state.active_id = sid
                        st.session_state.generated = None
                        st.rerun()
            except Exception as exc:
                st.error(f"Unable to read the PDF: {exc}")
    if st.session_state.active_id:
        record = get_session(st.session_state.active_id)
        if record:
            st.divider()
            view_pack(
                json.loads(record["payload"]),
                record["course"],
                record["source_text"],
                record["source_hash"],
            )

elif page == "📚 My Library":
    st.title("📚 My Study Library")
    available = courses()
    filters = ["All courses"] + [r["name"] for r in available]
    chosen = st.selectbox("Filter by course", filters)
    course_id = next((r["id"] for r in available if r["name"] == chosen), None)
    records = history(course_id)
    search = st.text_input(
        "🔎 Search saved notes", placeholder="Search by title or course"
    )
    records = [
        r for r in records if search.lower() in (r["title"] + " " + r["course"]).lower()
    ]
    if not records:
        st.info("No saved study packs yet. Generate one on the Create page.")
    for r in records:
        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 1, 1])
            col1.markdown(f"**{r['title']}**")
            col1.caption(f"📁 {r['course']} · 🕒 {r['created_at']}")
            if col2.button("Open", key="open_" + str(r["id"])):
                st.session_state.active_id = r["id"]
                st.rerun()
            if col3.button("Delete", key="del_" + str(r["id"])):
                delete_session(r["id"])
                if st.session_state.active_id == r["id"]:
                    st.session_state.active_id = None
                st.rerun()
    if st.session_state.active_id:
        record = get_session(st.session_state.active_id)
        if record:
            st.divider()
            view_pack(
                json.loads(record["payload"]),
                record["course"],
                record["source_text"],
                record["source_hash"],
            )

else:
    st.title("📊 Learning Dashboard")
    all_records = history()
    a, b, c = st.columns(3)
    a.metric("Courses", len(courses()))
    b.metric("Saved study packs", len(all_records))
    c.metric(
        "Flashcards created",
        sum(len(json.loads(r["payload"]).get("flashcards", [])) for r in all_records),
    )
    st.markdown("### Recent study sessions")
    for r in all_records[:10]:
        st.write(f"📘 **{r['title']}** · {r['course']} · {r['created_at']}")
    st.caption(
        "Progress tracking across devices and user accounts is not yet implemented. This dashboard shows local records only."
    )

st.divider()
st.caption(
    "StudyMind AI · Built with Python, Streamlit, SQLite, Gemini and ReportLab · Always verify AI-generated notes against the source."
)
