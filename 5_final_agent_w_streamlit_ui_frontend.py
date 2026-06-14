from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import zipfile
import importlib
import uuid
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
backend = importlib.import_module("5_final_agent_w_streamlit_ui_backend")
app = backend.app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(Path("images") / p.relative_to(images_dir)))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(Path("images") / p.relative_to(images_dir)))
    return buf.getvalue()


def try_stream(
    graph_app,
    inputs: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    final_config: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress and then read final checkpointed state.
    Yields ("updates"/"values"/"final", payload).
    """
    for step in graph_app.stream(inputs, config=config, stream_mode="updates"):
        yield ("updates", step)
    out = graph_app.get_state(final_config or config).values
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


def progress_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    plan_obj = state.get("plan")
    if hasattr(plan_obj, "tasks"):
        task_count = len(plan_obj.tasks or [])
    elif isinstance(plan_obj, dict):
        task_count = len(plan_obj.get("tasks", []) or [])
    else:
        task_count = None

    return {
        "mode": state.get("mode"),
        "needs_research": state.get("needs_research"),
        "queries": state.get("queries", [])[:5] if isinstance(state.get("queries"), list) else [],
        "evidence_count": len(state.get("evidence", []) or []),
        "tasks": task_count,
        "images": len(state.get("image_specs", []) or []),
        "sections_done": len(state.get("sections", []) or []),
    }


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    p = Path(src)
    if p.is_absolute():
        return p
    return (Path("output") / p).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Saved session helpers
# -----------------------------
def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def list_saved_thread_ids() -> List[str]:
    checkpoint_db = Path(getattr(backend, "checkpoint_db", Path("output/langgraph_checkpoints_webapp.sqlite")))
    if not checkpoint_db.exists():
        return []

    with sqlite3.connect(str(checkpoint_db)) as conn:
        rows = conn.execute(
            """
            SELECT thread_id, MAX(rowid) AS last_rowid
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY last_rowid DESC
            """
        ).fetchall()
    return [row[0] for row in rows]


def load_saved_session(thread_id: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    return dict(state.values or {})


def list_checkpoint_namespaces(thread_id: str) -> List[str]:
    checkpoint_db = Path(getattr(backend, "checkpoint_db", Path("output/langgraph_checkpoints_webapp.sqlite")))
    if not checkpoint_db.exists():
        return [""]

    with sqlite3.connect(str(checkpoint_db)) as conn:
        rows = conn.execute(
            """
            SELECT checkpoint_ns, MAX(rowid) AS last_rowid
            FROM checkpoints
            WHERE thread_id = ?
            GROUP BY checkpoint_ns
            ORDER BY last_rowid DESC
            """,
            (thread_id,),
        ).fetchall()
    namespaces = [row[0] for row in rows]
    return namespaces or [""]


def _history_config(thread_id: str, checkpoint_ns: str = "") -> Dict[str, Any]:
    configurable = {"thread_id": thread_id}
    if checkpoint_ns:
        configurable["checkpoint_ns"] = checkpoint_ns
    return {"configurable": configurable}


def iter_saved_snapshots(thread_id: str):
    for checkpoint_ns in list_checkpoint_namespaces(thread_id):
        for snapshot in app.get_state_history(_history_config(thread_id, checkpoint_ns)):
            yield checkpoint_ns, snapshot


def list_resume_nodes(thread_id: str) -> List[str]:
    nodes: List[str] = []
    seen = set()
    for _, snapshot in iter_saved_snapshots(thread_id):
        for node in snapshot.next or ():
            if node == "__start__" or node in seen:
                continue
            seen.add(node)
            nodes.append(node)
    return nodes


def latest_snapshot_for_node(thread_id: str, node_name: str):
    for _, snapshot in iter_saved_snapshots(thread_id):
        if node_name in (snapshot.next or ()):
            return snapshot
    return None


def load_latest_session_state(thread_id: str) -> Dict[str, Any]:
    for checkpoint_ns in list_checkpoint_namespaces(thread_id):
        values = dict(app.get_state(_history_config(thread_id, checkpoint_ns)).values or {})
        if values:
            return values
    return {}


def title_from_state(state: Dict[str, Any], fallback: str) -> str:
    plan_obj = state.get("plan")
    if hasattr(plan_obj, "blog_title") and plan_obj.blog_title:
        return str(plan_obj.blog_title)
    if isinstance(plan_obj, dict) and plan_obj.get("blog_title"):
        return str(plan_obj["blog_title"])

    final_md = state.get("final") or ""
    if final_md:
        return extract_title_from_md(final_md, fallback)

    topic = state.get("topic")
    return str(topic or fallback)


def saved_session_log_lines(thread_id: str) -> List[str]:
    lines: List[str] = []
    for i, (checkpoint_ns, snapshot) in enumerate(iter_saved_snapshots(thread_id), start=1):
        values = dict(snapshot.values or {})
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        next_nodes = ", ".join(snapshot.next or []) or "END"
        namespace_label = checkpoint_ns or "main"
        lines.append(
            f"[checkpoint {i}] namespace={namespace_label} checkpoint_id={checkpoint_id} next={next_nodes} "
            f"state_keys={list(values.keys())}"
        )
    return lines


def delete_saved_session(thread_id: str) -> List[str]:
    deleted: List[str] = []
    checkpoint_db = Path(getattr(backend, "checkpoint_db", Path("output/langgraph_checkpoints_webapp.sqlite")))
    if checkpoint_db.exists():
        with sqlite3.connect(str(checkpoint_db)) as conn:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.commit()
        deleted.append(str(checkpoint_db))

    thread_slug = safe_slug(thread_id)
    for p in Path("output").glob(f"5_{thread_slug}_*.md"):
        p.unlink(missing_ok=True)
        deleted.append(str(p))

    images_dir = Path("output/images") / thread_slug
    if images_dir.exists():
        shutil.rmtree(images_dir)
        deleted.append(str(images_dir))

    return deleted


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")


@st.dialog("Delete past session")
def confirm_delete_session(thread_id: str):
    st.warning(f"This will permanently delete session `{thread_id}`.")
    st.caption("This removes its checkpoint state, markdown file, and session image folder.")
    col_cancel, col_delete = st.columns(2)
    if col_cancel.button("Cancel"):
        st.rerun()
    if col_delete.button("Delete permanently"):
        deleted = delete_saved_session(thread_id)
        if st.session_state.get("thread_id") == thread_id:
            st.session_state["thread_id"] = f"blog-{uuid.uuid4().hex[:8]}"
        if st.session_state.get("loaded_past_thread_id") == thread_id:
            st.session_state.pop("loaded_past_thread_id", None)
            st.session_state.pop("selected_past_thread_id", None)
            st.session_state["last_out"] = None
            st.session_state["logs"] = []
        st.session_state["delete_message"] = f"Deleted `{thread_id}` ({len(deleted)} item(s))."
        st.rerun()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"blog-{uuid.uuid4().hex[:8]}"
if "graph_running" not in st.session_state:
    st.session_state["graph_running"] = False
if "resume_request" not in st.session_state:
    st.session_state["resume_request"] = None


def new_thread_id():
    st.session_state["thread_id"] = f"blog-{uuid.uuid4().hex[:8]}"
    st.session_state.pop("selected_past_thread_id", None)
    st.session_state.pop("loaded_past_thread_id", None)
    st.session_state["resume_request"] = None
    st.session_state["graph_running"] = False
    st.session_state["last_out"] = None
    st.session_state["logs"] = []


with st.sidebar:
    st.header("Generate New Blog")
    if st.session_state.get("delete_message"):
        st.success(st.session_state.pop("delete_message"))
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    st.caption(f"Session ID: `{st.session_state['thread_id']}`")
    st.button("New session", on_click=new_thread_id)
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past sessions")

    saved_thread_ids = list_saved_thread_ids()
    if not saved_thread_ids:
        st.caption("No saved sessions found in checkpoint state.")
    else:
        options: List[str] = ["New session"]
        session_by_label: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for thread_id in saved_thread_ids[:50]:
            try:
                saved_state = load_latest_session_state(thread_id)
                title = title_from_state(saved_state, thread_id)
            except Exception:
                saved_state = {}
                title = thread_id
            label = f"{title}  -  {thread_id}"
            options.append(label)
            session_by_label[label] = (thread_id, saved_state)

        selected_label = st.radio(
            "Select a session to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_session = session_by_label.get(selected_label)

        col_load, col_delete = st.columns(2)
        with col_load:
            load_clicked = st.button(
                "Load",
                disabled=selected_session is None or st.session_state["graph_running"],
            )
        with col_delete:
            delete_clicked = st.button(
                "Delete",
                disabled=selected_session is None or st.session_state["graph_running"],
            )

        if selected_session:
            selected_thread_id, saved_state = selected_session
            if load_clicked:
                st.session_state["thread_id"] = selected_thread_id
                st.session_state["last_out"] = saved_state
                st.session_state["logs"] = saved_session_log_lines(selected_thread_id)
                st.session_state["loaded_past_thread_id"] = selected_thread_id
                st.session_state["selected_past_thread_id"] = selected_thread_id
            if delete_clicked:
                confirm_delete_session(selected_thread_id)

    

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

selected_past_thread_id = st.session_state.get("loaded_past_thread_id")
if selected_past_thread_id:
    with st.container():
        st.caption(f"Selected past session: `{selected_past_thread_id}`")
        resume_nodes = list_resume_nodes(selected_past_thread_id)
        if resume_nodes:
            col_node, col_button = st.columns([3, 1])
            with col_node:
                resume_node = st.selectbox("Rerun from node", options=resume_nodes)
            with col_button:
                st.write("")
                st.write("")
                rerun_disabled = st.session_state["graph_running"] or st.session_state.get("resume_request") is not None
                rerun_button = st.empty()
                if rerun_button.button("Rerun selected session", disabled=rerun_disabled):
                    st.session_state["graph_running"] = True
                    st.session_state["resume_request"] = (selected_past_thread_id, resume_node)
                    rerun_button.button("Rerun selected session", disabled=True, key="rerun_selected_session_disabled")
                    st.info(f"Rerun started from `{resume_node}`. Progress will appear below.")
        else:
            st.caption("No rerunnable checkpoints found for this session.")

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


def run_graph_stream(
    inputs: Optional[Dict[str, Any]],
    graph_config: Dict[str, Any],
    final_config: Dict[str, Any],
    label: str,
):
    st.session_state["graph_running"] = True
    status = st.status(label, expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs, graph_config, final_config=final_config):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"➡️ Node: `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)
                try:
                    checkpoint_state = dict(app.get_state(final_config).values or {})
                    if checkpoint_state:
                        current_state.update(checkpoint_state)
                except Exception:
                    pass

                progress_area.json(progress_summary(current_state))

                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                out = payload
                st.session_state["last_out"] = out
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")
    finally:
        st.session_state["graph_running"] = False


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()
    st.session_state.pop("selected_past_thread_id", None)
    st.session_state.pop("loaded_past_thread_id", None)
    st.session_state["resume_request"] = None

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "thread_id": st.session_state["thread_id"],
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    graph_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    run_graph_stream(inputs, graph_config, graph_config, "Running graph…")

resume_request = st.session_state.get("resume_request")
if resume_request:
    resume_thread_id, resume_node = resume_request
    try:
        resume_snapshot = latest_snapshot_for_node(resume_thread_id, resume_node)
        if resume_snapshot is None:
            st.error(f"No checkpoint found for node `{resume_node}` in session `{resume_thread_id}`.")
        else:
            st.session_state["thread_id"] = resume_thread_id
            final_config = {"configurable": {"thread_id": resume_thread_id}}
            checkpoint_id = resume_snapshot.config.get("configurable", {}).get("checkpoint_id")
            log(f"[resume] node={resume_node} checkpoint_id={checkpoint_id}")
            run_graph_stream(
                None,
                resume_snapshot.config,
                final_config,
                f"Rerunning from `{resume_node}`…",
            )
            st.session_state["last_out"] = load_latest_session_state(resume_thread_id)
            st.session_state["logs"] = saved_session_log_lines(resume_thread_id)
    finally:
        st.session_state["resume_request"] = None
        st.session_state["graph_running"] = False
        st.rerun()

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("output/images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        thread_slug = safe_slug(str(out.get("thread_id") or st.session_state.get("thread_id", "")))
        session_images_dir = Path("output/images") / thread_slug
        images_dir = session_images_dir if session_images_dir.exists() else Path("output/images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")
