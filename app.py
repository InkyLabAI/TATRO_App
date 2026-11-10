"""Dash interface for local transcription, semantic video search, and summaries."""
import base64
import hashlib
import os
import secrets
import time
from functools import wraps
from progress import task, snapshot, fail
from pathlib import Path

import dash
from dash import Input, Output, State, clientside_callback, dcc, html
import flask

from artifacts import Artifacts, atomic_write, read_json, write_json
from languages import LANGUAGES, by_name
from local_llm import LLM_MODELS
from retrieval import EMBEDDING_MODELS, timestamp
from video_picker import choose_video
from transcript_data import raw_transcript_path
from workflow import document_preview, index_video, query_video, saved_pdf, summary_video, transcribe, translate_video


SUMMARY_MODE = "Transcription + translation + summary"
TRANSLATION_MODES = {"Transcription + translation", SUMMARY_MODE}
LANGUAGE_NAMES = [item["name"] for item in LANGUAGES.values()]
app = dash.Dash(__name__, title="TATRO - Local Video Assistant")
server = app.server
server.secret_key = os.environ.get("TATRO_SESSION_SECRET") or secrets.token_hex(32)


@server.before_request
def identify_browser():
    if "progress_id" not in flask.session:
        flask.session["progress_id"] = secrets.token_hex(24)


TASK_TITLES = {
    "btn-run": "Transcription", "btn-translate": "Translation", "btn-index": "Embeddings",
    "btn-generate-summary": "Summary", "btn-show-summary": "Open summary",
    "btn-chat-send": "Video search", "chat-input": "Video search", "video-selection": "Load video",
    "btn-browse-video": "Choose video", "video-path": "Choose video", "upload-json": "Import transcript",
    "upload-existing-pdf": "Import PDF", "upload-summary-pdf": "Import summary PDF",
    "btn-show-translation": "Open translation",
}


def track_progress(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        title = TASK_TITLES.get(dash.ctx.triggered_id)
        if title is None:
            return function(*args, **kwargs)
        with task(flask.session["progress_id"], title):
            return function(*args, **kwargs)
    return wrapped


SERVED_FILES = {}


def file_url(path):
    path = Path(path).resolve()
    # Refresh an iframe after a PDF is regenerated under the same filename.
    version = path.stat().st_mtime_ns if path.exists() else 0
    for token, known in SERVED_FILES.items():
        if known == path:
            return f"/files/{token}?v={version}"
    token = secrets.token_urlsafe(24)
    SERVED_FILES[token] = path
    return f"/files/{token}?v={version}"


@server.route("/files/<token>")
def serve_file(token):
    path = SERVED_FILES.get(token)
    if path is None or not path.is_file():
        flask.abort(404)
    return flask.send_file(path, conditional=True)


def save_upload(contents, filename, directory):
    if not contents or not filename:
        raise ValueError("Select a file first.")
    # Browsers expose a filename, never the original source directory.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if name in {"", ".", ".."}:
        raise ValueError("Invalid filename.")
    path = Path(directory).resolve() / name
    data = base64.b64decode(contents.split(",", 1)[1], validate=True)
    if path.exists() and path.suffix.lower() == ".mp4":
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
            raise ValueError("A different video has this filename. Rename the upload to preserve its saved results.")
        return path
    atomic_write(path, data)
    return path


def pdf_preview(path=None):
    if not path or not Path(path).is_file():
        return html.Div([html.Span("PDF", className="document-icon"), html.P("Your document will appear here."), html.Small("Generate a result or open a saved PDF.")], className="document-empty")
    url = file_url(path)
    return html.Div([html.Iframe(src=url, title="Document preview", className="pdf-frame"),
                     html.A("Open or download PDF", href=url, target="_blank")])


def button(label, identifier):
    return html.Button(label, id=identifier, className="button")


def dropdown(identifier, options, value):
    return dcc.Dropdown(id=identifier, options=options, value=value, clearable=False)


def field(label, control):
    return html.Div([html.Label(label, htmlFor=control.id), control], className="field")


app.layout = html.Div([
    dcc.Interval(id="progress-poll", interval=1000, n_intervals=0),
    dcc.Store(id="context-store"), dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="seek-timestamp-store"), dcc.Store(id="video-selection"),
    html.Div(id="seek-result", hidden=True),
    html.Header([
        html.Div([html.Div("T", className="brand-mark"), html.Div([
            html.H1("TATRO"), html.P("Your video, understood.")])], className="brand"),
        html.Span([html.Span(className="status-dot"), "Local AI processing"], className="local-badge"),
    ], className="app-header"),
    html.Section([
        html.Div([html.Span("01", className="step-label"), html.H2("Start with a video")], className="section-heading"),
        html.Label("Video file path", htmlFor="video-path", className="sr-only"),
        html.Div([
            dcc.Input(id="video-path", type="text", value="",
                      placeholder="Choose an MP4, or paste its full path and press Enter"),
            button("Browse video", "btn-browse-video"),
        ], className="file-row"),
        html.Div(id="picker-status", className="status-message", role="status"),
        html.P("Results stay beside your video, in a folder with the same name. Saved results are reused when you open it.", className="helper"),
    ], className="card video-source"),
    html.Main([
        html.Aside([
            html.Div([html.Span("02", className="step-label"), html.H2("Make it yours")], className="section-heading"),
            field("Video language", dropdown("source-dd", LANGUAGE_NAMES, "English")),
            field("Translate into", dropdown("target-dd", LANGUAGE_NAMES, "Arabic")),
            field("Your chat & summary language", dropdown("user-language", LANGUAGE_NAMES, "English")),
            html.Hr(),
            html.Details([
                html.Summary("Models & access"),
                field("Embedding model · 1,024 dimensions", dropdown("embedding-model", EMBEDDING_MODELS, EMBEDDING_MODELS[0])),
                field("Correction, translation & summary model", dropdown("llm-model", LLM_MODELS, LLM_MODELS[0])),
                field("Custom model (optional)", dcc.Input(id="custom-llm", placeholder="Model ID or local folder")),
                field("Hugging Face token", dcc.Input(id="hf-token-box", type="password", placeholder="Token for speaker detection")),
                html.P("Models run on your computer. Weights download on first use.", className="helper"),
            ], className="advanced-settings"),
            html.Hr(),
            dcc.Checklist(id="correction-check", options=[{"label": " Correct transcription", "value": "correct"}], value=["correct"], className="check-option"),
            dcc.RadioItems(id="mode-radio", options=["Transcription only", "Transcription + translation", SUMMARY_MODE], value="Transcription only", className="mode-options"),
            button("Transcribe & prepare search", "btn-run"),
            html.P("Processing again replaces the existing results.", className="helper"),
            dcc.Loading(html.Div(id="status-box", className="status-message", role="status"), color="#087f72"),
            html.Details([
                html.Summary("Manage saved results"),
                button("Rebuild search index", "btn-index"),
                dcc.Upload(id="upload-json", accept=".json", children=button("Import transcript JSON", "json-label")),
            ], className="advanced-settings"),
        ], className="card settings-panel"),
        html.Div([
            html.Section([
                html.Div([html.Div([html.Span("03", className="step-label"), html.H2("Watch & explore")], className="section-heading"),
                          html.Span("VIDEO", className="eyebrow")], className="card-topline"),
                html.Video(id="video-player", controls=True, preload="metadata", className="video-player"),
                html.Div([
                    html.Div([html.H3("Ask your video"), html.P("Find the moment that answers your question. We'll jump to it.", className="helper")]),
                    html.Div(id="chat-display", className="chat-history", role="log", **{"aria-live": "polite"}),
                    html.Div([
                        html.Label("Question about the video", htmlFor="chat-input", className="sr-only"),
                        dcc.Input(id="chat-input", placeholder="What would you like to know?"),
                        button("Send →", "btn-chat-send"),
                    ], className="chat-composer"),
                    dcc.Loading(html.Div(id="chat-status", className="status-message", role="status"), color="#087f72"),
                ], className="chat-panel"),
            ], className="card watch-card"),
        ], className="watch-column"),
    ], className="workspace-grid"),
    html.Section([
        html.Div([html.H2("Task progress"), html.Span("LIVE · updates every second", className="eyebrow")], className="card-topline"),
        html.Div(id="progress-jobs", children=html.P("Ready. Choose a task to see its progress here.", className="helper"), **{"aria-live": "polite"}),
        html.Details([html.Summary("Activity log · latest first"),
                      html.Pre(id="progress-log", children="No tasks started yet.", className="progress-log")], open=True),
    ], className="card progress-panel"),
    html.Div([html.H2("Your documents"), html.P("Translate the conversation or keep the essential ideas.", className="helper")], className="documents-heading"),
    html.Div([
        html.Section([
            html.Div([html.Span("TRANSCRIPT & TRANSLATION", className="eyebrow"), html.H3("Your transcript"),
                      html.P("Read your corrected transcript, or its translation when requested.", className="helper")]),
            html.Div([button("Generate translation", "btn-translate"), button("Open saved", "btn-show-translation"),
                      dcc.Upload(id="upload-existing-pdf", accept=".pdf", children=button("Import PDF", "pdf-label"))], className="document-actions"),
            html.Div(id="pdf-preview-container", children=pdf_preview()),
        ], className="card document-card"),
        html.Section([
            html.Div([html.Span("SUMMARY", className="eyebrow"), html.H3("The essential ideas"),
                      html.P("The key ideas, ready to read and download.", className="helper")]),
            html.Div([button("Generate summary", "btn-generate-summary"), button("Open saved", "btn-show-summary"),
                      dcc.Upload(id="upload-summary-pdf", accept=".pdf", children=button("Import PDF", "summary-label"))], className="document-actions"),
            html.P("Generate replaces the saved document. Open saved reuses it when available.", className="helper"),
            dcc.Loading(html.Div(id="summary-status-box", className="status-message", role="status"), color="#087f72"),
            html.Div(id="summary-preview-container", children=pdf_preview()),
        ], className="card document-card"),
    ], className="documents-grid"),
    html.Footer("Transcribe. Translate. Understand.", className="app-footer"),
], className="app-shell")


def selected_video(context):
    if not context or not context.get("video_path"):
        raise ValueError("Load a video first.")
    return Path(context["video_path"])


@app.callback(Output("video-path", "value"), Output("video-selection", "data"), Output("picker-status", "children"),
              Input("btn-browse-video", "n_clicks"), Input("video-path", "n_submit"),
              State("video-path", "value"), prevent_initial_call=True,
              running=[(Output("btn-browse-video", "disabled"), True, False)])
@track_progress
def browse_video(clicks, submit, typed_path):
    try:
        if dash.ctx.triggered_id == "btn-browse-video":
            path = choose_video(typed_path)
            if path is None:
                return dash.no_update, dash.no_update, "Selection cancelled."
        else:
            path = Path((typed_path or "").strip().strip('"')).expanduser()
        path = path.resolve()
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise ValueError("Select an existing MP4 video. In WSL, pasted paths must use /mnt/c/… format.")
        return str(path), {"path": str(path), "nonce": secrets.token_hex(8)}, f"Results folder: {Artifacts(path).directory}"
    except (OSError, ValueError, RuntimeError) as error:
        fail(error)
        return dash.no_update, dash.no_update, str(error)


@app.callback(
    Output("context-store", "data"), Output("video-player", "src"), Output("status-box", "children"),
    Input("video-selection", "data"),
    Input("btn-run", "n_clicks"), Input("btn-translate", "n_clicks"), Input("btn-index", "n_clicks"),
    Input("upload-json", "contents"),
    State("context-store", "data"), State("source-dd", "value"), State("target-dd", "value"),
    State("embedding-model", "value"), State("llm-model", "value"), State("custom-llm", "value"),
    State("hf-token-box", "value"), State("correction-check", "value"), State("mode-radio", "value"),
    prevent_initial_call=True,
    running=[(Output("btn-run", "disabled"), True, False),
             (Output("btn-translate", "disabled"), True, False), (Output("btn-index", "disabled"), True, False)],
)
@track_progress
def process_action(selection, run, translate, rebuild, uploaded_json,
                   context, source, target, embedding_model, llm_model, custom_llm,
                   hf_token, correction, mode):
    action = dash.ctx.triggered_id
    llm_model = (custom_llm or "").strip() or llm_model
    try:
        if action == "video-selection":
            video = Path(selection["path"]).resolve()
            if not video.is_file() or video.suffix.lower() != ".mp4":
                raise ValueError("Select an existing MP4 file.")
            context = {"video_path": str(video), "revision": secrets.token_hex(8)}
            artifacts = Artifacts(video)
            saved = [label for label, path in [("transcript", artifacts.transcript), ("merged chunks", artifacts.merged),
                                               ("embeddings", artifacts.embeddings)] if path.exists()]
            return context, file_url(video), "Loaded video. Saved results: " + (", ".join(saved) or "none yet")
        video = selected_video(context)
        artifacts = Artifacts(video)
        if action == "btn-run":
            token = (hf_token or os.environ.get("HF_TOKEN", "")).strip()
            if not token and raw_transcript_path(artifacts) is None:
                raise ValueError("Provide a Hugging Face token for speaker diarization.")
            config = {"whisper_model": "large-v3", "diarization_model": "pyannote/speaker-diarization-3.1",
                      "qwen_model": llm_model, "embedding_model": embedding_model, "max_chunk_duration": 30,
                      "use_qwen_correction": bool(correction)}
            context = {**context, "processing_mode": mode, "document_kind": "translation" if mode in TRANSLATION_MODES else "transcript"}
            transcribe(video, by_name(source), token, config)
            if mode in TRANSLATION_MODES:
                translate_video(video, by_name(target), llm_model, force=True)
            if mode == SUMMARY_MODE:
                summary_video(video, by_name(target), llm_model, force=True)
            message = "Processing complete. Results saved in the video’s results folder."
        elif action == "btn-translate":
            translate_video(video, by_name(target), llm_model, force=True)
            context = {**context, "processing_mode": mode, "document_kind": "translation"}
            message = "Document PDF saved (translation is skipped when the languages match)."
        elif action == "btn-index":
            index_video(video, embedding_model, force=True)
            message = "Embeddings rebuilt with the selected model."
        elif action == "upload-json":
            import json
            rows = json.loads(base64.b64decode(uploaded_json.split(",", 1)[1]).decode("utf-8-sig"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("Expected a non-empty transcript dataset JSON list.")
            required = {"id", "source_text", "source_language", "speaker_number", "start", "end"}
            if any(not required.issubset(row) for row in rows):
                raise ValueError("Import the transcript_dataset.json file, including text, language, speaker and times.")
            write_json(artifacts.transcript, rows)
            message = "Transcript loaded. Search will reuse or rebuild its index as needed."
        else:
            return dash.no_update, dash.no_update, dash.no_update
        context = {**context, "revision": secrets.token_hex(8)}
        return context, file_url(video), message
    except Exception as error:
        fail(error)
        return dash.no_update, dash.no_update, f"Could not complete this action: {error}"


@app.callback(Output("pdf-preview-container", "children"),
              Input("context-store", "data"), Input("target-dd", "value"), Input("mode-radio", "value"),
              Input("btn-show-translation", "n_clicks"), Input("upload-existing-pdf", "contents"),
              State("upload-existing-pdf", "filename"))
@track_progress
def show_translation(context, target, mode, clicks, contents, filename):
    try:
        if dash.ctx.triggered_id == "upload-existing-pdf" and contents:
            return pdf_preview(save_upload(contents, filename, Artifacts(selected_video(context)).directory))
        if context:
            requested = mode in TRANSLATION_MODES
            if context.get("processing_mode") == mode:
                requested = context.get("document_kind") == "translation"
            return pdf_preview(document_preview(selected_video(context), by_name(target), requested))
        return pdf_preview()
    except Exception as error:
        fail(error)
        return html.P(f"Preview unavailable: {error}")


@app.callback(Output("summary-status-box", "children"), Output("summary-preview-container", "children"),
              Input("btn-generate-summary", "n_clicks"), Input("btn-show-summary", "n_clicks"),
              Input("context-store", "data"), Input("user-language", "value"),
              Input("target-dd", "value"), Input("mode-radio", "value"),
              Input("upload-summary-pdf", "contents"), State("upload-summary-pdf", "filename"),
              State("llm-model", "value"), State("custom-llm", "value"),
              prevent_initial_call=True,
              running=[(Output("btn-generate-summary", "disabled"), True, False)])
@track_progress
def handle_summary(generate, show, context, language, target, mode, contents, filename, llm, custom):
    if mode == SUMMARY_MODE:
        language = target
    try:
        action = dash.ctx.triggered_id
        if action == "upload-summary-pdf":
            return "Loaded PDF.", pdf_preview(save_upload(contents, filename, Artifacts(selected_video(context)).directory))
        if not context:
            fail("Load a video first.")
            return "Load a video first.", pdf_preview()
        video = selected_video(context)
        if action in {"context-store", "user-language", "target-dd", "mode-radio"}:
            return "", pdf_preview(saved_pdf(video, by_name(language), "summary"))
        pdf = summary_video(video, by_name(language), (custom or "").strip() or llm,
                            force=action == "btn-generate-summary")
        return f"Summary saved: {pdf.name}", pdf_preview(pdf)
    except Exception as error:
        fail(error)
        return f"Summary unavailable: {error}", pdf_preview()


@app.callback(Output("chat-display", "children"), Output("chat-history-store", "data"),
              Output("chat-input", "value"), Output("seek-timestamp-store", "data"), Output("chat-status", "children"),
              Input("btn-chat-send", "n_clicks"), Input("chat-input", "n_submit"), Input("context-store", "data"),
              State("chat-input", "value"), State("chat-history-store", "data"), State("user-language", "value"),
              State("embedding-model", "value"), State("llm-model", "value"), State("custom-llm", "value"),
              prevent_initial_call=True, running=[(Output("btn-chat-send", "disabled"), True, False)])
@track_progress
def handle_chat(clicks, submit, context, question, history, language, embedding_model, llm, custom):
    if dash.ctx.triggered_id == "context-store":
        return [], [], "", None, ""
    if not question or not question.strip():
        fail("Enter a question first.")
        return dash.no_update, dash.no_update, "", dash.no_update, "Enter a question."
    try:
        result = query_video(selected_video(context), question, language, embedding_model, (custom or "").strip() or llm)
        response = f"The most relevant answer passage starts at {timestamp(result['start'])} and ends at {timestamp(result['end'])}."
        history = (history or []) + [{"role": "You", "text": question}, {"role": "Assistant", "text": response}]
        display = [html.Div([html.Span(item["role"], className="message-role"), html.P(item["text"])], className="chat-message " + ("user-message" if item["role"] == "You" else "assistant-message")) for item in history]
        return display, history, "", {"start": result["start"], "nonce": secrets.token_hex(8)}, ""
    except Exception as error:
        fail(error)
        return dash.no_update, dash.no_update, question, dash.no_update, f"Search unavailable: {error}"


@app.callback(Output("progress-jobs", "children"), Output("progress-log", "children"),
              Input("progress-poll", "n_intervals"))
def refresh_progress(tick):
    state = snapshot(flask.session["progress_id"])
    if not state["jobs"]:
        return html.P("Ready. Choose a task to see its progress here.", className="helper"), "No tasks started yet."
    active = [job for job in state["jobs"] if job["status"] == "running"]
    finished = [job for job in state["jobs"] if job["status"] != "running"][-3:]
    cards = []
    for job in active + finished:
        elapsed = int((job["ended"] or time.time()) - job["started"])
        label = job["step"]
        attributes = {"max": 1}
        if job["status"] == "completed":
            attributes["value"] = 1
        elif job["total"]:
            attributes.update(value=job["current"], max=job["total"])
            label += f" — {job['current']}/{job['total']} ({job['current'] * 100 // job['total']}%)"
        elif job["status"] == "failed":
            attributes["value"] = 0
        # No invented percentage for weight downloads or other uncounted steps.
        cards.append(html.Div([
            html.Div([html.Strong(job["title"]), html.Span(f"{job['status'].capitalize()} · {elapsed // 60}:{elapsed % 60:02d}")], className="progress-heading"),
            html.P(label), html.Progress(**attributes, **{"aria-label": job["title"] + " current step"}),
        ], className="progress-job " + job["status"]))
    log = "\n".join(f"[{event['time']}] {event['task']} · {event['message']}" for event in reversed(state["events"]))
    return cards, log


clientside_callback("""
function(result) {
    if (!result) return window.dash_clientside.no_update;
    const video = document.getElementById('video-player');
    if (!video) return window.dash_clientside.no_update;
    const seek = () => {
        video.currentTime = Math.max(0, Math.min(result.start, Number.isFinite(video.duration) ? video.duration : result.start));
        video.play().catch(() => {});
    };
    if (video.readyState >= 1) seek();
    else video.addEventListener('loadedmetadata', seek, {once: true});
    return String(result.nonce);
}
""", Output("seek-result", "children"), Input("seek-timestamp-store", "data"), prevent_initial_call=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False, threaded=True)
