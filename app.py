import base64
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import dash
from dash import ClientsideFunction, Input, Output, State, clientside_callback, dcc, html
import flask
import pandas as pd

# Import transcription and translation modules from your local workspace
from transcription import LANGUAGES, run_transcription
from translation import run_translation

try:
    from google.colab import drive, userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")

DEFAULT_OUTPUT_DIR = str(("/content/drive/MyDrive/fypoutput/" if IN_COLAB else Path.cwd() / "fypoutput").resolve())
Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

MODE_BOTH = "Both (transcription + translation)"
MODE_TRANSCRIBE = "Transcription only"
MODE_TRANSLATE = "Translation only"

# Initialize Dash App
app = dash.Dash(__name__, title="TATRO - Dash UI")
server = app.server  # Flask server instance

# --- Custom Flask Route for Direct PDF/Video Serving ---
@server.route("/files/<path:filepath>")
def serve_file(filepath):
    """Serves local files securely and directly without url encoding issues"""
    abs_path = Path("/" + filepath if not filepath.startswith("/") else filepath).resolve()
    if abs_path.exists():
        return flask.send_file(str(abs_path))
    return flask.abort(404, description="File not found")

# --- Helper Functions ---
def get_lang_by_name(name):
    for lang in LANGUAGES.values():
        if lang["name"] == name:
            return lang
    raise ValueError(f"Unknown language: {name}")

def derive_stem_from_dataset_json(path: Path):
    stem = path.stem
    if stem.endswith("_transcript_dataset"):
        stem = stem[: -len("_transcript_dataset")]
    return stem

def get_hf_token(cli_token=None):
    if cli_token:
        return cli_token.strip()
    if IN_COLAB:
        try:
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")

def save_dash_upload(contents, filename, target_dir):
    """Decodes dcc.Upload base64 content and saves it to a local folder"""
    if not contents:
        return None
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = (out_dir / filename).resolve()
    file_path.write_bytes(decoded)
    return file_path

def make_template_summary_pdf(output_path):
    """Create a template summary PDF"""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    text = "This is your summary"
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 18 Tf 72 720 ({escaped_text}) Tj ET"
    
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj",
        f"4 0 obj\n<< /Length {len(content_stream.encode('latin-1'))} >>\nstream\n{content_stream}\nendstream\nendobj",
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    ]
    
    pdf_bytes = bytearray(b"%PDF-1.4\n")
    object_offsets = []
    for obj in objects:
        object_offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj.encode("latin-1"))
        pdf_bytes.extend(b"\n")
    
    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for offset in object_offsets:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf_bytes.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    
    output_path.write_bytes(pdf_bytes)
    return output_path

def embedding_lookup(question, transcript_json_path=None):
    """Template retrieval function - returns random timestamp and placeholder message"""
    start_time = random.randint(0, 1800)
    return start_time, "Retrieval is not working now"

def create_pdf_preview_component(pdf_path):
    """Generates iframe component pointing to custom Flask server route"""
    if not pdf_path or not Path(pdf_path).exists():
        return html.Div(
            "No PDF available yet.", 
            style={"border": "1px solid #ddd", "padding": "20px", "textAlign": "center", "borderRadius": "8px", "minHeight": "650px", "backgroundColor": "#fff"}
        )
    
    abs_path = str(Path(pdf_path).resolve())
    file_url = f"/files/{abs_path.lstrip('/')}"
    
    return html.Div([
        html.Iframe(src=file_url, style={"width": "100%", "height": "650px", "border": "1px solid #ddd", "borderRadius": "8px"}),
        html.Div([
            html.A("Open PDF in a new tab", href=file_url, target="_blank", style={"fontWeight": "bold", "color": "#1e40af"})
        ], style={"textAlign": "center", "marginTop": "10px"})
    ])

# --- Dash Layout ---
app.layout = html.Div([
    # Stores for state management
    dcc.Store(id="context-store", data={"transcript_json_path": None, "video_path": None}),
    dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="seek-timestamp-store", data=None),

    html.H1("Video Transcription + Translation App", style={"fontFamily": "Arial", "marginBottom": "20px"}),

    # Options Row
    html.Div([
        html.Label("What do you want to run?", style={"fontWeight": "bold"}),
        dcc.RadioItems(
            id="mode-radio",
            options=[MODE_BOTH, MODE_TRANSCRIBE, MODE_TRANSLATE],
            value=MODE_BOTH,
            inline=True,
            style={"margin": "10px 0"}
        ),
        html.Label("Output Folder:", style={"fontWeight": "bold"}),
        dcc.Input(id="output-dir-box", type="text", value=DEFAULT_OUTPUT_DIR, style={"width": "100%", "padding": "8px", "marginBottom": "15px"})
    ], style={"padding": "15px", "backgroundColor": "#f8fafc", "borderRadius": "8px", "marginBottom": "20px"}),

    # Main Grid Layout: Video Left, PDF Right
    html.Div([
        # Left Column: Inputs & Controls
        html.Div([
            html.H3("Video & Inputs"),
            
            # Video Player
            html.Video(id="video-player", controls=True, style={"width": "100%", "maxHeight": "360px", "backgroundColor": "#000", "borderRadius": "8px"}),
            
            # Video Upload
            dcc.Upload(
                id="upload-video",
                children=html.Div(["Drag and Drop or ", html.A("Select Video (.mp4)")]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "margin": "10px 0"}
            ),

            # JSON Upload Section
            html.Div([
                dcc.Upload(
                    id="upload-json",
                    children=html.Div(["Drag & Drop or ", html.A("Upload Transcript JSON")]),
                    style={"width": "100%", "height": "50px", "lineHeight": "50px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "margin": "5px 0"}
                ),
                html.Button("Load JSON (Bypass Processing)", id="btn-load-json", style={"margin": "5px 0", "padding": "8px", "width": "100%", "backgroundColor": "#64748b", "color": "white", "border": "none", "borderRadius": "4px"})
            ]),

            # HuggingFace Token
            html.Label("HF Token:", style={"marginTop": "10px", "display": "block"}),
            dcc.Input(id="hf-token-box", type="password", placeholder="Enter HF Token...", style={"width": "100%", "padding": "8px", "marginBottom": "10px"}),

            # Languages Dropdown
            html.Div([
                html.Div([
                    html.Label("Spoken Language:"),
                    dcc.Dropdown(id="source-dd", options=["Hebrew", "Arabic", "English", "French", "Spanish", "Chinese", "Turkish"], value="Hebrew")
                ], style={"width": "48%", "display": "inline-block"}),
                html.Div([
                    html.Label("Translate Into:"),
                    dcc.Dropdown(id="target-dd", options=["Hebrew", "Arabic", "English", "French", "Spanish", "Chinese", "Turkish"], value="Arabic")
                ], style={"width": "48%", "float": "right", "display": "inline-block"})
            ], style={"marginBottom": "10px"}),

            # Sliders and Options
            html.Label("Merge Window:"),
            dcc.Slider(id="merge-slider", min=1, max=6, step=1, value=3, marks={i: str(i) for i in range(1, 7)}),
            
            dcc.Checklist(id="correction-check", options=[{"label": " ASR Correction (Slower)", "value": "use_corr"}], value=[], style={"margin": "10px 0"}),

            # Run Action Buttons
            html.Div([
                html.Button("Run Processing", id="btn-run", style={"padding": "10px 20px", "backgroundColor": "#2563eb", "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer", "marginRight": "10px"}),
                html.Button("Translate Existing Transcription", id="btn-run-existing", style={"padding": "10px 20px", "backgroundColor": "#059669", "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"})
            ], style={"marginTop": "15px"}),

            html.Div(id="status-box", style={"marginTop": "10px", "padding": "10px", "backgroundColor": "#f1f5f9", "borderRadius": "4px", "fontWeight": "bold"})

        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        # Right Column: PDF Preview with Upload Option Next to Heading
        html.Div([
            html.Div([
                html.H3("PDF Display", style={"display": "inline-block", "marginRight": "15px"}),
                dcc.Upload(
                    id="upload-existing-pdf",
                    children=html.Button("Upload PDF to Preview", style={"padding": "5px 12px", "backgroundColor": "#475569", "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                    style={"display": "inline-block"}
                )
            ], style={"marginBottom": "10px"}),
            html.Div(id="pdf-preview-container", children=create_pdf_preview_component(None))
        ], style={"width": "48%", "float": "right", "display": "inline-block", "verticalAlign": "top"})
    ]),

    html.Hr(style={"margin": "30px 0"}),

    # Summary Section
    html.Div([
        html.H2("Summary"),
        html.Div([
            html.Button("Generate Summary", id="btn-generate-summary", style={"padding": "8px 16px", "marginRight": "10px"}),
            html.Button("Show Summary", id="btn-show-summary", style={"padding": "8px 16px", "marginRight": "10px"}),
        ], style={"marginBottom": "10px"}),

        dcc.Upload(
            id="upload-summary-pdf",
            children=html.Div(["Upload Existing Summary PDF"]),
            style={"width": "100%", "height": "40px", "lineHeight": "40px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "marginBottom": "10px"}
        ),

        html.Div(id="summary-status-box", style={"marginBottom": "10px"}),
        html.Div(id="summary-preview-container")
    ]),

    html.Hr(style={"margin": "30px 0"}),

    # Chat Assistant Section
    html.Div([
        html.H2("Assistant Chat"),
        html.Div(id="chat-display", style={"height": "250px", "overflowY": "auto", "border": "1px solid #cbd5e1", "borderRadius": "8px", "padding": "15px", "backgroundColor": "#ffffff", "marginBottom": "10px"}),
        
        html.Div([
            dcc.Input(id="chat-input", type="text", placeholder="Type a question about the video...", style={"width": "80%", "padding": "10px", "borderRadius": "4px", "border": "1px solid #ccc"}),
            html.Button("Send", id="btn-chat-send", style={"width": "18%", "float": "right", "padding": "10px", "backgroundColor": "#1e293b", "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"})
        ])
    ])

], style={"maxWidth": "1300px", "margin": "0 auto", "padding": "20px", "fontFamily": "Segoe UI, sans-serif"})


# --- Dash Callbacks ---

# Video Upload Handler
@app.callback(
    Output("video-player", "src"),
    Input("upload-video", "contents"),
    State("upload-video", "filename"),
    State("output-dir-box", "value")
)
def update_video_source(contents, filename, output_dir):
    if contents and filename:
        saved_path = save_dash_upload(contents, filename, output_dir)
        return f"/files/{str(saved_path).lstrip('/')}"
    return None

# Direct Upload PDF Preview Handler
@app.callback(
    Output("pdf-preview-container", "children"),
    Input("upload-existing-pdf", "contents"),
    State("upload-existing-pdf", "filename"),
    State("output-dir-box", "value"),
    prevent_initial_call=True
)
def update_pdf_preview_direct(contents, filename, output_dir):
    if not contents:
        return dash.no_update
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    saved_path = save_dash_upload(contents, filename, output_path)
    return create_pdf_preview_component(saved_path)

# Load JSON Bypass Callback
@app.callback(
    Output("status-box", "children", allow_duplicate=True),
    Output("context-store", "data", allow_duplicate=True),
    Input("btn-load-json", "n_clicks"),
    State("upload-json", "contents"),
    State("upload-json", "filename"),
    State("output-dir-box", "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def load_json_only_callback(n_clicks, contents, filename, output_dir, context_data):
    if not contents:
        return "No JSON file selected for upload.", context_data
    saved_path = save_dash_upload(contents, filename, output_dir)
    context_data = context_data or {}
    context_data["transcript_json_path"] = str(saved_path)
    return f"Loaded {filename} successfully for Chat/Summary.", context_data

# Run Processing Callback
@app.callback(
    Output("status-box", "children"),
    Output("pdf-preview-container", "children", allow_duplicate=True),
    Output("context-store", "data"),
    Input("btn-run", "n_clicks"),
    Input("btn-run-existing", "n_clicks"),
    State("mode-radio", "value"),
    State("upload-video", "contents"),
    State("upload-video", "filename"),
    State("upload-json", "contents"),
    State("upload-json", "filename"),
    State("output-dir-box", "value"),
    State("source-dd", "value"),
    State("target-dd", "value"),
    State("merge-slider", "value"),
    State("correction-check", "value"),
    State("hf-token-box", "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def process_job_callback(n_run, n_run_existing, mode, vid_cont, vid_name, json_cont, json_name,
                         output_dir, source_lang, target_lang, merge_win, corr_val,
                         hf_token, context_data):
    
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"]
    context_data = context_data or {}
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Save inputs
    video_path = save_dash_upload(vid_cont, vid_name, output_path) if vid_cont else None
    transcript_path = save_dash_upload(json_cont, json_name, output_path) if json_cont else None

    # Handle "Translate Existing" override
    if triggered == "btn-run-existing.n_clicks":
        mode = MODE_TRANSLATE

    # Process Job standard execution
    if mode in (MODE_TRANSCRIBE, MODE_BOTH):
        if not video_path:
            return "Upload an MP4 video file first.", dash.no_update, context_data
        
        token = get_hf_token(hf_token)
        if not token:
            return "HF Token is required.", dash.no_update, context_data
        
        source_selected = get_lang_by_name(source_lang)
        config = {
            "whisper_model": "large-v3", "diarization_model": "pyannote/speaker-diarization-3.1",
            "qwen_model": "Qwen/Qwen3.5-9B", "max_chunk_duration": 30.0, "merge_window": int(merge_win),
            "max_new_tokens": 2048, "use_qwen_correction": bool(corr_val), "NbChunksToMerge": 2, "NbChunksOverlap": 1
        }
        
        try:
            dataset_df, dataset_json_out = run_transcription(video_path, output_path, token, source_selected, config)
            context_data["transcript_json_path"] = str(dataset_json_out)
            context_data["video_path"] = str(video_path)
        except Exception as e:
            return f"Error in transcription: {str(e)}", dash.no_update, context_data

        if mode == MODE_TRANSCRIBE:
            return "Transcription complete!", dash.no_update, context_data

    if mode == MODE_TRANSLATE:
        if not transcript_path and context_data.get("transcript_json_path"):
            transcript_path = Path(context_data["transcript_json_path"])
        if not transcript_path:
            return "Transcript JSON is required for Translation mode.", dash.no_update, context_data
        
        dataset_df = pd.read_json(transcript_path)
        context_data["transcript_json_path"] = str(transcript_path)

    # Translation Stage
    source_selected = get_lang_by_name(source_lang)
    target_selected = get_lang_by_name(target_lang)
    
    config = {"qwen_model": "Qwen/Qwen3.5-9B", "max_new_tokens": 2048, "NbChunksToMerge": 1, "NbChunksOverlap": 0}
    input_stem = video_path.stem if video_path else derive_stem_from_dataset_json(transcript_path)

    try:
        translated_df, translated_json_out, final_pdf_out = run_translation(
            dataset_df, Path(input_stem), output_path, source_selected, target_selected, config
        )
        context_data["transcript_json_path"] = str(translated_json_out)
        return "Processing complete!", create_pdf_preview_component(final_pdf_out), context_data
    except Exception as e:
        return f"Error in translation: {str(e)}", dash.no_update, context_data


# Summary Callback
@app.callback(
    Output("summary-status-box", "children"),
    Output("summary-preview-container", "children"),
    Input("btn-generate-summary", "n_clicks"),
    Input("upload-summary-pdf", "contents"),
    State("upload-summary-pdf", "filename"),
    State("output-dir-box", "value"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def summary_handler(gen_clicks, upload_cont, upload_name, output_dir, context_data):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"]
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()

    if triggered == "upload-summary-pdf.contents":
        saved_path = save_dash_upload(upload_cont, upload_name, output_path)
        return f"Loaded summary: {upload_name}", create_pdf_preview_component(saved_path)

    # Generate summary logic
    transcript_json = context_data.get("transcript_json_path") if context_data else None
    if not transcript_json:
        return "No transcript JSON found to summarize.", create_pdf_preview_component(None)

    summary_pdf_path = output_path / f"{Path(transcript_json).stem}_summary.pdf"
    summary_pdf_path = make_template_summary_pdf(summary_pdf_path)
    return f"Summary created at {summary_pdf_path.name}", create_pdf_preview_component(summary_pdf_path)


# Chat Callback
@app.callback(
    Output("chat-display", "children"),
    Output("chat-history-store", "data"),
    Output("chat-input", "value"),
    Output("seek-timestamp-store", "data"),
    Input("btn-chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history-store", "data"),
    State("context-store", "data"),
    prevent_initial_call=True
)
def handle_chat(n_clicks, n_submit, user_message, history_data, context_data):
    if not user_message:
        return dash.no_update, dash.no_update, "", dash.no_update

    history_data = history_data or []
    transcript_json = context_data.get("transcript_json_path") if context_data else None
    
    timestamp, response_text = embedding_lookup(user_message, transcript_json)
    
    history_data.append({"role": "user", "text": user_message})
    history_data.append({"role": "assistant", "text": response_text})

    # Build Chat UI Bubbles
    chat_ui = []
    for msg in history_data:
        is_user = msg["role"] == "user"
        chat_ui.append(html.Div(
            msg["text"],
            style={
                "backgroundColor": "#2563eb" if is_user else "#e2e8f0",
                "color": "white" if is_user else "black",
                "padding": "8px 12px",
                "borderRadius": "12px",
                "maxWidth": "70%",
                "marginBottom": "8px",
                "marginLeft": "auto" if is_user else "0",
                "marginRight": "0" if is_user else "auto",
            }
        ))

    return chat_ui, history_data, "", timestamp


# Native Javascript Clientside Callback to jump video cursor to timestamp
clientside_callback(
    """
    function(timestamp) {
        if (timestamp !== null && timestamp !== undefined) {
            var vid = document.getElementById('video-player');
            if (vid) {
                vid.currentTime = parseFloat(timestamp);
                vid.play();
                console.log("Dash Clientside Seek to: " + timestamp + "s");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("seek-timestamp-store", "data", allow_duplicate=True),
    Input("seek-timestamp-store", "data"),
    prevent_initial_call=True
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)