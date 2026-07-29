import os
import re
import json
import base64
import secrets
import subprocess
import tempfile
import time
import requests
import anthropic
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

app = FastAPI(title="Content Demolition Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "demolition-secret")

# Needed by the scheduler clock, which runs with no browser session attached:
# it mints its own Google access token from the refresh token saved at sign-in.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# This worker's own public URL — Instagram fetches the video from it.
WORKER_PUBLIC_URL = os.environ.get("WORKER_PUBLIC_URL", "")
TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
# Set SCHEDULER_ENABLED=0 to stop the clock firing (e.g. on a staging copy that
# shares the same Firestore — otherwise both copies would post the same video).
SCHEDULER_ENABLED = os.environ.get("SCHEDULER_ENABLED", "1") not in ("0", "false", "False")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Firestore helpers ────────────────────────────────────────────────────────

def firestore_get(path: str):
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{path}"
    r = requests.get(url)
    return r.json()

def firestore_query(collection: str, field: str, value: str):
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery"
    body = {"structuredQuery": {"from": [{"collectionId": collection}], "where": {"fieldFilter": {"field": {"fieldPath": field}, "op": "EQUAL", "value": {"stringValue": value}}}, "limit": 1}}
    r = requests.post(url, json=body)
    return r.json()

def _fs_value(v):
    """Encode one Python value as a Firestore REST value, recursively.

    Nested maps/arrays matter for the scheduler: a post's per-platform `results`
    is a map of maps, and flattening it to strings would lose the structure the
    UI reads back."""
    if v is None:
        return {"nullValue": None}
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, (int, float)):
        return {"doubleValue": float(v)}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_fs_value(item) for item in v]}}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {kk: _fs_value(vv) for kk, vv in v.items()}}}
    return {"stringValue": str(v)}

def firestore_patch(path: str, fields: dict):
    url_fields = "&".join([f"updateMask.fieldPaths={k}" for k in fields])
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{path}?{url_fields}"
    fs_fields = {k: _fs_value(v) for k, v in fields.items()}
    r = requests.patch(url, json={"fields": fs_fields})
    return r.json()

def firestore_create(collection: str, doc_id: str, fields: dict):
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection}/{doc_id}"
    fs_fields = {}
    for k, v in fields.items():
        if v is None:
            fs_fields[k] = {"nullValue": None}
        elif isinstance(v, bool):
            fs_fields[k] = {"booleanValue": v}
        elif isinstance(v, int) or isinstance(v, float):
            fs_fields[k] = {"doubleValue": float(v)}
        else:
            fs_fields[k] = {"stringValue": str(v)}
    r = requests.patch(url, json={"fields": fs_fields})
    return r.json()

def _parse_fs_value(v: dict):
    """Decode one Firestore REST value back to Python (mirror of _fs_value)."""
    if "stringValue" in v:
        return v["stringValue"]
    if "booleanValue" in v:
        return v["booleanValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_parse_fs_value(item) for item in v["arrayValue"].get("values", [])]
    if "mapValue" in v:
        return {kk: _parse_fs_value(vv) for kk, vv in v["mapValue"].get("fields", {}).items()}
    return None

def parse_fs_doc(doc: dict) -> dict:
    result = {}
    if not doc or "fields" not in doc:
        return result
    for k, v in doc["fields"].items():
        if "stringValue" in v:
            result[k] = v["stringValue"]
        elif "booleanValue" in v:
            result[k] = v["booleanValue"]
        elif "integerValue" in v:
            result[k] = int(v["integerValue"])
        elif "doubleValue" in v:
            result[k] = float(v["doubleValue"])
        elif "arrayValue" in v:
            result[k] = [_parse_fs_value(item) for item in v["arrayValue"].get("values", [])]
        elif "mapValue" in v:
            result[k] = {kk: _parse_fs_value(vv) for kk, vv in v["mapValue"].get("fields", {}).items()}
        elif "nullValue" in v:
            result[k] = None
    return result

# ─── Video analysis with Claude Vision ───────────────────────────────────────

def extract_frames(video_url: str, num_frames: int = 3, extra_headers: dict = {}) -> list[str]:
    """Download video and extract frames as base64 images using FFmpeg."""
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")

        # ── Stage 1: download (with optional auth headers for Google Drive) ──
        try:
            r = requests.get(video_url, stream=True, timeout=180, headers=extra_headers)
            if r.status_code != 200:
                raise RuntimeError(f"DOWNLOAD_FAILED http {r.status_code}: {r.text[:150]}")
            total = 0
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    total += len(chunk)
            if total == 0:
                raise RuntimeError("DOWNLOAD_FAILED empty file")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"DOWNLOAD_FAILED {type(e).__name__}: {str(e)[:150]}")

        # ── Stage 2: probe + frame extraction ──
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True
        )
        duration = 30  # default
        try:
            info = json.loads(probe.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 30))
                    break
        except:
            pass

        err_holder = [""]
        # -nostdin prevents the interactive "Press [q]" hang; DEVNULL stdin too.
        def run_ff(args):
            p = subprocess.run(["ffmpeg", "-nostdin", "-y", *args],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL)
            if p.returncode != 0:
                err = p.stderr or ""
                # surface a REAL error line if present, else the tail
                lines = [ln for ln in err.splitlines()
                         if any(k in ln.lower() for k in ("error", "invalid", "could not", "no such", "killed", "unable", "failed"))]
                err_holder[0] = (lines[-1] if lines else err[-160:]).strip()
            return p

        def collect(prefix):
            for fp in sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith(prefix)):
                with open(fp, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())

        # ── 2a: fast input-seek per timestamp (only reliable if duration is real) ──
        have_duration = duration and duration > 1.5
        if have_duration:
            timestamps = [0.5, min(2.0, duration * 0.1), duration * 0.5, duration * 0.85]
            for i, ts in enumerate(timestamps[:num_frames]):
                run_ff(["-ss", str(ts), "-i", video_path, "-frames:v", "1",
                        "-q:v", "3", "-vf", "scale=512:-1", "-an", os.path.join(tmp, f"a_{i}.jpg")])
            collect("a_")

        # ── 2b: representative frames via thumbnail filter — NO duration dependency.
        # Handles Dolby Vision / short clips where ffprobe couldn't read duration. ──
        if not frames:
            run_ff(["-i", video_path, "-vf", "scale=512:-1,thumbnail=120",
                    "-frames:v", str(num_frames), "-q:v", "3", "-an", os.path.join(tmp, "b_%02d.jpg")])
            collect("b_")

        # ── 2c: last resort — first decodable frame from the very start (no seek). ──
        if not frames:
            run_ff(["-i", video_path, "-frames:v", "1", "-q:v", "3",
                    "-vf", "scale=512:-1", "-an", os.path.join(tmp, "c_0.jpg")])
            collect("c_")

        if not frames:
            raise RuntimeError(f"FFMPEG_FAILED no frames: {err_holder[0] or 'unknown'}")

    return frames

def analyse_video_with_claude(frames: list[str], caption: str = "", niche: str = "", taxonomy: dict = None) -> dict:
    """Send frames to Claude Vision for content analysis."""
    if not frames:
        return {"error": "No frames extracted"}

    content = []
    labels = ["Hook frame (first 0.5s)", "Early frame (2s)", "Mid video", "Near end"]

    for i, frame_b64 in enumerate(frames):
        content.append({
            "type": "text",
            "text": f"**{labels[i] if i < len(labels) else f'Frame {i+1}'}:**"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}
        })

    # Build taxonomy hint for Claude
    taxonomy_hint = ""
    if taxonomy and taxonomy.get("categories"):
        cat_list = []
        for cat in taxonomy["categories"]:
            subs = [s["name"] for s in cat.get("subcategories", [])]
            if subs:
                cat_list.append(f'{cat["id"]} ({cat["name"]}) → subcategories: {", ".join(subs)}')
            else:
                cat_list.append(f'{cat["id"]} ({cat["name"]})')
        taxonomy_hint = f"\n\nThe operator uses these categories:\n" + "\n".join(cat_list) + "\nAssign content_type using the category IDs above."

    content.append({
        "type": "text",
        "text": f"""Caption: "{caption}"
Niche: {niche or "general"}{taxonomy_hint}

Tag this video for a B-roll library. Respond ONLY with valid JSON:
{{
  "content_type": "talking_reel|action_reel|broll|carousel|transformation|tutorial|vlog",
  "tags": ["Give the 5 MOST useful, SPECIFIC tags so someone can sort this clip WITHOUT watching it. Cover every dimension you can see: ACTIVITY/SPORT (running, cycling, swimming, climbing, motorbike, gym, weightlifting, flips, hiking, surfing), PLACE/SETTING (ocean, beach, mountains, gym, kitchen, street, forest, pool, desert), OBJECTS (bike, barbell, food, coffee, car, dog, drone), PEOPLE (friends, family, girlfriend, group, solo), VIBE/THEME (travel, adventure, food, party, nature, sunset, training), and FORMAT (vlog, talking-head, broll, action). Be concrete and specific — prefer 'ocean kayaking' over just 'outdoor'. Lowercase. Example for a clip of him doing flips off a boat into the sea with friends: [\\"flips\\", \\"jumping\\", \\"ocean\\", \\"boat\\", \\"friends\\", \\"swimming\\", \\"summer\\", \\"vlog\\", \\"adventure\\"]."],
  "has_face": true/false,
  "topic": "brief topic description (3-6 words)"
}}"""
    })

    message = claude.messages.create(
        # Sonnet — ~10x cheaper than Opus, excellent for vision tagging.
        # Override per-deploy with SCAN_MODEL env var if you ever want Opus/Haiku.
        model=os.environ.get("SCAN_MODEL", "claude-sonnet-4-6"),
        max_tokens=500,
        messages=[{"role": "user", "content": content}]
    )

    raw = message.content[0].text
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        return json.loads(json_match.group()) if json_match else {"raw": raw}
    except:
        return {"raw": raw}

# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Content Demolition Worker",
        "version": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "unknown")[:7],
        "agents": ["video-analyser", "drive-scanner", "reel-builder", "auto-poster"],
    }

@app.get("/test-reel")
def test_reel(url: str):
    """De-risk test: can we download an IG reel from its link? Browser-friendly GET.
    Returns whether yt-dlp fetched the video + size + caption, or the error."""
    import shutil
    if not shutil.which("yt-dlp"):
        return {"ok": False, "error": "yt-dlp not installed yet (redeploy)"}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "reel.%(ext)s")
        try:
            proc = subprocess.run(
                ["yt-dlp", "--no-playlist", "-f", "mp4/best", "--print-json",
                 "-o", out, url],
                capture_output=True, text=True, timeout=120
            )
        except Exception as e:
            return {"ok": False, "error": f"yt-dlp crashed: {str(e)[:200]}"}
        files = [f for f in os.listdir(tmp)]
        got = next((f for f in files if not f.endswith(".json")), None)
        if not got:
            return {"ok": False, "error": "download failed", "detail": (proc.stderr or "")[-400:]}
        size_mb = round(os.path.getsize(os.path.join(tmp, got)) / 1024 / 1024, 1)
        # try to read caption/duration from the printed json
        caption, duration = "", None
        try:
            meta = json.loads((proc.stdout or "").strip().splitlines()[-1])
            caption = (meta.get("description") or "")[:200]
            duration = meta.get("duration")
        except Exception:
            pass
        return {"ok": True, "downloaded": got, "size_mb": size_mb, "duration_sec": duration, "caption": caption}

class AnalyseVideoRequest(BaseModel):
    video_url: str
    caption: Optional[str] = ""
    niche: Optional[str] = ""
    clip_id: Optional[str] = None  # Firestore clip doc ID to update

@app.post("/analyse-video")
def analyse_video(req: AnalyseVideoRequest):
    """Extract frames from a video and analyse with Claude Vision."""
    try:
        frames = extract_frames(req.video_url)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames")

        analysis = analyse_video_with_claude(frames, req.caption, req.niche)

        # Save to Firestore if clip_id provided
        if req.clip_id and "error" not in analysis:
            firestore_patch(f"clips/{req.clip_id}", {
                "aiContentType": analysis.get("content_type", "unknown"),
                "aiHasFace": analysis.get("has_face", False),
                "aiIsTalking": analysis.get("is_talking_to_camera", False),
                "aiEnergyLevel": analysis.get("energy_level", "medium"),
                "aiHookType": analysis.get("hook_type", ""),
                "aiHookQuality": analysis.get("hook_quality", 5),
                "aiTopic": analysis.get("topic", ""),
                "aiSetting": analysis.get("setting", ""),
                "aiUsabilityScore": analysis.get("usability_score", 5),
                "aiNotes": analysis.get("notes", ""),
                "aiAnalysedAt": datetime.utcnow().isoformat(),
            })

        return {"success": True, "analysis": analysis, "frames_extracted": len(frames)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ScanDriveRequest(BaseModel):
    client_id: str
    google_access_token: Optional[str] = None
    taxonomy: Optional[dict] = None  # custom category structure from operator
    protected_folders: Optional[list[str]] = None  # folder paths to skip (frozen/additive)
    folder_filter: Optional[str] = None  # if set, only scan clips in this folder (+ subfolders)
    batch_size: Optional[int] = 50  # how many clips to analyse this call


def _is_protected(path: str, protected: list[str]) -> bool:
    """A clip is protected if its folder path is, or is inside, a protected folder."""
    for folder in protected:
        if path == folder or path.startswith(folder + "/"):
            return True
    return False

def _select_scan_candidates(req: ScanDriveRequest):
    """Return (niche, [clip dicts to scan]) honoring protection + folder filter + batch size."""
    result = firestore_query("users", "clientId", req.client_id)
    doc = next((r.get("document") for r in result if r.get("document")), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Client not found")
    niche = parse_fs_doc(doc).get("niche", "")

    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery"
    batch_size = req.batch_size or 50
    PAGE = 1000

    # Page through ALL of the client's clips (a client can have 2000+), so we don't
    # miss unscanned clips sitting beyond the first page.
    candidates = []
    offset = 0
    while len(candidates) < batch_size:
        body = {
            "structuredQuery": {
                "from": [{"collectionId": "clips"}],
                "where": {
                    "fieldFilter": {"field": {"fieldPath": "clientId"}, "op": "EQUAL", "value": {"stringValue": req.client_id}}
                },
                "offset": offset,
                "limit": PAGE,
            }
        }
        page = requests.post(url, json=body).json()
        rows = [it for it in page if it.get("document")]
        if not rows:
            break  # no more clips

        for item in rows:
            if len(candidates) >= batch_size:
                break
            doc = item["document"]
            clip = parse_fs_doc(doc)
            path = clip.get("path", "")
            if clip.get("aiAnalysedAt"):
                continue
            if clip.get("mediaType") == "image":
                continue  # scanner is video-only — skip photos
            if req.protected_folders and _is_protected(path, req.protected_folders):
                continue
            # Match folder by EFFECTIVE path (where the operator filed it in-app),
            # falling back to real Drive path — so "scan ספורט" catches clips moved
            # into ספורט even if their real Drive folder is still elsewhere.
            eff_path = clip.get("organizedPath") or path
            if req.folder_filter and not (eff_path == req.folder_filter or eff_path.startswith(req.folder_filter + "/")):
                continue
            clip["_id"] = doc["name"].split("/")[-1]
            candidates.append(clip)

        if len(rows) < PAGE:
            break  # reached the last page
        offset += PAGE

    return niche, candidates


def _run_scan(req: ScanDriveRequest, niche: str, candidates: list):
    """Analyse the selected clips and write results to Firestore. Runs in background.
    Writes live progress + errors to scanStatus/{client_id} so the app can show them."""
    total = len(candidates)
    done = 0
    errors = 0
    last_error = ""

    def status():
        firestore_patch(f"scanStatus/{req.client_id}", {
            "running": (done + errors) < total,
            "total": total, "done": done, "errors": errors,
            "lastError": last_error[:300],
            "updatedAt": datetime.utcnow().isoformat(),
        })

    status()
    for clip in candidates:
        clip_id = clip["_id"]
        video_url = clip.get("bunnyUrl") or clip.get("driveUrl") or clip.get("downloadUrl")
        drive_file_id = clip.get("driveFileId")
        if not video_url and drive_file_id and req.google_access_token:
            video_url = f"https://www.googleapis.com/drive/v3/files/{drive_file_id}?alt=media"
        if not video_url:
            errors += 1
            last_error = f"{clip.get('name','?')}: no video URL"
            status()
            continue
        download_headers = {}
        if "googleapis.com" in video_url and req.google_access_token:
            download_headers["Authorization"] = f"Bearer {req.google_access_token}"
        try:
            frames = extract_frames(video_url, num_frames=3, extra_headers=download_headers)
            if frames:
                analysis = analyse_video_with_claude(frames, clip.get("caption", ""), niche, req.taxonomy)
                fields = {
                    "aiContentType": analysis.get("content_type", "unknown"),
                    "aiHasFace": str(analysis.get("has_face", False)),
                    "aiTopic": analysis.get("topic", ""),
                    "aiAnalysedAt": datetime.utcnow().isoformat(),
                }
                tags = analysis.get("tags")
                if isinstance(tags, list):
                    fields["aiTags"] = [str(t).lower().strip() for t in tags if t]
                firestore_patch(f"clips/{clip_id}", fields)
                done += 1
            else:
                errors += 1
                last_error = f"{clip.get('name','?')}: no frames extracted"
        except Exception as e:
            errors += 1
            last_error = f"{clip.get('name','?')}: {e}"
            print(f"Scan error on {clip_id}: {e}")
        status()


@app.post("/scan-drive")
def scan_drive(req: ScanDriveRequest, background_tasks: BackgroundTasks):
    """Kick off a background scan and return immediately with how many will be scanned.
    The frontend polls Firestore to watch tags appear."""
    try:
        niche, candidates = _select_scan_candidates(req)
        background_tasks.add_task(_run_scan, req, niche, candidates)
        return {"success": True, "started": True, "to_scan": len(candidates), "has_token": bool(req.google_access_token)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Push to Drive: mirror the in-app structure into real Google Drive ────────

def _drive_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _drive_find_folder(name: str, parent_id: str, token: str) -> Optional[str]:
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    q = (f"name = '{safe}' and '{parent_id}' in parents "
         "and mimeType = 'application/vnd.google-apps.folder' and trashed = false")
    r = requests.get("https://www.googleapis.com/drive/v3/files",
                     params={"q": q, "fields": "files(id,name)", "spaces": "drive"},
                     headers=_drive_headers(token))
    files = r.json().get("files", [])
    return files[0]["id"] if files else None

def _drive_create_folder(name: str, parent_id: str, token: str) -> str:
    r = requests.post("https://www.googleapis.com/drive/v3/files",
                      headers={**_drive_headers(token), "Content-Type": "application/json"},
                      json={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]})
    data = r.json()
    if "id" not in data:
        raise RuntimeError(f"create folder failed: {data}")
    return data["id"]

def _resolve_path(path: str, root_id: str, token: str, cache: dict) -> str:
    """Resolve (creating if needed) a 'a/b/c' path to a Drive folder id, under root_id."""
    if path in cache:
        return cache[path]
    parent = root_id
    cur = ""
    for part in [p for p in path.split("/") if p]:
        cur = f"{cur}/{part}" if cur else part
        if cur in cache:
            parent = cache[cur]
            continue
        fid = _drive_find_folder(part, parent, token) or _drive_create_folder(part, parent, token)
        cache[cur] = fid
        parent = fid
    cache[path] = parent
    return parent

def _drive_move(file_id: str, add_parent: str, token: str):
    g = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                     params={"fields": "parents"}, headers=_drive_headers(token))
    parents = g.json().get("parents", [])
    if add_parent in parents:
        return  # already in the home folder — nothing to move
    old = ",".join(parents)
    r = requests.patch(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                       params={"addParents": add_parent, "removeParents": old, "fields": "id,parents"},
                       headers=_drive_headers(token))
    if r.status_code != 200:
        raise RuntimeError(f"move failed http {r.status_code}: {r.text[:150]}")

def _drive_create_shortcut(file_id: str, name: str, parent_id: str, token: str):
    """Create a Drive shortcut to file_id inside parent_id (skip if one already there)."""
    # already linked here?
    q = (f"'{parent_id}' in parents and trashed = false "
         "and mimeType = 'application/vnd.google-apps.shortcut'")
    existing = requests.get("https://www.googleapis.com/drive/v3/files",
                            params={"q": q, "fields": "files(id,shortcutDetails)"},
                            headers=_drive_headers(token)).json().get("files", [])
    for sc in existing:
        if (sc.get("shortcutDetails") or {}).get("targetId") == file_id:
            return  # shortcut already exists
    r = requests.post("https://www.googleapis.com/drive/v3/files",
                      headers={**_drive_headers(token), "Content-Type": "application/json"},
                      json={"name": name, "mimeType": "application/vnd.google-apps.shortcut",
                            "parents": [parent_id], "shortcutDetails": {"targetId": file_id}})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"shortcut failed http {r.status_code}: {r.text[:120]}")

class PushRequest(BaseModel):
    client_id: str
    google_access_token: str
    root_folder_id: str
    moves: list  # [{"drive_file_id":..., "target_path":..., "extra_paths":[], "name":..., "clip_id":...}]

def _run_push(req: PushRequest):
    cache: dict = {}
    total = len(req.moves)
    done = 0
    errors = 0
    last_error = ""

    def status():
        firestore_patch(f"pushStatus/{req.client_id}", {
            "running": (done + errors) < total,
            "total": total, "done": done, "errors": errors,
            "lastError": last_error[:300],
            "updatedAt": datetime.utcnow().isoformat(),
        })

    status()
    for m in req.moves:
        try:
            target_id = _resolve_path(m["target_path"], req.root_folder_id, req.google_access_token, cache)
            _drive_move(m["drive_file_id"], target_id, req.google_access_token)
            # Multi-folder: create a shortcut in each "also add" folder
            for extra in (m.get("extra_paths") or []):
                extra_id = _resolve_path(extra, req.root_folder_id, req.google_access_token, cache)
                _drive_create_shortcut(m["drive_file_id"], m.get("name", "clip"), extra_id, req.google_access_token)
            # Settle the clip: its real Drive location is now target_path, so it
            # won't be re-pushed next time. Clear the pending organizedPath.
            if m.get("clip_id"):
                firestore_patch(f"clips/{m['clip_id']}", {"path": m["target_path"], "organizedPath": ""})
            done += 1
        except Exception as e:
            errors += 1
            last_error = f"{m.get('name', '?')}: {e}"
            print(f"Push error: {last_error}")
        status()

@app.post("/push-to-drive")
def push_to_drive(req: PushRequest, background_tasks: BackgroundTasks):
    """Move files in the client's real Drive to mirror the in-app structure.
    Runs in background; frontend polls pushStatus/{client_id}."""
    background_tasks.add_task(_run_push, req)
    return {"started": True, "to_move": len(req.moves)}


# Meta reuses type "OAuthException" for rate limits, so the type alone proves nothing.
# These codes are the ones that fix themselves; everything else is treated as a dead
# credential, because a broken connection must never go on looking healthy.
_IG_TRANSIENT_CODES = {1, 2, 4, 17, 32, 341, 613}


def _is_ig_auth_error(payload: dict) -> bool:
    """False only for failures that resolve on their own. Flagging a rate limit or a Meta
    5xx as 'reconnect me' takes an account offline for days over a hiccup."""
    err = (payload or {}).get("error") or {}
    return err.get("code") not in _IG_TRANSIENT_CODES


def _refresh_ig_tokens_core():
    """Keep every connected Instagram account ALIVE so clients connect ONCE and never have
    to reconnect. IG long-lived tokens last 60 days but can be refreshed for another 60 while
    still valid. Also re-fetches the profile photo + follower count (those go stale faster
    than the token). Idempotent + safe to run repeatedly."""
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery"
    body = {"structuredQuery": {"from": [{"collectionId": "users"}], "where": {"fieldFilter": {
        "field": {"fieldPath": "instagramConnected"}, "op": "EQUAL", "value": {"booleanValue": True}}}}}
    rows = requests.post(url, json=body).json()

    refreshed, failed, results = 0, 0, []
    for row in rows:
        doc = row.get("document")
        if not doc:
            continue
        data = parse_fs_doc(doc)
        doc_id = doc["name"].split("/")[-1]
        token = data.get("instagramAccessToken")
        username = data.get("instagramUsername", "?")
        if not token:
            continue
        try:
            rr = requests.get("https://graph.instagram.com/refresh_access_token",
                              params={"grant_type": "ig_refresh_token", "access_token": token}, timeout=30)
            rj = rr.json()
            new_token = rj.get("access_token")
            if not new_token:
                # Only a genuine auth failure means "reconnect me". Flagging a rate limit or a
                # Meta 5xx here would block the account's posting until a human reconnects it,
                # for a blip that fixes itself by the next pass.
                failed += 1
                results.append({"username": username, "ok": False, "error": str(rj)[:150]})
                if _is_ig_auth_error(rj):
                    firestore_patch(f"users/{doc_id}", {"instagramTokenError": str(rj)[:250]})
                else:
                    print(f"[ig-refresh] transient error for {username}, not flagging: {str(rj)[:150]}")
                continue
            fields = {"instagramAccessToken": new_token,
                      "instagramConnectedAt": datetime.utcnow().isoformat() + "Z",
                      "instagramTokenError": ""}
            # Refresh display fields so the photo/followers don't go stale
            try:
                pr = requests.get("https://graph.instagram.com/v19.0/me",
                                  params={"fields": "username,followers_count,profile_picture_url",
                                          "access_token": new_token}, timeout=30).json()
                # NB: field names must match what the connect callback + UI use:
                # `profilePhoto` (raw URL) and `followers` (formatted "3.0K").
                if pr.get("username"):
                    fields["instagramUsername"] = pr["username"]
                if pr.get("profile_picture_url"):
                    fields["profilePhoto"] = pr["profile_picture_url"]
                if pr.get("followers_count") is not None:
                    fields["followers"] = f"{pr['followers_count'] / 1000:.1f}K"
            except Exception:
                pass
            firestore_patch(f"users/{doc_id}", fields)
            refreshed += 1
            results.append({"username": username, "ok": True})
        except Exception as e:
            failed += 1
            results.append({"username": username, "ok": False, "error": str(e)[:150]})

    return {"refreshed": refreshed, "failed": failed, "results": results}


@app.post("/refresh-ig-tokens")
def refresh_ig_tokens(x_worker_secret: str = Header(default="")):
    """Manual trigger for the IG token refresh (also runs automatically every ~3 days)."""
    if x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    return _refresh_ig_tokens_core()


def _ig_token_refresh_loop():
    """Self-contained scheduler: refresh all connected IG tokens every ~3 days so clients
    never have to reconnect. Runs in a daemon thread — no external cron needed."""
    import time
    time.sleep(60)  # let the app finish booting
    while True:
        try:
            summary = _refresh_ig_tokens_core()
            print(f"[ig-refresh] {summary.get('refreshed')} refreshed, {summary.get('failed')} failed")
        except Exception as e:
            print(f"[ig-refresh] loop error: {e}")
        time.sleep(3 * 24 * 3600)  # every 3 days (well inside the 60-day expiry window)


@app.on_event("startup")
def _start_ig_refresh():
    import threading
    threading.Thread(target=_ig_token_refresh_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Reel auto-poster: publish a Drive reel to Instagram.
# Instagram's API needs the video at a PUBLIC url, so the worker downloads the
# reel from Drive and serves it at /reel-media/{id} just long enough for Meta
# to fetch it, then creates the media container and publishes it.
# ---------------------------------------------------------------------------
_SERVED_FILES: dict = {}  # media_id -> local filepath (public for Meta to fetch)


@app.get("/reel-media/{media_id}")
def serve_reel_media(media_id: str):
    path = _SERVED_FILES.get(media_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")


class PublishReelRequest(BaseModel):
    post_id: Optional[str] = None          # Firestore scheduledPosts doc id (status updates)
    drive_file_id: str
    # Optional because the scheduler clock has no browser session to borrow tokens from —
    # it looks up / mints every credential itself from what's stored in Firestore.
    google_access_token: Optional[str] = None   # to download the video from Drive
    ig_access_token: Optional[str] = None       # the client's Instagram token
    worker_public_url: Optional[str] = None     # this worker's public base url (so Meta can fetch)
    caption: Optional[str] = ""
    client_id: Optional[str] = None             # which connected account to post as
    platforms: list[str] = ["instagram"]
    youtube_title: Optional[str] = None
    youtube_privacy: str = "private"
    tiktok_privacy: str = "SELF_ONLY"


# ─── Credentials the clock fetches for itself ────────────────────────────────

def _google_access_token() -> str:
    """Mint a fresh Google access token from the refresh token saved at sign-in.

    The web app stores it at integrations/google when Ilai signs in with Google.
    Without this the scheduler could only ever run while a browser tab was open."""
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise RuntimeError("worker is missing GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET")
    doc = firestore_get("integrations/google")
    data = parse_fs_doc(doc)
    refresh = data.get("refreshToken")
    if not refresh:
        raise RuntimeError("no saved Google login — open the app and sign in with Google once")
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh,
    }, timeout=30)
    j = r.json()
    token = j.get("access_token")
    if not token:
        firestore_patch("integrations/google", {"error": str(j)[:250]})
        raise RuntimeError(f"Google token refresh failed: {str(j)[:150]}")
    return token


def _client_doc(client_id: str):
    """Find a connected account's Firestore doc by its clientId slug."""
    rows = firestore_query("users", "clientId", client_id)
    for row in rows if isinstance(rows, list) else []:
        doc = row.get("document")
        if doc:
            return doc["name"].split("/")[-1], parse_fs_doc(doc)
    return None, {}


def _tiktok_access_token(doc_id: str, data: dict) -> tuple:
    """Return (access_token, granted_scope), refreshing first if it's close to expiry.

    TikTok access tokens only last 24h, so a video scheduled for tomorrow will almost
    always need this refresh before it can post."""
    token = data.get("tiktokAccessToken")
    refresh = data.get("tiktokRefreshToken")
    scope = data.get("tiktokScope") or ""
    expires_at = data.get("tiktokTokenExpiresAt") or ""

    still_fresh = False
    if token and expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()
            still_fresh = exp - time.time() > 600  # 10 min safety margin
        except ValueError:
            still_fresh = False
    if still_fresh:
        return token, scope

    if not refresh:
        raise RuntimeError("TikTok isn't connected for this account — connect it first")
    if not (TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET):
        raise RuntimeError("worker is missing TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET")

    r = requests.post("https://open.tiktokapis.com/v2/oauth/token/",
                      headers={"Content-Type": "application/x-www-form-urlencoded"},
                      data={"client_key": TIKTOK_CLIENT_KEY,
                            "client_secret": TIKTOK_CLIENT_SECRET,
                            "grant_type": "refresh_token",
                            "refresh_token": refresh}, timeout=30)
    j = r.json()
    new_token = j.get("access_token")
    if not new_token:
        firestore_patch(f"users/{doc_id}", {"tiktokTokenError": str(j)[:250]})
        raise RuntimeError(f"TikTok token refresh failed: {str(j)[:150]}")
    scope = j.get("scope") or scope
    firestore_patch(f"users/{doc_id}", {
        "tiktokAccessToken": new_token,
        "tiktokRefreshToken": j.get("refresh_token") or refresh,
        "tiktokScope": scope,
        "tiktokTokenExpiresAt": datetime.utcfromtimestamp(
            time.time() + (j.get("expires_in") or 86400)).isoformat() + "Z",
        "tiktokTokenError": "",
    })
    return new_token, scope


# ─── Per-platform publishers ─────────────────────────────────────────────────
# Each returns a dict shaped like the UI's PlatformResult, and raises on failure.

def _publish_instagram(tmp_path: str, media_id: str, caption: str,
                       ig_token: str, public_base: str) -> dict:
    """Instagram pulls the file from a public URL, so serve it while Meta fetches."""
    if not ig_token:
        raise RuntimeError("this account has no connected Instagram")
    if not public_base:
        raise RuntimeError("worker has no public URL configured (WORKER_PUBLIC_URL)")

    _SERVED_FILES[media_id] = tmp_path
    try:
        public_url = f"{public_base.rstrip('/')}/reel-media/{media_id}"
        c = requests.post("https://graph.instagram.com/v19.0/me/media",
                          params={"media_type": "REELS", "video_url": public_url,
                                  "caption": caption or "", "access_token": ig_token},
                          timeout=60).json()
        creation_id = c.get("id")
        if not creation_id:
            raise RuntimeError(f"container failed: {c}")

        # Wait for Instagram to fetch + process the video
        for _ in range(60):  # up to ~5 min
            s = requests.get(f"https://graph.instagram.com/v19.0/{creation_id}",
                             params={"fields": "status_code,status", "access_token": ig_token},
                             timeout=30).json()
            code = s.get("status_code")
            if code == "FINISHED":
                break
            if code == "ERROR":
                raise RuntimeError(f"IG processing error: {s}")
            time.sleep(5)
        else:
            raise RuntimeError("timed out waiting for Instagram to process the reel")

        p = requests.post("https://graph.instagram.com/v19.0/me/media_publish",
                          params={"creation_id": creation_id, "access_token": ig_token},
                          timeout=60).json()
        published_id = p.get("id")
        if not published_id:
            raise RuntimeError(f"publish failed: {p}")
        return {"status": "posted", "postId": str(published_id),
                "postedAt": datetime.utcnow().isoformat() + "Z"}
    finally:
        _SERVED_FILES.pop(media_id, None)


def _youtube_title_from(caption: str, fallback: str) -> str:
    """YouTube shows a title; IG/TikTok don't. Take the caption's first real line."""
    for line in (caption or "").splitlines():
        cleaned = re.sub(r"#\S+", "", line).strip()
        if cleaned:
            return cleaned[:100]
    return (fallback or "Untitled")[:100]


def _publish_youtube(tmp_path: str, title: str, description: str,
                     privacy: str, google_token: str) -> dict:
    """Resumable upload to YouTube.

    NOTE: until this Google Cloud project passes YouTube's API audit, every upload is
    locked to private by YouTube and CANNOT be made public afterwards — the video has to
    be re-uploaded. That's why the app defaults privacy to 'private' and says so loudly."""
    size = os.path.getsize(tmp_path)
    metadata = {
        "snippet": {"title": title, "description": description or "", "categoryId": "22"},
        "status": {"privacyStatus": privacy or "private", "selfDeclaredMadeForKids": False},
    }
    start = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos",
        params={"uploadType": "resumable", "part": "snippet,status"},
        headers={"Authorization": f"Bearer {google_token}",
                 "Content-Type": "application/json; charset=UTF-8",
                 "X-Upload-Content-Length": str(size),
                 "X-Upload-Content-Type": "video/mp4"},
        json=metadata, timeout=60)
    if start.status_code not in (200, 201):
        raise RuntimeError(f"youtube init failed {start.status_code}: {start.text[:200]}")
    upload_url = start.headers.get("Location")
    if not upload_url:
        raise RuntimeError("youtube did not return an upload url")

    with open(tmp_path, "rb") as f:
        up = requests.put(upload_url, data=f,
                          headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
                          timeout=1800)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"youtube upload failed {up.status_code}: {up.text[:200]}")
    video_id = (up.json() or {}).get("id")
    if not video_id:
        raise RuntimeError(f"youtube upload returned no id: {up.text[:200]}")
    return {"status": "posted", "postId": video_id,
            "url": f"https://youtube.com/watch?v={video_id}",
            "postedAt": datetime.utcnow().isoformat() + "Z"}


def _publish_tiktok(tmp_path: str, caption: str, privacy: str,
                    tiktok_token: str, scope: str) -> dict:
    """Upload to TikTok.

    Two routes, picked by what TikTok has actually granted this app:
      - video.publish (granted only after TikTok's audit) → posts straight to the profile.
      - video.upload  → drops it into the creator's TikTok inbox as a draft to tap publish on.
    Unaudited apps are forced to private viewing regardless, so the draft route is the
    honest default until the audit clears."""
    size = os.path.getsize(tmp_path)
    can_direct_post = "video.publish" in (scope or "")

    # TikTok wants chunks of 5MB-64MB; the final chunk carries any remainder.
    if size <= 64 * 1024 * 1024:
        chunk_size, total_chunks = size, 1
    else:
        chunk_size = 10 * 1024 * 1024
        total_chunks = size // chunk_size

    headers = {"Authorization": f"Bearer {tiktok_token}",
               "Content-Type": "application/json; charset=UTF-8"}
    source_info = {"source": "FILE_UPLOAD", "video_size": size,
                   "chunk_size": chunk_size, "total_chunk_count": total_chunks}

    if can_direct_post:
        init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
        body = {"post_info": {"title": (caption or "")[:2200],
                              "privacy_level": privacy or "SELF_ONLY",
                              "disable_duet": False, "disable_stitch": False,
                              "disable_comment": False},
                "source_info": source_info}
    else:
        init_url = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
        body = {"source_info": source_info}

    init = requests.post(init_url, headers=headers, json=body, timeout=60)
    ij = init.json()
    if ij.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"tiktok init failed: {str(ij.get('error'))[:200]}")
    data = ij.get("data") or {}
    publish_id, upload_url = data.get("publish_id"), data.get("upload_url")
    if not (publish_id and upload_url):
        raise RuntimeError(f"tiktok init returned no upload url: {str(ij)[:200]}")

    # Push the bytes
    with open(tmp_path, "rb") as f:
        for i in range(total_chunks):
            start = i * chunk_size
            end = size - 1 if i == total_chunks - 1 else start + chunk_size - 1
            f.seek(start)
            payload = f.read(end - start + 1)
            up = requests.put(upload_url, data=payload, headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(len(payload)),
                "Content-Range": f"bytes {start}-{end}/{size}",
            }, timeout=1800)
            if up.status_code not in (200, 201, 206, 308):
                raise RuntimeError(f"tiktok chunk {i + 1}/{total_chunks} failed "
                                   f"{up.status_code}: {up.text[:150]}")

    # TikTok processes asynchronously; poll so a failure surfaces in the app, not silently.
    final_state = "PROCESSING"
    for _ in range(40):  # ~3.5 min
        time.sleep(5)
        st = requests.post("https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                           headers=headers, json={"publish_id": publish_id}, timeout=30).json()
        final_state = (st.get("data") or {}).get("status") or final_state
        if final_state in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"):
            break
        if final_state == "FAILED":
            reason = (st.get("data") or {}).get("fail_reason") or st.get("error")
            raise RuntimeError(f"tiktok processing failed: {str(reason)[:200]}")

    note = "" if can_direct_post else "Waiting in your TikTok inbox — open the app and tap publish."
    return {"status": "posted", "postId": str(publish_id), "error": note,
            "postedAt": datetime.utcnow().isoformat() + "Z"}


# ─── The job that posts one scheduled video everywhere it's meant to go ──────

def _run_publish(req: PublishReelRequest):
    def status(**kw):
        if req.post_id:
            firestore_patch(f"scheduledPosts/{req.post_id}",
                            {"updatedAt": datetime.utcnow().isoformat() + "Z", **kw})

    media_id = secrets.token_urlsafe(16)
    tmp_path = os.path.join(tempfile.gettempdir(), f"reel_{media_id}.mp4")
    platforms = [p for p in (req.platforms or ["instagram"]) if p in ("instagram", "youtube", "tiktok")]
    if not platforms:
        platforms = ["instagram"]
    results: dict = {}

    try:
        status(status="posting", error="")

        # 1. Credentials. Prefer what the caller passed (the "Post now" button has a live
        #    browser session); otherwise fetch our own — that's the scheduled path.
        google_token = req.google_access_token or _google_access_token()
        doc_id, client_data = (None, {})
        if req.client_id:
            doc_id, client_data = _client_doc(req.client_id)
        ig_token = req.ig_access_token or client_data.get("instagramAccessToken") or ""
        public_base = req.worker_public_url or WORKER_PUBLIC_URL

        # 2. Download the video from Drive once, then reuse it for every platform.
        with requests.get(f"https://www.googleapis.com/drive/v3/files/{req.drive_file_id}?alt=media",
                          headers={"Authorization": f"Bearer {google_token}"},
                          stream=True, timeout=600) as r:
            if r.status_code != 200:
                raise RuntimeError(f"drive download failed {r.status_code}: {r.text[:150]}")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)

        # 3. Post to each platform independently — one failing must not block the others.
        for plat in platforms:
            try:
                results[plat] = {"status": "posting"}
                status(results=results)
                if plat == "instagram":
                    results[plat] = _publish_instagram(tmp_path, media_id, req.caption or "",
                                                       ig_token, public_base)
                elif plat == "youtube":
                    title = req.youtube_title or _youtube_title_from(req.caption or "", "")
                    results[plat] = _publish_youtube(tmp_path, title, req.caption or "",
                                                     req.youtube_privacy, google_token)
                elif plat == "tiktok":
                    if not doc_id:
                        raise RuntimeError("no connected account found for this post")
                    tt_token, tt_scope = _tiktok_access_token(doc_id, client_data)
                    results[plat] = _publish_tiktok(tmp_path, req.caption or "",
                                                    req.tiktok_privacy, tt_token, tt_scope)
            except Exception as e:
                results[plat] = {"status": "failed", "error": str(e)[:300]}
                print(f"[publish-reel] {plat} failed: {e}")
            status(results=results)

        posted = [p for p in platforms if results.get(p, {}).get("status") == "posted"]
        failed = [p for p in platforms if results.get(p, {}).get("status") == "failed"]
        if posted and not failed:
            overall, err = "posted", ""
        elif posted and failed:
            overall, err = "partial", "Didn't post to: " + ", ".join(failed)
        else:
            overall, err = "failed", "; ".join(
                f"{p}: {results.get(p, {}).get('error', '?')}" for p in failed)[:300]
        status(status=overall, error=err, results=results,
               postedAt=datetime.utcnow().isoformat() + "Z")
        # Keep the old single-platform field alive for anything still reading it.
        ig_id = results.get("instagram", {}).get("postId")
        if ig_id:
            status(igMediaId=str(ig_id))
    except Exception as e:
        # Something before the per-platform loop (credentials, Drive download) broke.
        status(status="failed", error=str(e)[:300], results=results)
        print(f"[publish-reel] error: {e}")
    finally:
        _SERVED_FILES.pop(media_id, None)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/publish-reel")
def publish_reel(req: PublishReelRequest, background_tasks: BackgroundTasks, x_worker_secret: str = Header(default="")):
    if x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    background_tasks.add_task(_run_publish, req)
    return {"started": True}


# ─── The clock ───────────────────────────────────────────────────────────────
# Turns the queue into an actual scheduler: every minute, look for posts whose time
# has come and publish them. Runs in a daemon thread, same as the IG token refresher,
# so there's no external cron to pay for or forget about.

MAX_PUBLISH_ATTEMPTS = 3

# A post whose time passed long ago must NOT quietly go live now. Before the clock existed
# the queue just sat there, so there is a backlog of "scheduled" posts with times in the
# past — switching the clock on would otherwise publish all of them at once. Anything more
# than this many hours overdue is marked missed and waits for a human to hit retry.
MISSED_POST_GRACE_HOURS = 2

# One run at a time. Uploads take minutes, and the manual /run-scheduler trigger must not
# start a second pass over posts the loop thread is already working through.
_SCHEDULER_LOCK = __import__("threading").Lock()


def _due_scheduled_posts() -> list:
    """Posts that are queued and whose time has passed."""
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery"
    body = {"structuredQuery": {
        "from": [{"collectionId": "scheduledPosts"}],
        "where": {"fieldFilter": {"field": {"fieldPath": "status"},
                                  "op": "EQUAL", "value": {"stringValue": "scheduled"}}}}}
    rows = requests.post(url, json=body, timeout=30).json()
    now = datetime.utcnow().isoformat() + "Z"
    due = []
    for row in rows if isinstance(rows, list) else []:
        doc = row.get("document")
        if not doc:
            continue
        data = parse_fs_doc(doc)
        # Compared as ISO strings — both are UTC with a Z suffix, so this sorts correctly
        # and avoids a composite Firestore index just to filter on time.
        if (data.get("scheduledFor") or "") <= now:
            data["_id"] = doc["name"].split("/")[-1]
            due.append(data)
    return due


def _run_scheduler_once() -> dict:
    if not _SCHEDULER_LOCK.acquire(blocking=False):
        return {"due": 0, "fired": 0, "skipped": "already running"}
    try:
        return _scheduler_pass()
    finally:
        _SCHEDULER_LOCK.release()


def _scheduler_pass() -> dict:
    due = _due_scheduled_posts()
    cutoff = (datetime.utcnow() - timedelta(hours=MISSED_POST_GRACE_HOURS)).isoformat() + "Z"
    fired, missed = 0, 0
    for post in due:
        post_id = post["_id"]

        if (post.get("scheduledFor") or "") < cutoff:
            firestore_patch(f"scheduledPosts/{post_id}", {
                "status": "failed",
                "error": "Missed its slot — it was more than "
                         f"{MISSED_POST_GRACE_HOURS}h overdue, so it wasn't posted automatically. "
                         "Hit retry to post it now.",
            })
            missed += 1
            continue

        attempts = int(post.get("attempts") or 0)
        if attempts >= MAX_PUBLISH_ATTEMPTS:
            firestore_patch(f"scheduledPosts/{post_id}", {
                "status": "failed",
                "error": f"Gave up after {attempts} tries — check the connection and retry it.",
            })
            continue
        # Claim it first: flipping the status out of "scheduled" means the next tick
        # won't pick up the same post while this one is still uploading.
        firestore_patch(f"scheduledPosts/{post_id}", {"status": "posting", "attempts": attempts + 1})
        try:
            _run_publish(PublishReelRequest(
                post_id=post_id,
                drive_file_id=post.get("driveFileId", ""),
                caption=post.get("caption", ""),
                client_id=post.get("clientId"),
                platforms=post.get("platforms") or ["instagram"],
                youtube_title=post.get("youtubeTitle"),
                youtube_privacy=post.get("youtubePrivacy") or "private",
                tiktok_privacy=post.get("tiktokPrivacy") or "SELF_ONLY",
            ))
            fired += 1
        except Exception as e:
            firestore_patch(f"scheduledPosts/{post_id}", {"status": "failed", "error": str(e)[:300]})
            print(f"[scheduler] {post_id} failed: {e}")
    return {"due": len(due), "fired": fired, "missed": missed}


@app.post("/run-scheduler")
def run_scheduler(x_worker_secret: str = Header(default="")):
    """Manual trigger for the clock (it also runs by itself every minute)."""
    if x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    return _run_scheduler_once()


def _scheduler_loop():
    time.sleep(45)  # let the app finish booting
    while True:
        try:
            summary = _run_scheduler_once()
            if summary["due"]:
                print(f"[scheduler] {summary['fired']}/{summary['due']} due posts fired")
        except Exception as e:
            print(f"[scheduler] loop error: {e}")
        time.sleep(60)


@app.on_event("startup")
def _start_scheduler():
    if not SCHEDULER_ENABLED:
        print("[scheduler] disabled via SCHEDULER_ENABLED=0")
        return
    import threading
    threading.Thread(target=_scheduler_loop, daemon=True).start()


class BuildReelClip(BaseModel):
    name: Optional[str] = ""
    driveFileId: Optional[str] = None
    driveUrl: Optional[str] = None
    bunnyUrl: Optional[str] = None
    downloadUrl: Optional[str] = None

class BuildReelRequest(BaseModel):
    client_id: str
    client_name: Optional[str] = ""
    title: Optional[str] = "AI rough cut"
    source_url: Optional[str] = ""       # the original reel being modelled
    google_access_token: Optional[str] = None
    root_folder_id: Optional[str] = None # client Drive root — output goes in a subfolder
    clips: list[BuildReelClip] = []
    clip_seconds: float = 2.0            # how long to keep from each clip


def _clip_download_url(clip: BuildReelClip, token: Optional[str]) -> Optional[str]:
    url = clip.bunnyUrl or clip.driveUrl or clip.downloadUrl
    if not url and clip.driveFileId and token:
        url = f"https://www.googleapis.com/drive/v3/files/{clip.driveFileId}?alt=media"
    return url


def _drive_upload(name: str, parent_id: str, filepath: str, token: str, mime: str = "video/mp4") -> dict:
    """Multipart-upload a local file into a Drive folder. Returns {id, webViewLink}."""
    with open(filepath, "rb") as f:
        data = f.read()
    boundary = "cdsreelboundary"
    meta = json.dumps({"name": name, "parents": [parent_id]})
    body = (
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
        f"--{boundary}\r\nContent-Type: {mime}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--".encode()
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files",
        params={"uploadType": "multipart", "fields": "id,webViewLink,webContentLink"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/related; boundary={boundary}"},
        data=body, timeout=300,
    )
    return r.json()


def _run_build_reel(req: BuildReelRequest):
    """Download the chosen clips, trim + normalise each, concat into a silent 9:16
    rough cut, upload it to the client's Drive, and drop a draft into the
    Production Queue (reels). Music + captions are added 1:1 by the editor after."""
    cid = req.client_id
    total = len(req.clips)

    def status(**extra):
        firestore_patch(f"buildStatus/{cid}", {
            "running": extra.get("running", True),
            "total": total,
            "done": extra.get("done", 0),
            "stage": extra.get("stage", ""),
            "error": extra.get("error", ""),
            "videoUrl": extra.get("videoUrl", ""),
            "updatedAt": datetime.utcnow().isoformat(),
        })

    if total == 0:
        status(running=False, error="No clips selected")
        return

    status(stage="starting")
    workdir = tempfile.mkdtemp(prefix="reel_")
    segments = []
    last_err = ""
    try:
        for i, clip in enumerate(req.clips):
            url = _clip_download_url(clip, req.google_access_token)
            if not url:
                last_err = f"{clip.name}: no download URL (driveFileId={clip.driveFileId}, token={bool(req.google_access_token)})"
                continue
            headers = {}
            if "googleapis.com" in url and req.google_access_token:
                headers["Authorization"] = f"Bearer {req.google_access_token}"
            raw = os.path.join(workdir, f"raw_{i}.mp4")
            seg = os.path.join(workdir, f"seg_{i}.mp4")
            status(stage=f"downloading clip {i+1}/{total}", done=i, error=last_err)
            try:
                with requests.get(url, stream=True, timeout=300, headers=headers) as resp:
                    if resp.status_code != 200:
                        last_err = f"{clip.name}: download HTTP {resp.status_code} — {resp.text[:150]}"
                        print(f"[build-reel] {last_err}")
                        continue
                    with open(raw, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            except Exception as e:
                last_err = f"{clip.name}: download error {e}"
                print(f"[build-reel] {last_err}")
                continue
            size = os.path.getsize(raw) if os.path.exists(raw) else 0
            if size < 1000:
                last_err = f"{clip.name}: downloaded only {size} bytes (likely not a video)"
                print(f"[build-reel] {last_err}")
                continue
            # Trim first N seconds, crop-fill to vertical 1080x1920, 30fps, silent, uniform codec.
            status(stage=f"trimming clip {i+1}/{total}", done=i, error=last_err)
            vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1"
            proc = subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-i", raw, "-t", str(req.clip_seconds),
                 "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", seg],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
            )
            if os.path.exists(seg) and os.path.getsize(seg) > 0:
                segments.append(seg)
            else:
                last_err = f"{clip.name}: ffmpeg trim failed — {proc.stderr[-200:]}"
                print(f"[build-reel] {last_err}")

        if not segments:
            status(running=False, error=f"Could not process any clips. Last: {last_err}")
            return

        # Concat the uniform segments (same codec params → stream copy is safe).
        status(stage="stitching")
        listfile = os.path.join(workdir, "list.txt")
        with open(listfile, "w") as fh:
            for s in segments:
                fh.write(f"file '{s}'\n")
        out = os.path.join(workdir, "rough_cut.mp4")
        cat = subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", "-movflags", "+faststart", out],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            # Fallback: re-encode concat if stream copy failed.
            cat = subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", listfile,
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
            )
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            status(running=False, error=f"Stitch failed: {cat.stderr[-300:]}")
            return

        # Upload the rough cut to Drive and register it in the Production Queue.
        video_url = ""
        if req.google_access_token and req.root_folder_id:
            status(stage="uploading")
            cache = {}
            folder_id = _resolve_path("🎬 AI Rough Cuts", req.root_folder_id, req.google_access_token, cache)
            fname = f"{(req.title or 'rough_cut').replace('/', '-')} {datetime.utcnow().strftime('%Y-%m-%d %H%M')}.mp4"
            up = _drive_upload(fname, folder_id, out, req.google_access_token)
            video_url = up.get("webViewLink", "")

        import uuid
        reel_id = uuid.uuid4().hex[:20]
        firestore_create("reels", reel_id, {
            "clientId": cid,
            "clientName": req.client_name or "",
            "title": req.title or "AI rough cut",
            "caption": f"Modelled from {req.source_url}" if req.source_url else "",
            "videoUrl": video_url,
            "status": "pending",
            "source": "ai-rough-cut",
            "createdAt": datetime.utcnow().isoformat(),
        })
        status(running=False, done=total, stage="done", videoUrl=video_url)
    except Exception as e:
        print(f"[build-reel] fatal: {e}")
        status(running=False, error=str(e)[:300])
    finally:
        try:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


@app.post("/build-reel")
def build_reel(req: BuildReelRequest, background_tasks: BackgroundTasks):
    """Assemble a silent 9:16 rough cut from the chosen clips. Runs in background;
    frontend polls buildStatus/{client_id}. Editor adds 1:1 music + captions after."""
    background_tasks.add_task(_run_build_reel, req)
    return {"started": True, "clips": len(req.clips)}

class AnalyseIGPostRequest(BaseModel):
    video_url: str
    caption: Optional[str] = ""
    post_id: Optional[str] = ""
    client_id: Optional[str] = ""
    niche: Optional[str] = ""

@app.post("/analyse-ig-post")
def analyse_ig_post(req: AnalyseIGPostRequest):
    """Analyse an Instagram post video with Claude Vision."""
    try:
        frames = extract_frames(req.video_url, num_frames=3)
        if not frames:
            return {"success": False, "error": "Could not extract frames", "post_id": req.post_id}

        analysis = analyse_video_with_claude(frames, req.caption, req.niche)
        return {"success": True, "post_id": req.post_id, "analysis": analysis, "frames_extracted": len(frames)}

    except Exception as e:
        return {"success": False, "error": str(e), "post_id": req.post_id}


# ─── Podcast Engine: Hebrew transcription (Whisper via Groq) ──────────────────

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")

class TranscribePodcastRequest(BaseModel):
    client_id: str
    episode_id: str                       # our own id for this episode (used as doc id)
    google_access_token: str
    drive_file_id: Optional[str] = None
    drive_file_name: Optional[str] = None # fallback: find by name (searches whole Drive)
    episode_title: Optional[str] = ""


def _drive_find_file_by_name(name: str, token: str) -> Optional[str]:
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    q = f"name contains '{safe}' and trashed = false"
    r = requests.get("https://www.googleapis.com/drive/v3/files",
                     params={"q": q, "fields": "files(id,name)", "spaces": "drive"},
                     headers=_drive_headers(token))
    files = r.json().get("files", [])
    return files[0]["id"] if files else None


def _deepgram_transcribe(audio_path: str) -> list:
    """Send the whole audio file to Deepgram (Hebrew + speaker diarization) in one call —
    no manual chunking needed, Deepgram handles long audio directly. Returns a list of
    {start, end, text, speaker} segments, one per speaker utterance."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    r = requests.post(
        "https://api.deepgram.com/v1/listen",
        params={
            "model": "nova-3", "language": "he", "diarize": "true", "punctuate": "true",
            "utterances": "true", "smart_format": "true",
        },
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/mpeg"},
        data=audio_bytes, timeout=1800,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Deepgram transcription failed: {r.status_code} {r.text[:300]}")
    data = r.json()
    utterances = data.get("results", {}).get("utterances", [])
    segments = []
    for u in utterances:
        text = (u.get("transcript") or "").strip()
        if not text:
            continue
        segments.append({
            "start": round(u.get("start", 0), 1),
            "end": round(u.get("end", 0), 1),
            "text": text,
            "speaker": f"Speaker {u.get('speaker', 0) + 1}",
        })
    return segments


def _run_transcribe_podcast(req: TranscribePodcastRequest):
    """Download the episode, strip audio, transcribe via Deepgram (Hebrew + speaker labels
    in one pass), save the timestamped transcript to Firestore. Runs in background; frontend
    polls transcribeStatus/{episode_id}."""
    eid = req.episode_id

    def status(**extra):
        firestore_patch(f"transcribeStatus/{eid}", {
            "running": extra.get("running", True),
            "stage": extra.get("stage", ""),
            "error": extra.get("error", ""),
            "updatedAt": datetime.utcnow().isoformat(),
        })

    if not DEEPGRAM_API_KEY:
        status(running=False, error="DEEPGRAM_API_KEY not set on the worker")
        return

    status(stage="resolving file")
    file_id = req.drive_file_id
    if not file_id and req.drive_file_name:
        file_id = _drive_find_file_by_name(req.drive_file_name, req.google_access_token)
    if not file_id:
        status(running=False, error="Could not find the Drive file (no id, name not found)")
        return

    workdir = tempfile.mkdtemp(prefix="podcast_")
    try:
        # Download the raw video from Drive.
        status(stage="downloading episode")
        video_path = os.path.join(workdir, "episode.mp4")
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {req.google_access_token}"}
        with requests.get(url, stream=True, headers=headers, timeout=600) as resp:
            if resp.status_code != 200:
                status(running=False, error=f"Drive download failed: HTTP {resp.status_code} — {resp.text[:200]}")
                return
            with open(video_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

        # Strip audio only, mono — Deepgram takes the whole file in one call, no chunking needed.
        status(stage="extracting audio")
        audio_path = os.path.join(workdir, "audio.mp3")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn",
             "-ac", "1", "-ar", "16000", "-b:a", "64k", audio_path],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        if not (os.path.exists(audio_path) and os.path.getsize(audio_path) > 0):
            status(running=False, error=f"Audio extraction failed: {proc.stderr[-300:]}")
            return

        status(stage="transcribing with Deepgram (this can take a few minutes)")
        try:
            all_segments = _deepgram_transcribe(audio_path)
        except Exception as e:
            status(running=False, error=f"Deepgram transcription failed: {e}")
            return

        if not all_segments:
            status(running=False, error="Transcription produced no segments")
            return

        full_text = " ".join(s["text"] for s in all_segments)
        firestore_patch(f"podcastTranscripts/{eid}", {
            "clientId": req.client_id,
            "episodeTitle": req.episode_title or "",
            "driveFileId": file_id,
            "segments": json.dumps(all_segments, ensure_ascii=False),
            "fullText": full_text,
            "segmentCount": len(all_segments),
            "transcribedAt": datetime.utcnow().isoformat(),
        })
        status(running=False, stage="done")
    except Exception as e:
        print(f"[transcribe-podcast] fatal: {e}")
        status(running=False, error=str(e)[:300])
    finally:
        try:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


@app.post("/transcribe-podcast")
def transcribe_podcast(req: TranscribePodcastRequest, background_tasks: BackgroundTasks):
    """Kick off background transcription of a podcast episode (Hebrew + speaker diarization,
    via Deepgram). Frontend polls transcribeStatus/{episode_id}; result lands in
    podcastTranscripts/{episode_id}."""
    background_tasks.add_task(_run_transcribe_podcast, req)
    return {"started": True, "episode_id": req.episode_id}


# ─── Podcast Engine: triage (Claude decides gold / keep / cut) ────────────────

TRIAGE_PROMPT = """You are triaging a Hebrew business/sales podcast episode hosted by Nimrod Avdala
("הפרלמנט של נמרוד"), CEO of Nova (outsourced sales). The goal: find the moments worth turning into
short Instagram reels, and flag what's just filler.

"Gold" for this show means: a strong sales/business insight, a real personal story, a punchy quotable
one-liner, or an emotional/authentic beat — something that could stand alone as a 30-60s reel.
"Cut" means: filler words, rambling tangents, dead air, small talk unrelated to the topic, repeated points.
"Keep" is everything solid but not standout — useful for a tighter full episode, not reel-worthy alone.

Here is ONE WINDOW of the episode's timestamped transcript (seconds) — each line is tagged with which
speaker said it ("Speaker 1" / "Speaker 2" — these are just voice labels, not roles; the same person can
be Speaker 1 in one episode and Speaker 2 in another). Analyse ONLY this window closely and thoroughly,
exactly like you would a short 7-10 minute clip. Do not worry about the rest of the episode; other windows
are analysed separately.

{transcript}

Return ONLY a JSON object with this exact shape:
{{
  "gold": [
    {{"start": <seconds>, "end": <seconds>, "speaker": "<Speaker 1 or Speaker 2 — whoever is mainly talking in this moment>", "why": "<one line in Hebrew or English>", "quote": "<the Hebrew quote>", "rank": <1 = best within this window>}}
  ],
  "keep": [
    {{"start": <seconds>, "end": <seconds>, "speaker": "<Speaker 1 or Speaker 2>", "why": "<one line>", "quote": "<a short representative Hebrew quote from this segment>"}}
  ],
  "cut": [
    {{"start": <seconds>, "end": <seconds>, "speaker": "<Speaker 1 or Speaker 2>", "why": "<one line>", "quote": "<a short representative Hebrew quote from this segment>"}}
  ]
}}

CRITICAL — read this window CLOSELY, line by line. Do not skim. Find EVERY gold moment in this window,
even a short one — a punchy one-liner is gold even if it's only 15 seconds. Do not just grab one "best"
moment per window and move on; a rich window can have 2-4 gold moments.

For GOLD: each is a self-contained "nugget" — starts on the hook (the punchy line), ends where that
specific idea/story pays off. Bias tight; most are 20-90 seconds; only a full pure-gold story goes up to
~150 seconds max. Do NOT extend a clip to a whole subject. Rank gold best-first WITHIN this window.

For KEEP: solid supporting context, a representative sample (doesn't need to be exhaustive).

For CUT: give only 2-5 REPRESENTATIVE examples of filler/rambling/silence in this window — do NOT try to
list every non-gold second. The rest of this window is implicitly cut.

Merge adjacent segments that belong to the same moment into one range."""


class TriagePodcastRequest(BaseModel):
    episode_id: str


def _fmt_transcript_for_claude(segments: list) -> str:
    lines = [f"[{s['start']}-{s['end']}] ({s.get('speaker', 'Speaker 1')}) {s['text']}" for s in segments]
    return "\n".join(lines)


def _chunk_segments(segments: list, window_seconds: float = 600, overlap_seconds: float = 30) -> list:
    """Split segments into ~10-min windows (with a little overlap so a moment spanning
    a window boundary isn't missed) for focused, per-window triage."""
    if not segments:
        return []
    total_end = segments[-1]["end"]
    windows = []
    start = 0.0
    while start < total_end:
        end = start + window_seconds
        windows.append([s for s in segments if s["end"] > start and s["start"] < end])
        start = end - overlap_seconds
    return [w for w in windows if w]


def _triage_one_window(segments: list) -> dict:
    transcript_text = _fmt_transcript_for_claude(segments)
    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": TRIAGE_PROMPT.format(transcript=transcript_text)}],
    )
    raw = msg.content[0].text if msg.content and msg.content[0].type == "text" else "{}"
    m = re.search(r"\{[\s\S]*\}", raw)
    return json.loads(m.group(0)) if m else {"gold": [], "keep": [], "cut": []}


def _dedupe_by_overlap(items: list) -> list:
    """Windows overlap by 30s, so the same moment can be caught twice. Drop items whose
    [start,end] range overlaps a previously-kept item's range by more than half."""
    kept = []
    for item in sorted(items, key=lambda x: x["start"]):
        dup = False
        for k in kept:
            overlap = min(item["end"], k["end"]) - max(item["start"], k["start"])
            shorter = min(item["end"] - item["start"], k["end"] - k["start"])
            if shorter > 0 and overlap / shorter > 0.5:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


def _run_triage_podcast(episode_id: str):
    """Chunk the transcript into ~10-min windows, triage each window closely (same focused
    approach that works well on a short clip), merge + dedupe + re-rank gold. Runs in
    background; frontend polls triageStatus/{episode_id}."""

    def status(**extra):
        firestore_patch(f"triageStatus/{episode_id}", {
            "running": extra.get("running", True),
            "windowsDone": extra.get("windowsDone", 0),
            "windowsTotal": extra.get("windowsTotal", 0),
            "error": extra.get("error", ""),
            "updatedAt": datetime.utcnow().isoformat(),
        })

    status(stage="starting")
    doc = firestore_get(f"podcastTranscripts/{episode_id}")
    data = parse_fs_doc(doc)
    if not data or "segments" not in data:
        status(running=False, error="No transcript found for this episode_id")
        return

    segments = json.loads(data["segments"])
    windows = _chunk_segments(segments)
    total = len(windows)
    status(windowsTotal=total)

    all_gold, all_keep, all_cut = [], [], []
    for i, window in enumerate(windows):
        status(windowsDone=i, windowsTotal=total)
        try:
            result = _triage_one_window(window)
            all_gold.extend(result.get("gold", []))
            all_keep.extend(result.get("keep", []))
            all_cut.extend(result.get("cut", []))
        except Exception as e:
            print(f"[triage-podcast] window {i+1}/{total} failed: {e}")

    gold = _dedupe_by_overlap(all_gold)
    keep = _dedupe_by_overlap(all_keep)
    cut = _dedupe_by_overlap(all_cut)
    # Re-rank gold globally, best-first, now that all windows are merged.
    gold.sort(key=lambda x: x.get("rank", 99))
    for idx, g in enumerate(gold):
        g["rank"] = idx + 1

    firestore_patch(f"podcastTriage/{episode_id}", {
        "gold": json.dumps(gold, ensure_ascii=False),
        "keep": json.dumps(keep, ensure_ascii=False),
        "cut": json.dumps(cut, ensure_ascii=False),
        "triagedAt": datetime.utcnow().isoformat(),
    })
    status(running=False, windowsDone=total, windowsTotal=total)


@app.post("/triage-podcast")
def triage_podcast(req: TriagePodcastRequest, background_tasks: BackgroundTasks):
    """Kick off background chunked triage (one focused pass per ~10-min window, merged).
    Frontend polls triageStatus/{episode_id}; result lands in podcastTriage/{episode_id}."""
    background_tasks.add_task(_run_triage_podcast, req.episode_id)
    return {"started": True, "episode_id": req.episode_id}
