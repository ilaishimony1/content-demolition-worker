import os
import re
import json
import base64
import subprocess
import tempfile
import requests
import anthropic
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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

def firestore_patch(path: str, fields: dict):
    url_fields = "&".join([f"updateMask.fieldPaths={k}" for k in fields])
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{path}?{url_fields}"
    fs_fields = {}
    for k, v in fields.items():
        if v is None:
            fs_fields[k] = {"nullValue": None}
        elif isinstance(v, bool):
            fs_fields[k] = {"booleanValue": v}
        elif isinstance(v, int) or isinstance(v, float):
            fs_fields[k] = {"doubleValue": float(v)}
        elif isinstance(v, list):
            fs_fields[k] = {"arrayValue": {"values": [{"stringValue": str(item)} for item in v]}}
        elif isinstance(v, dict):
            fs_fields[k] = {"mapValue": {"fields": {kk: {"stringValue": str(vv)} for kk, vv in v.items()}}}
        else:
            fs_fields[k] = {"stringValue": str(v)}
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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


def _groq_transcribe_chunk(filepath: str) -> dict:
    """Send one audio chunk to Groq's Whisper endpoint (OpenAI-compatible). Returns verbose_json."""
    with open(filepath, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (os.path.basename(filepath), f, "audio/mpeg")},
            data={"model": "whisper-large-v3", "language": "he", "response_format": "verbose_json"},
            timeout=300,
        )
    if r.status_code != 200:
        raise RuntimeError(f"Groq transcription failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _run_transcribe_podcast(req: TranscribePodcastRequest):
    """Download the episode, strip + chunk the audio, transcribe each chunk via Groq Whisper,
    stitch into one timestamped Hebrew transcript, save to Firestore. Runs in background;
    frontend polls transcribeStatus/{episode_id}."""
    eid = req.episode_id

    def status(**extra):
        firestore_patch(f"transcribeStatus/{eid}", {
            "running": extra.get("running", True),
            "stage": extra.get("stage", ""),
            "error": extra.get("error", ""),
            "chunksDone": extra.get("chunksDone", 0),
            "chunksTotal": extra.get("chunksTotal", 0),
            "updatedAt": datetime.utcnow().isoformat(),
        })

    if not GROQ_API_KEY:
        status(running=False, error="GROQ_API_KEY not set on the worker")
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

        # Strip audio only, mono, low bitrate — keeps chunks well under the 25MB API cap.
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

        # Split into ~10-min chunks (well under Groq's 25MB/file limit at 64kbps mono).
        status(stage="chunking audio")
        chunk_pattern = os.path.join(workdir, "chunk_%03d.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-i", audio_path,
             "-f", "segment", "-segment_time", "600", "-c", "copy", chunk_pattern],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        chunks = sorted(f for f in os.listdir(workdir) if f.startswith("chunk_"))
        if not chunks:
            status(running=False, error="No audio chunks produced")
            return

        # Transcribe each chunk, offsetting segment timestamps by the chunk's start time.
        all_segments = []
        for i, chunk_name in enumerate(chunks):
            status(stage=f"transcribing chunk {i+1}/{len(chunks)}", chunksDone=i, chunksTotal=len(chunks))
            chunk_offset = i * 600  # seconds
            try:
                result = _groq_transcribe_chunk(os.path.join(workdir, chunk_name))
                for seg in result.get("segments", []):
                    all_segments.append({
                        "start": round(seg["start"] + chunk_offset, 1),
                        "end": round(seg["end"] + chunk_offset, 1),
                        "text": seg["text"].strip(),
                    })
            except Exception as e:
                print(f"[transcribe-podcast] chunk {chunk_name} failed: {e}")
                status(stage=f"chunk {i+1} failed, continuing", chunksDone=i, chunksTotal=len(chunks),
                       error=f"chunk {i+1}: {e}")

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
        status(running=False, stage="done", chunksDone=len(chunks), chunksTotal=len(chunks))
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
    """Kick off background transcription of a podcast episode (Hebrew, via Groq Whisper).
    Frontend polls transcribeStatus/{episode_id}; result lands in podcastTranscripts/{episode_id}."""
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

Here is the timestamped transcript (seconds):
{transcript}

Return ONLY a JSON object with this exact shape:
{{
  "gold": [
    {{"start": <seconds>, "end": <seconds>, "why": "<one line in Hebrew or English>", "quote": "<the Hebrew quote>", "rank": <1 = best>}}
  ],
  "keep": [
    {{"start": <seconds>, "end": <seconds>, "why": "<one line>", "quote": "<a short representative Hebrew quote from this segment>"}}
  ],
  "cut": [
    {{"start": <seconds>, "end": <seconds>, "why": "<one line>", "quote": "<a short representative Hebrew quote from this segment>"}}
  ]
}}

Cover the full episode — every segment should fall into exactly one of gold/keep/cut. Aim for 5-10 gold
moments if the content supports it; don't force it if the episode is weak. Merge adjacent segments that
belong to the same moment into one range."""


class TriagePodcastRequest(BaseModel):
    episode_id: str


def _fmt_transcript_for_claude(segments: list) -> str:
    lines = [f"[{s['start']}-{s['end']}] {s['text']}" for s in segments]
    return "\n".join(lines)


@app.post("/triage-podcast")
def triage_podcast(req: TriagePodcastRequest):
    """Read the saved transcript and ask Claude to produce the gold/keep/cut triage map.
    Synchronous (transcript analysis is a single fast call) — saves to podcastTriage/{episode_id}."""
    doc = firestore_get(f"podcastTranscripts/{req.episode_id}")
    data = parse_fs_doc(doc)
    if not data or "segments" not in data:
        raise HTTPException(status_code=404, detail="No transcript found for this episode_id")

    segments = json.loads(data["segments"])
    transcript_text = _fmt_transcript_for_claude(segments)

    try:
        msg = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{"role": "user", "content": TRIAGE_PROMPT.format(transcript=transcript_text)}],
        )
        raw = msg.content[0].text if msg.content and msg.content[0].type == "text" else "{}"
        m = re.search(r"\{[\s\S]*\}", raw)
        triage = json.loads(m.group(0)) if m else {"gold": [], "keep": [], "cut": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Triage failed: {e}")

    firestore_patch(f"podcastTriage/{req.episode_id}", {
        "gold": json.dumps(triage.get("gold", []), ensure_ascii=False),
        "keep": json.dumps(triage.get("keep", []), ensure_ascii=False),
        "cut": json.dumps(triage.get("cut", []), ensure_ascii=False),
        "triagedAt": datetime.utcnow().isoformat(),
    })
    return {"success": True, "episode_id": req.episode_id, "triage": triage}
