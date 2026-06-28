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

        # ── 2a: fast seek-based extraction (input seek) — works for most files ──
        timestamps = [0.5, min(2.0, duration * 0.1), duration * 0.5, duration * 0.85]
        last_ffmpeg_err = ""
        for i, ts in enumerate(timestamps[:num_frames]):
            frame_path = os.path.join(tmp, f"frame_{i}.jpg")
            proc = subprocess.run([
                "ffmpeg", "-ss", str(ts), "-i", video_path,
                "-vframes", "1", "-q:v", "3", "-vf", "scale=512:-1",
                "-an", frame_path, "-y"
            ], capture_output=True, text=True)
            if proc.returncode != 0:
                last_ffmpeg_err = (proc.stderr or "")[-150:]
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())

        # ── 2b: robust fallback — single sequential decode pass. Handles iPhone
        # Dolby Vision / HEVC HDR clips that the fast seek method can't grab. ──
        if not frames:
            interval = max(0.5, duration / (num_frames + 1))
            out_pattern = os.path.join(tmp, "seq_%02d.jpg")
            proc = subprocess.run([
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{interval:.3f},scale=512:-1",
                "-frames:v", str(num_frames), "-q:v", "3", "-an",
                out_pattern, "-y"
            ], capture_output=True, text=True)
            if proc.returncode != 0:
                last_ffmpeg_err = (proc.stderr or "")[-150:]
            for i in range(1, num_frames + 1):
                fp = os.path.join(tmp, f"seq_{i:02d}.jpg")
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        frames.append(base64.b64encode(f.read()).decode())

        if not frames:
            raise RuntimeError(f"FFMPEG_FAILED no frames: {last_ffmpeg_err or 'unknown'}")

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
    old = ",".join(g.json().get("parents", []))
    r = requests.patch(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                       params={"addParents": add_parent, "removeParents": old, "fields": "id,parents"},
                       headers=_drive_headers(token))
    if r.status_code != 200:
        raise RuntimeError(f"move failed http {r.status_code}: {r.text[:150]}")

class PushRequest(BaseModel):
    client_id: str
    google_access_token: str
    root_folder_id: str
    moves: list  # [{"drive_file_id":..., "target_path":..., "name":...}]

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


class BuildReelRequest(BaseModel):
    client_id: str
    brief: str  # e.g. "high energy fitness reel, talking hook + 3 action b-rolls"
    clip_ids: Optional[list[str]] = None  # specific clips, or auto-select

@app.post("/build-reel")
def build_reel(req: BuildReelRequest):
    """Select clips based on brief and assemble a reel with FFmpeg."""
    # TODO: implement full reel building
    # For now returns the plan
    return {
        "status": "coming_soon",
        "message": "Reel builder agent is being built",
        "brief": req.brief
    }

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
