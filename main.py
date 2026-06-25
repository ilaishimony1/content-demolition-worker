import os
import re
import json
import base64
import subprocess
import tempfile
import requests
import anthropic
from fastapi import FastAPI, HTTPException
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

def extract_frames(video_url: str, num_frames: int = 4, extra_headers: dict = {}) -> list[str]:
    """Download video and extract frames as base64 images using FFmpeg."""
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")

        # Download video (with optional auth headers for Google Drive)
        r = requests.get(video_url, stream=True, timeout=60, headers=extra_headers)
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Get video duration
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

        # Extract frames: hook (0.5s), early (2s), mid, near-end
        timestamps = [0.5, min(2.0, duration * 0.1), duration * 0.5, duration * 0.85]

        for i, ts in enumerate(timestamps[:num_frames]):
            frame_path = os.path.join(tmp, f"frame_{i}.jpg")
            subprocess.run([
                "ffmpeg", "-ss", str(ts), "-i", video_path,
                "-vframes", "1", "-q:v", "3", "-vf", "scale=720:-1",
                frame_path, "-y"
            ], capture_output=True)

            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())

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

Analyse this video content and respond ONLY with valid JSON:
{{
  "content_type": "talking_reel|action_reel|broll|carousel|transformation|tutorial|vlog",
  "has_face": true/false,
  "is_talking_to_camera": true/false,
  "energy_level": "low|medium|high|explosive",
  "hook_type": "question|statement|action|curiosity|transformation|story",
  "hook_quality": 1-10,
  "topic": "brief topic description",
  "setting": "gym|outdoor|home|studio|street|other",
  "has_text_overlay": true/false,
  "usability_score": 1-10,
  "notes": "1 sentence about what makes this content work or not"
}}"""
    })

    message = claude.messages.create(
        model="claude-opus-4-5",
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
    return {"status": "running", "service": "Content Demolition Worker", "agents": ["video-analyser", "drive-scanner", "reel-builder", "auto-poster"]}

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

@app.post("/scan-drive")
def scan_drive(req: ScanDriveRequest):
    """Scan a client's Firestore clips and analyse any untagged ones."""
    try:
        # Get client info
        result = firestore_query("users", "clientId", req.client_id)
        doc = next((r.get("document") for r in result if r.get("document")), None)
        if not doc:
            raise HTTPException(status_code=404, detail="Client not found")

        client = parse_fs_doc(doc)
        niche = client.get("niche", "")

        # Fetch a wide candidate set for this client, then filter in Python
        # (Firestore can't query "field does not exist" or string prefixes).
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery"
        body = {
            "structuredQuery": {
                "from": [{"collectionId": "clips"}],
                "where": {
                    "fieldFilter": {"field": {"fieldPath": "clientId"}, "op": "EQUAL", "value": {"stringValue": req.client_id}}
                },
                "limit": 1000
            }
        }
        r = requests.post(url, json=body)
        clips_data = r.json()

        analysed = []
        errors = []
        skipped = []
        batch_size = req.batch_size or 50

        for item in clips_data:
            if len(analysed) >= batch_size:
                break  # stop once we've analysed a full batch
            doc = item.get("document")
            if not doc:
                continue
            clip = parse_fs_doc(doc)
            path = clip.get("path", "")
            # Skip clips already analysed
            if clip.get("aiAnalysedAt"):
                continue
            # Skip clips in protected (frozen/additive) folders — agent won't move them.
            if req.protected_folders and _is_protected(path, req.protected_folders):
                continue
            # If a folder filter is set, only scan clips in that folder (or its subfolders)
            if req.folder_filter and not (path == req.folder_filter or path.startswith(req.folder_filter + "/")):
                continue
            clip_id = doc["name"].split("/")[-1]

            # Build video URL — prefer Bunny CDN (no auth needed), fall back to Drive API
            video_url = clip.get("bunnyUrl") or clip.get("driveUrl") or clip.get("downloadUrl")
            drive_file_id = clip.get("driveFileId")

            # If no direct URL but we have a Drive file ID + token, use Drive API
            if not video_url and drive_file_id and req.google_access_token:
                video_url = f"https://www.googleapis.com/drive/v3/files/{drive_file_id}?alt=media"

            if not video_url:
                skipped.append({"clip_id": clip_id, "reason": "No video URL", "has_token": bool(req.google_access_token), "drive_file_id": drive_file_id})
                continue

            # Build headers for download (Drive API needs auth)
            download_headers = {}
            if "googleapis.com" in video_url and req.google_access_token:
                download_headers["Authorization"] = f"Bearer {req.google_access_token}"

            try:
                frames = extract_frames(video_url, num_frames=4, extra_headers=download_headers)
                if frames:
                    analysis = analyse_video_with_claude(frames, clip.get("caption", ""), niche, req.taxonomy)
                    firestore_patch(f"clips/{clip_id}", {
                        "aiContentType": analysis.get("content_type", "unknown"),
                        "aiHasFace": str(analysis.get("has_face", False)),
                        "aiIsTalking": str(analysis.get("is_talking_to_camera", False)),
                        "aiEnergyLevel": analysis.get("energy_level", "medium"),
                        "aiHookQuality": str(analysis.get("hook_quality", 5)),
                        "aiTopic": analysis.get("topic", ""),
                        "aiUsabilityScore": str(analysis.get("usability_score", 5)),
                        "aiNotes": analysis.get("notes", ""),
                        "aiAnalysedAt": datetime.utcnow().isoformat(),
                    })
                    analysed.append({"clip_id": clip_id, "analysis": analysis})
            except Exception as e:
                errors.append({"clip_id": clip_id, "error": str(e)})

        return {
            "success": True,
            "analysed": len(analysed),
            "errors": len(errors),
            "skipped": len(skipped),
            "skip_reasons": skipped[:5],  # first 5 for debugging
            "has_token": bool(req.google_access_token),
            "results": analysed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        frames = extract_frames(req.video_url, num_frames=4)
        if not frames:
            return {"success": False, "error": "Could not extract frames", "post_id": req.post_id}

        analysis = analyse_video_with_claude(frames, req.caption, req.niche)
        return {"success": True, "post_id": req.post_id, "analysis": analysis, "frames_extracted": len(frames)}

    except Exception as e:
        return {"success": False, "error": str(e), "post_id": req.post_id}
