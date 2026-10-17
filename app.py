
import os
import uuid
import threading
import requests

from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, Response
from dotenv import load_dotenv
from gradio_client import Client

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

music_lock = threading.Lock()
video_lock = threading.Lock()


def get_client(space):
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return Client(space, token=token)
    return Client(space)


def save_remote_file(source, prefix):
    if isinstance(source, (tuple, list)):
        if not source:
            raise ValueError("Hakuna faili lililorudishwa.")
        source = source[0]

    if isinstance(source, dict):
        source = (
            source.get("url")
            or source.get("path")
            or source.get("name")
            or source.get("audio")
            or source.get("video")
        )

    if not isinstance(source, (str, os.PathLike)):
        raise ValueError("Faili lililorudishwa halitambuliki.")

    source_text = os.fspath(source)

    if source_text.startswith(("http://", "https://")):
        response = requests.get(source_text, timeout=240)
        response.raise_for_status()
        content = response.content
        suffix = Path(source_text.split("?")[0]).suffix
    else:
        source_path = Path(source_text)
        if not source_path.is_file():
            raise FileNotFoundError("Faili halijapatikana.")
        content = source_path.read_bytes()
        suffix = source_path.suffix

    if not suffix:
        suffix = ".mp4" if prefix == "video" else ".wav"

    filename = f"freemaxy_{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    (OUTPUT_DIR / filename).write_bytes(content)
    return filename


@app.route("/")
def home():
    html_file = TEMPLATE_DIR / "index.html"

    if not html_file.is_file():
        html_file = BASE_DIR / "index.html"

    if not html_file.is_file():
        return "Faili index.html halijapatikana.", 404

    return Response(
        html_file.read_text(encoding="utf-8"),
        content_type="text/html; charset=utf-8"
    )


@app.route("/api/status")
def status():
    return jsonify({
        "app": "FREEMAXY AI STUDIO",
        "server": "running",
        "music_ai": "ACE-Step",
        "video_ai": "Wan2.2",
        "audio_generation_enabled": True,
        "video_generation_enabled": True
    })


@app.route("/api/hf-test")
def hf_test():
    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        return jsonify({
            "ok": False,
            "message": "HF_TOKEN haijawekwa kwenye Render."
        }), 400

    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20
        )

        if response.status_code == 200:
            return jsonify({
                "ok": True,
                "message": "Token imekubaliwa."
            })

        return jsonify({
            "ok": False,
            "message": f"Hali ya majibu: {response.status_code}"
        }), 400

    except requests.RequestException:
        return jsonify({
            "ok": False,
            "message": "Imeshindikana kuwasiliana na Hugging Face."
        }), 503


@app.route("/api/music", methods=["POST"])
def generate_music():
    data = request.get_json(silent=True) or {}

    prompt = str(data.get("prompt", "")).strip()
    lyrics = str(data.get("lyrics", "")).strip()

    try:
        duration = int(data.get("duration", 180))
    except (TypeError, ValueError):
        duration = 180

    duration = max(30, min(duration, 240))

    if not prompt:
        return jsonify({
            "ok": False,
            "error": "Andika aina ya muziki unaotaka."
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "ok": False,
            "error": "Maelezo ya muziki yasizidi herufi 1000."
        }), 400

    if not lyrics:
        return jsonify({
            "ok": False,
            "error": "Andika maneno ya wimbo kwenye sehemu ya Lyrics."
        }), 400

    if len(lyrics) > 12000:
        return jsonify({
            "ok": False,
            "error": "Maneno ya wimbo ni marefu sana."
        }), 400

    if not music_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "Wimbo mwingine unatengenezwa. Subiri kwanza."
        }), 429

    try:
        client = get_client("ACE-Step/ACE-Step")

        result = client.predict(
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=60,
            guidance_scale=15,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10,
            manual_seeds=None,
            guidance_interval=0.5,
            guidance_interval_decay=0,
            min_guidance_scale=3,
            use_erg_tag=True,
            use_erg_lyric=False,
            use_erg_diffusion=True,
            oss_steps=None,
            guidance_scale_text=0,
            guidance_scale_lyric=0,
            api_name="/__call__"
        )

        filename = save_remote_file(result, "audio")

        return jsonify({
            "ok": True,
            "message": "Wimbo umetengenezwa!",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
            "duration_requested": duration
        })

    except Exception:
        app.logger.exception("ACE-Step music generation failed")
        return jsonify({
            "ok": False,
            "error": (
                "Wimbo haukutengenezwa. "
                "Kagua Render Logs kwa sababu ya hitilafu."
            )
        }), 502

    finally:
        music_lock.release()


@app.route("/api/video", methods=["POST"])
def generate_video():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({
            "ok": False,
            "error": "Andika maelezo ya video."
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "ok": False,
            "error": "Maelezo ya video yasizidi herufi 1000."
        }), 400

    try:
        seconds = float(data.get("seconds", 3))
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance_scale", 1.0))
        seed = int(data.get("seed", 0))
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "Mipangilio ya video si sahihi."
        }), 400

    if not 1 <= seconds <= 5 or not 1 <= steps <= 20:
        return jsonify({
            "ok": False,
            "error": "Tumia sekunde 1–5 na steps 1–20."
        }), 400

    if not video_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "Video nyingine inatengenezwa. Subiri kwanza."
        }), 429

    try:
        client = get_client("mastaP/wan22-remix-sfw-t2v")

        result = client.predict(
            prompt=prompt,
            negative_prompt=str(data.get("negative_prompt", "")),
            seconds=seconds,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            randomize_seed=bool(data.get("randomize_seed", True)),
            api_name="/generate_video"
        )

        filename = save_remote_file(result, "video")

        return jsonify({
            "ok": True,
            "message": "Video imetengenezwa!",
            "filename": filename,
            "video_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}"
        })

    except Exception:
        app.logger.exception("Wan2.2 video generation failed")
        return jsonify({
            "ok": False,
            "error": (
                "Video haikutengenezwa. "
                "Kagua Render Logs kwa sababu ya hitilafu."
            )
        }), 502

    finally:
        video_lock.release()


@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=False
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
