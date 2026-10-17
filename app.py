
from flask import Flask, jsonify, request, send_from_directory, Response
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client
import os
import uuid
import threading
import requests

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

music_lock = threading.Lock()
video_lock = threading.Lock()


def get_hf_token():
    return os.getenv("HF_TOKEN", "").strip() or None


def get_client(space):
    token = get_hf_token()
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
            source.get("video")
            or source.get("audio")
            or source.get("path")
            or source.get("url")
            or source.get("name")
        )

    if isinstance(source, dict):
        source = source.get("path") or source.get("url")

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
            raise FileNotFoundError(
                "Faili halikupatikana kwenye seva."
            )
        content = source_path.read_bytes()
        suffix = source_path.suffix

    if not suffix:
        suffix = ".mp4" if prefix == "video" else ".wav"

    filename = (
        f"freemaxy_{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    )
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
        content_type="text/html; charset=utf-8",
    )


@app.route("/api/status")
def status():
    return jsonify({
        "app": "FREEMAXY AI STUDIO",
        "server": "running",
        "audio_generation_enabled": True,
        "video_generation_enabled": True,
        "hf_token_configured": bool(get_hf_token()),
    })


@app.route("/api/hf-test")
def hf_test():
    token = get_hf_token()

    if not token:
        return jsonify({
            "ok": False,
            "message": "HF_TOKEN haijawekwa Render."
        }), 400

    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

        if response.ok:
            return jsonify({
                "ok": True,
                "message": "Hugging Face imekubali token."
            })

        return jsonify({
            "ok": False,
            "message": f"Token imekataliwa: HTTP {response.status_code}"
        }), 400

    except requests.RequestException:
        app.logger.exception("Hugging Face connection error")
        return jsonify({
            "ok": False,
            "message": "Imeshindikana kuwasiliana na Hugging Face."
        }), 503


@app.route("/api/music", methods=["POST"])
def generate_music():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({
            "ok": False,
            "error": "Andika maelezo ya muziki kwanza."
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "ok": False,
            "error": "Maelezo yasizidi herufi 1000."
        }), 400

    if not music_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "Muziki mwingine unatengenezwa. Subiri."
        }), 429

    try:
        client = get_client("stabilityai/stable-audio-3")

        result = client.predict(
            variant_key="small-music",
            prompt=prompt,
            duration=10,
            steps=8,
            cfg_scale=1.0,
            sampler_type="pingpong",
            seed=0,
            api_name="/infer",
        )

        filename = save_remote_file(result, "audio")

        return jsonify({
            "ok": True,
            "message": "Muziki umetengenezwa!",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
        })

    except Exception:
        app.logger.exception("Music generation failed")
        return jsonify({
            "ok": False,
            "error": (
                "Muziki haukutengenezwa. Kagua Render Logs "
                "kwa sababu halisi ya hitilafu."
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
            "error": "Andika maelezo ya video kwanza."
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "ok": False,
            "error": "Maelezo ya video yasizidi herufi 1000."
        }), 400

    if not video_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "Video nyingine inatengenezwa. Subiri."
        }), 429

    try:
        client = get_client("OpenKing/wan2-video-generation")

        result = client.predict(
            prompt=prompt,
            image=None,
            width=1280,
            height=704,
            num_frames=73,
            num_inference_steps=35,
            guidance_scale=5,
            seed=-1,
            api_name="/generate_video",
        )

        if not isinstance(result, (tuple, list)) or not result:
            raise ValueError("Matokeo ya video hayajatambulika.")

        video_data = result[0]

        if isinstance(video_data, dict):
            video_source = (
                video_data.get("video")
                or video_data.get("path")
                or video_data.get("url")
            )
        else:
            video_source = video_data

        filename = save_remote_file(video_source, "video")

        return jsonify({
            "ok": True,
            "message": "Video imetengenezwa!",
            "filename": filename,
            "video_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
            "status": str(result[1]) if len(result) > 1 else "",
        })

    except Exception:
        app.logger.exception("Video generation failed")
        return jsonify({
            "ok": False,
            "error": (
                "Video haikutengenezwa. Kagua Render Logs "
                "kwa sababu halisi ya hitilafu."
            )
        }), 502

    finally:
        video_lock.release()


@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        as_attachment=False,
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Ukurasa au njia haijapatikana."
    }), 404


@app.errorhandler(500)
def server_error(error):
    app.logger.exception("Internal server error")
    return jsonify({
        "error": "Hitilafu ya ndani ya seva."
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
