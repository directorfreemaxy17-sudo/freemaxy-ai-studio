
from flask import Flask, jsonify, request, send_from_directory, Response
from pathlib import Path
from dotenv import load_dotenv
from gradio_client import Client
import os
import uuid
import threading

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

music_lock = threading.Lock()


def get_hf_token():
    return os.getenv("HF_TOKEN", "").strip() or None


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


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "app": "FREEMAXY AI STUDIO",
        "server": "running",
        "video_ai": "Browser Video Maker",
        "music_ai": "Stable Audio Gradio Space",
        "audio_generation_enabled": True,
        "browser_video_enabled": True,
        "real_ai_video_enabled": False,
        "hf_token_configured": bool(get_hf_token()),
        "login_enabled": False,
        "secure_user_quotas_enabled": False,
    })


@app.route("/api/hf-test", methods=["GET"])
def hf_test():
    token = get_hf_token()

    if not token:
        return jsonify({
            "ok": False,
            "message": "HF_TOKEN haijawekwa kwenye Render Environment."
        }), 400

    try:
        import requests

        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

        if response.status_code == 200:
            return jsonify({
                "ok": True,
                "message": "Hugging Face imekubali token."
            })

        return jsonify({
            "ok": False,
            "message": f"Hugging Face imerudisha hali {response.status_code}."
        }), 400

    except Exception:
        app.logger.exception("Hitilafu ya Hugging Face")
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
            "error": "Andika maelezo ya muziki kwanza."
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "error": "Maelezo yasizidi herufi 1000."
        }), 400

    if not music_lock.acquire(blocking=False):
        return jsonify({
            "error": "Muziki mwingine unatengenezwa. Subiri umalizike."
        }), 429

    try:
        app.logger.info("Inaunganisha na Stable Audio...")

        client = Client("stabilityai/stable-audio-3")

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

        source = result

        if isinstance(result, dict):
            source = (
                result.get("path")
                or result.get("name")
                or result.get("url")
            )

        if isinstance(source, (tuple, list)) and source:
            source = source[0]

        if not isinstance(source, (str, os.PathLike)):
            app.logger.error("Matokeo ya audio yasiyotambulika: %r", result)
            return jsonify({
                "error": "Matokeo ya audio hayajatambulika."
            }), 502

        source_path = Path(source)

        if not source_path.is_file():
            app.logger.error("Faili la audio halipo: %s", source_path)
            return jsonify({
                "error": "Faili la audio halikupatikana."
            }), 502

        filename = (
            f"freemaxy_audio_{uuid.uuid4().hex[:10]}"
            f"{source_path.suffix or '.wav'}"
        )

        (OUTPUT_DIR / filename).write_bytes(
            source_path.read_bytes()
        )

        return jsonify({
            "ok": True,
            "message": "Muziki umetengenezwa!",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
        })

    except Exception:
        app.logger.exception("Hitilafu ya kutengeneza muziki")
        return jsonify({
            "error": (
                "Imeshindikana kutengeneza muziki. "
                "Huduma ya Stable Audio inaweza kuwa haipatikani "
                "au kuwa na kikomo cha matumizi."
            )
        }), 502

    finally:
        music_lock.release()


@app.route("/api/video", methods=["POST"])
def generate_video():
    return jsonify({
        "ok": False,
        "browser_video": True,
        "error": (
            "Video AI ya seva imezimwa kwa sasa. "
            "Tumia VIDEO MAKER kwenye ukurasa wa FREEMAXY AI STUDIO "
            "kutengeneza video ya animation kupitia browser."
        )
    }), 410


@app.route("/generated/<path:filename>", methods=["GET"])
def generated_file(filename):
    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        as_attachment=False
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Ukurasa au njia uliyoomba haijapatikana."
    }), 404


@app.errorhandler(500)
def server_error(error):
    app.logger.exception("Hitilafu ya ndani ya seva")
    return jsonify({
        "error": "Hitilafu ya ndani ya seva."
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("FREEMAXY AI STUDIO inaanza...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
