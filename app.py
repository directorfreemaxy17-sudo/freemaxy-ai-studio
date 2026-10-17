
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
video_lock = threading.Lock()


@app.route("/")
def home():
    html_file = TEMPLATE_DIR / "index.html"

    if not html_file.is_file():
        html_file = BASE_DIR / "index.html"

    if not html_file.is_file():
        return (
            "Faili index.html halijapatikana. "
            "Weka index.html kwenye repository yako.",
            404,
        )

    return Response(
        html_file.read_text(encoding="utf-8"),
        content_type="text/html; charset=utf-8",
    )


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "app": "FREEMAXY AI STUDIO",
        "server": "running",
        "video_ai": "Wan2.2 Remix Gradio Space",
        "music_ai": "Stable Audio Gradio Space",
        "audio_generation_enabled": True,
        "video_generation_enabled": True,
    })


@app.route("/api/hf-test", methods=["GET"])
def hf_test():
    import requests

    token = os.getenv("HF_TOKEN")

    if not token:
        return jsonify({
            "ok": False,
            "message": (
                "HF_TOKEN haijawekwa. "
                "Weka token kwenye Render Environment ikiwa inahitajika."
            ),
        }), 400

    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

        if response.status_code == 200:
            return jsonify({
                "ok": True,
                "message": "Token imekubaliwa.",
            })

        return jsonify({
            "ok": False,
            "message": f"Hali ya majibu: {response.status_code}",
        }), 400

    except requests.RequestException:
        return jsonify({
            "ok": False,
            "message": "Imeshindikana kuwasiliana na Hugging Face.",
        }), 503


@app.route("/api/music", methods=["POST"])
def generate_music():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({
            "error": "Andika maelezo ya muziki kwanza.",
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "error": "Maelezo yasizidi herufi 1000.",
        }), 400

    if not music_lock.acquire(blocking=False):
        return jsonify({
            "error": "Muziki mwingine unatengenezwa. Subiri umalizike.",
        }), 429

    try:
        print("Inaunganisha na huduma ya Stable Audio...")

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

        # Baadhi ya Gradio Spaces hurudisha dictionary
        # yenye njia ya faili badala ya string moja kwa moja.
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
            return jsonify({
                "error": (
                    "Muundo wa matokeo ya audio haujatambulika. "
                    "Huenda API ya huduma imebadilika."
                ),
            }), 502

        source_path = Path(source)

        if not source_path.is_file():
            return jsonify({
                "error": (
                    "Faili la audio halikupatikana. "
                    "Kagua matokeo ya huduma ya AI."
                ),
            }), 502

        filename = f"freemaxy_audio_{uuid.uuid4().hex[:10]}{source_path.suffix or '.wav'}"
        output_path = OUTPUT_DIR / filename
        output_path.write_bytes(source_path.read_bytes())

        return jsonify({
            "ok": True,
            "message": "Muziki umetengenezwa!",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
        })

    except Exception as error:
        print("Hitilafu ya muziki:", repr(error))
        return jsonify({
            "error": (
                "Imeshindikana kutengeneza muziki. "
                "Huduma ya AI inaweza kuwa haipatikani au API imebadilika."
            ),
        }), 502

    finally:
        music_lock.release()


@app.route("/api/video", methods=["POST"])
def generate_video():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({
            "error": "Tafadhali andika maelezo ya video.",
        }), 400

    if len(prompt) > 1000:
        return jsonify({
            "error": "Maelezo ya video yasizidi herufi 1000.",
        }), 400

    try:
        seconds = float(data.get("seconds", 3))
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance_scale", 1.0))
        seed = int(data.get("seed", 0))
    except (TypeError, ValueError):
        return jsonify({
            "error": "Muda, steps, seed au guidance scale si sahihi.",
        }), 400

    if not 1 <= seconds <= 5:
        return jsonify({
            "error": "Muda uwe sekunde 1 hadi 5.",
        }), 400

    if not 1 <= steps <= 20:
        return jsonify({
            "error": "Steps ziwe kati ya 1 na 20.",
        }), 400

    if not video_lock.acquire(blocking=False):
        return jsonify({
            "error": "Video nyingine inatengenezwa. Subiri imalizike.",
        }), 429

    try:
        print("Inaunganisha na huduma ya Wan2.2 AI VIDEO...")

        client = Client("mastaP/wan22-remix-sfw-t2v")

        result = client.predict(
            prompt=prompt,
            negative_prompt=str(data.get("negative_prompt", "")),
            seconds=seconds,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            randomize_seed=bool(data.get("randomize_seed", True)),
            api_name="/generate_video",
        )

        if isinstance(result, dict):
            video_path = (
                result.get("path")
                or result.get("name")
                or result.get("video")
            )
            seed_used = result.get("seed")
        elif isinstance(result, (tuple, list)) and result:
            video_path = result[0]
            seed_used = result[1] if len(result) > 1 else None
        else:
            video_path = result
            seed_used = None

        if not isinstance(video_path, (str, os.PathLike)):
            return jsonify({
                "error": (
                    "Matokeo ya video hayajatambulika. "
                    "Huenda API ya huduma imebadilika."
                ),
            }), 502

        source_path = Path(video_path)

        if not source_path.is_file():
            return jsonify({
                "error": "Faili la video halikupatikana kwenye seva.",
            }), 502

        filename = f"freemaxy_video_{uuid.uuid4().hex[:10]}{source_path.suffix or '.mp4'}"
        output_path = OUTPUT_DIR / filename
        output_path.write_bytes(source_path.read_bytes())

        return jsonify({
            "ok": True,
            "message": "Video imetengenezwa!",
            "filename": filename,
            "video_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
            "seed": seed_used,
        })

    except Exception as error:
        print("Hitilafu ya video:", repr(error))
        return jsonify({
            "error": (
                "Imeshindikana kutengeneza video. "
                "Jaribu tena baadaye au kagua huduma ya AI."
            ),
        }), 502

    finally:
        video_lock.release()


@app.route("/generated/<path:filename>", methods=["GET"])
def generated_file(filename):
    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        as_attachment=False,
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Ukurasa au njia uliyoomba haijapatikana.",
    }), 404


@app.errorhandler(500)
def server_error(error):
    app.logger.exception("Hitilafu ya ndani ya seva")
    return jsonify({
        "error": "Hitilafu ya ndani ya seva. Jaribu tena baadaye.",
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    print("FREEMAXY AI STUDIO inaanza...")
    print(f"Port: {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
