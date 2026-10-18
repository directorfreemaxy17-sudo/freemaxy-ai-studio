
import os
import uuid
import threading
import logging
import requests

from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, jsonify, request, send_from_directory, Response
from dotenv import load_dotenv
from gradio_client import Client

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

music_lock = threading.Lock()
video_lock = threading.Lock()


# COMMON HELPERS

def get_client(space):
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return Client(space, token=token)
    return Client(space)


def find_file(value, prefix):
    if isinstance(value, dict):
        keys = (
            ["audio", "audio_path", "first_audio_path",
             "path", "url", "name"]
            if prefix == "audio"
            else ["video", "video_path", "path", "url", "name"]
        )

        for key in keys:
            if value.get(key):
                found = find_file(value[key], prefix)
                if found:
                    return found

        for item in value.values():
            found = find_file(item, prefix)
            if found:
                return found

    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_file(item, prefix)
            if found:
                return found

    elif isinstance(value, (str, os.PathLike)):
        text_value = os.fspath(value)

        if text_value.startswith(("http://", "https://")):
            return text_value

        path = Path(text_value)
        if path.is_file():
            return path

    return None


def save_remote_file(result, prefix):
    source = find_file(result, prefix)

    if source is None:
        raise ValueError("AI haikurudisha faili linalotambulika.")

    if isinstance(source, Path):
        content = source.read_bytes()
        suffix = source.suffix
    else:
        parsed = urlparse(source)

        if parsed.scheme != "https":
            raise ValueError("Chanzo cha faili hakikubaliki.")

        response = requests.get(source, timeout=240)
        response.raise_for_status()
        content = response.content
        suffix = Path(parsed.path).suffix

    if not content:
        raise ValueError("Faili lililorudishwa halina data.")

    if not suffix:
        suffix = ".mp3" if prefix == "audio" else ".mp4"

    filename = f"freemaxy_{prefix}_{uuid.uuid4().hex[:12]}{suffix}"
    (OUTPUT_DIR / filename).write_bytes(content)
    return filename


def error_json(message, code=400):
    return jsonify({"ok": False, "error": message}), code


# HOME PAGE

@app.route("/")
def home():
    html_file = TEMPLATE_DIR / "index.html"

    if not html_file.is_file():
        html_file = BASE_DIR / "index.html"

    if not html_file.is_file():
        return "Faili index.html halijapatikana.", 404

    response = Response(
        html_file.read_text(encoding="utf-8"),
        content_type="text/html; charset=utf-8"
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


# PWA FILES

@app.route("/manifest.json")
def serve_manifest():
    if not (BASE_DIR / "manifest.json").is_file():
        return error_json("manifest.json haipo.", 404)

    return send_from_directory(
        str(BASE_DIR), "manifest.json",
        mimetype="application/manifest+json"
    )


@app.route("/icon.svg")
def serve_icon():
    if not (BASE_DIR / "icon.svg").is_file():
        return error_json("icon.svg haipo.", 404)

    return send_from_directory(
        str(BASE_DIR), "icon.svg",
        mimetype="image/svg+xml"
    )


@app.route("/sw.js")
def serve_service_worker():
    if not (BASE_DIR / "sw.js").is_file():
        return error_json("sw.js haipo.", 404)

    response = send_from_directory(
        str(BASE_DIR), "sw.js",
        mimetype="application/javascript"
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Service-Worker-Allowed"] = "/"
    return response


# STATUS

@app.route("/api/status")
def status():
    return jsonify({
        "app": "FREEMAXY AI STUDIO",
        "server": "running",
        "chat": "configured",
        "music_ai": "ACE-Step",
        "video_ai": "Wan2.2"
    })


# HUGGING FACE TOKEN TEST

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
            "message": f"Hugging Face imerudisha HTTP {response.status_code}."
        }), 400

    except requests.RequestException as exc:
        app.logger.exception("Hugging Face token check failed")
        return jsonify({
            "ok": False,
            "message": f"Muunganisho umeshindwa: {type(exc).__name__}"
        }), 503


# FREEMAXY AI CHAT

@app.route("/api/chat", methods=["POST"])
def ai_chat():
    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        return error_json("HF_TOKEN haijawekwa kwenye Render.", 503)

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return error_json("Tuma ujumbe katika muundo sahihi wa JSON.")

    incoming = data.get("messages")

    if not isinstance(incoming, list) or not incoming:
        return error_json("Andika ujumbe kwanza.")

    messages = [{
        "role": "system",
        "content": (
            "You are FREEMAXY AI, a helpful general-purpose assistant. "
            "Answer clearly, accurately and respectfully. "
            "If the user writes in Swahili, answer in Swahili. "
            "Help with education, technology, writing, ideas and creative work. "
            "Never claim to have performed an action you did not do."
        )
    }]

    for item in incoming[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        if len(content) > 4000:
            return error_json("Ujumbe mmoja usizidi herufi 4000.")

        messages.append({"role": role, "content": content})

    if len(messages) < 2 or messages[-1]["role"] != "user":
        return error_json("Tuma ujumbe mpya ili AI ijibu.")

    model = os.getenv(
        "HF_CHAT_MODEL",
        "openai/gpt-oss-120b:fastest"
    ).strip()

    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 700,
                "stream": False
            },
            timeout=90
        )

        if response.status_code != 200:
            app.logger.warning(
                "Chat provider returned HTTP %s: %s",
                response.status_code, response.text[:500]
            )
            return error_json(
                "Huduma ya AI haikupatikana. Kagua token, ruhusa, modeli na quota ya Hugging Face.",
                502
            )

        result = response.json()
        choices = result.get("choices", [])

        if not choices:
            raise ValueError("AI haikurudisha majibu.")

        reply = choices[0].get("message", {}).get("content", "")

        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("AI haikurudisha jibu la maandishi.")

        return jsonify({"ok": True, "reply": reply.strip()})

    except requests.Timeout:
        return error_json("AI imechelewa kujibu. Jaribu tena.", 504)

    except requests.RequestException:
        app.logger.exception("FREEMAXY chat connection failed")
        return error_json("Imeshindikana kuwasiliana na huduma ya AI.", 502)

    except (ValueError, KeyError, IndexError):
        app.logger.exception("FREEMAXY chat response failed")
        return error_json("Majibu ya AI hayakuweza kusomwa.", 502)


# AI MUSIC

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
        return error_json("Andika aina ya muziki unaotaka.")
    if len(prompt) > 1000:
        return error_json("Maelezo ya muziki yasizidi herufi 1000.")
    if not lyrics:
        return error_json("Andika maneno ya wimbo kwenye sehemu ya Lyrics.")
    if len(lyrics) > 12000:
        return error_json("Maneno ya wimbo ni marefu sana.")

    if not music_lock.acquire(blocking=False):
        return error_json("Wimbo mwingine unatengenezwa. Subiri kwanza.", 429)

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
            "message": "AI imerudisha faili la audio.",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
            "duration_requested": duration
        })

    except Exception as exc:
        app.logger.exception("ACE-Step music generation failed")
        return error_json(
            f"ACE-Step: {type(exc).__name__}: {str(exc)[:300]}",
            502
        )

    finally:
        music_lock.release()


# AI VIDEO

@app.route("/api/video", methods=["POST"])
def generate_video():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return error_json("Andika maelezo ya video.")
    if len(prompt) > 1000:
        return error_json("Maelezo ya video yasizidi herufi 1000.")

    try:
        seconds = float(data.get("seconds", 3))
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance_scale", 1.0))
        seed = int(data.get("seed", 0))
    except (TypeError, ValueError):
        return error_json("Mipangilio ya video si sahihi.")

    if not 1 <= seconds <= 5 or not 1 <= steps <= 20 or not 0 <= guidance <= 20:
        return error_json("Tumia sekunde 1–5, steps 1–20 na guidance scale 0–20.")

    if not video_lock.acquire(blocking=False):
        return error_json("Video nyingine inatengenezwa. Subiri kwanza.", 429)

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
            "message": "AI imerudisha faili la video.",
            "filename": filename,
            "video_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}"
        })

    except Exception as exc:
        app.logger.exception("Wan2.2 video generation failed")
        return error_json(
            f"Wan2.2: {type(exc).__name__}: {str(exc)[:300]}",
            502
        )

    finally:
        video_lock.release()


# GENERATED FILES

@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(
        str(OUTPUT_DIR), filename, as_attachment=False
    )


# START SERVER

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
