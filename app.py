
@app.route("/api/music", methods=["POST"])
def generate_music():
    data = request.get_json(silent=True) or {}

    prompt = str(data.get("prompt", "")).strip()
    lyrics = str(data.get("lyrics", "")).strip()

    try:
        duration = int(data.get("duration", 180))
    except (TypeError, ValueError):
        duration = 180

    # ACE-Step inaunga mkono muda wa sekunde 30 hadi 240
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
            audio2audio_enable=False,
            ref_audio_strength=0.5,
            ref_audio_input=None,
            lora_name_or_path="none",
            api_name="/__call__",
        )

        filename = save_remote_file(result, "audio")

        return jsonify({
            "ok": True,
            "message": "Wimbo umetengenezwa!",
            "filename": filename,
            "audio_url": f"/generated/{filename}",
            "download_url": f"/generated/{filename}",
            "duration_requested": duration,
        })

    except Exception:
        app.logger.exception("ACE-Step music generation failed")
        return jsonify({
            "ok": False,
            "error": (
                "Wimbo haukutengenezwa. Jaribu tena au "
                "kagua Render Logs kwa sababu ya hitilafu."
            )
        }), 502

    finally:
        music_lock.release()
