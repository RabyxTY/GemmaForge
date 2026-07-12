from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import edge_tts
import requests
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
TEXT_MODEL = "accounts/fireworks/models/kimi-k2p6"
VISION_MODEL = "accounts/fireworks/models/kimi-k2p6"
IMAGE_MODEL = "accounts/fireworks/models/flux-1-schnell-fp8"
ASR_MODEL = "whisper-v3"


def _build_session(api_key: str) -> requests.Session:
    session = requests.Session()
    # trust_env=False disables reading proxy settings from Windows Registry or environment variables
    session.trust_env = False
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def _chat_completion(
    session: requests.Session,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    response_format: Dict[str, Any] | None = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    response = session.post(
        f"{FIREWORKS_BASE_URL}/chat/completions",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response: {data}") from exc


def _guess_mime_type(path: str | os.PathLike[str]) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _encode_file_base64(path: str | os.PathLike[str]) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _sample_frame_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return [0]

    sample_count = max(1, min(sample_count, frame_count))
    if sample_count == 1:
        return [frame_count // 2]

    indices = []
    for i in range(sample_count):
        index = round(i * (frame_count - 1) / (sample_count - 1))
        indices.append(min(frame_count - 1, index))
    return sorted(set(indices))


def transcribe_audio(audio_path: str, api_key: str) -> str:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return ""

    import speech_recognition as sr
    from moviepy.editor import AudioFileClip
    
    wav_path = audio_file.with_suffix(".wav")
    try:
        with AudioFileClip(str(audio_file)) as clip:
            clip.write_audiofile(str(wav_path), codec="pcm_s16le", fps=16000, logger=None)
            
        r = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            audio_data = r.record(source)
            
        try:
            text = r.recognize_google(audio_data, language="ru-RU")
        except Exception:
            try:
                text = r.recognize_google(audio_data, language="en-US")
            except Exception as e:
                text = f"[Речь не распознана: {str(e)}]"
        return text
    except Exception as e:
        return f"[Ошибка транскрибации: {str(e)}]"
    finally:
        if wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass


def analyze_scenes(video_path: str, api_key: str) -> List[Dict[str, Any]]:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    from moviepy.editor import VideoFileClip
    from PIL import Image

    scene_descriptions: List[Dict[str, Any]] = []

    try:
        with VideoFileClip(str(video_file)) as clip:
            duration = clip.duration
            sample_count = 5 if duration < 15 else 7
            
            # Sample timestamps uniformly
            timestamps = [i * duration / (sample_count - 1) for i in range(sample_count)] if sample_count > 1 else [duration / 2]
            
            with _build_session(api_key) as session:
                with tempfile.TemporaryDirectory(prefix="gemmaforge_frames_") as temp_dir:
                    for sample_number, t in enumerate(timestamps, start=1):
                        t_pos = max(0.0, min(t, duration - 0.05))
                        frame_rgb = clip.get_frame(t_pos)
                        
                        frame_path = Path(temp_dir) / f"frame_{sample_number:02d}.jpg"
                        img = Image.fromarray(frame_rgb)
                        img.save(str(frame_path), "JPEG")

                        image_b64 = _encode_file_base64(frame_path)
                        mime_type = _guess_mime_type(frame_path)
                        timestamp_seconds = round(t_pos, 2)

                        raw_content = _chat_completion(
                            session,
                            model=VISION_MODEL,
                            temperature=0.2,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You describe ad video frames. Return JSON with keys: "
                                        "summary, objects, setting, mood, style, text_overlay."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Describe this frame for downstream short-video generation. "
                                                "Focus on subject, action, setting, composition, visible text, and tone."
                                            ),
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{mime_type};base64,{image_b64}"
                                            },
                                        },
                                    ],
                                },
                            ],
                        )

                        try:
                            description = json.loads(raw_content)
                        except json.JSONDecodeError:
                            description = {"summary": raw_content.strip()}

                        description["frame_index"] = sample_number
                        description["timestamp_seconds"] = timestamp_seconds
                        scene_descriptions.append(description)
    except Exception as e:
        raise ValueError(f"Failed to analyze scenes using moviepy: {str(e)}")

    return scene_descriptions

def describe_reference_image(image_bytes: bytes, mime_type: str, api_key: str) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    with _build_session(api_key) as session:
        description = _chat_completion(
            session,
            model=VISION_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional image analysis assistant. Describe the scene, mood, details, and color palette of this reference image. Provide specific keywords that will help FLUX.1 text-to-image model generate matching styles.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this reference image and extract the style, visual theme, and descriptions.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
        )
    return description


def generate_script(
    concept: str,
    visual_context: Sequence[Dict[str, Any]] | str,
    weights: Dict[str, Any],
    api_key: str,
    lang_code: str = "en",
) -> Dict[str, Any]:
    if isinstance(visual_context, str):
        visual_context_text = visual_context
    else:
        visual_context_text = json.dumps(list(visual_context), ensure_ascii=False, indent=2)

    weights_text = json.dumps(weights, ensure_ascii=False, indent=2)

    LANG_NAMES = {
        "en": "English",
        "es": "Spanish",
        "ru": "Russian",
        "de": "German",
        "fr": "French"
    }
    target_lang_name = LANG_NAMES.get(lang_code, "English")

    # Стили по умолчанию
    if lang_code == "es":
        default_styles = ["Sarcástico", "Humorístico", "Jerga de TI", "Formal"]
    elif lang_code == "ru":
        default_styles = ["Саркастический", "Юмористический", "IT-сленг", "Деловой"]
    else:
        default_styles = ["Sarcastic", "Humorous", "IT-slang", "Business"]

    custom_style_desc = weights.get("custom_style_desc", "")
    if custom_style_desc:
        style_list_desc = f"The 4 styles must be: '{default_styles[0]}', '{default_styles[1]}', '{default_styles[2]}', '{custom_style_desc}'."
        user_style_desc = f"Generate 4 distinct script variants ({default_styles[0]}, {default_styles[1]}, {default_styles[2]}, and {custom_style_desc} style)."
    else:
        style_list_desc = f"The 4 styles must be: '{default_styles[0]}', '{default_styles[1]}', '{default_styles[2]}', '{default_styles[3]}'."
        user_style_desc = f"Generate 4 distinct script variants ({default_styles[0]}, {default_styles[1]}, {default_styles[2]}, {default_styles[3]})."

    with _build_session(api_key) as session:
        raw_content = _chat_completion(
            session,
            model=TEXT_MODEL,
            temperature=0.8,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert short-form video scriptwriter. "
                        "Return a valid JSON object with a single key 'variants' which contains a list of exactly 4 script variants. "
                        "Each variant in the list must have the keys: "
                        "style, title, hook, scenes. "
                        f"{style_list_desc} "
                        "Each variant's 'scenes' list must contain exactly 4 scene items. "
                        "Each scene item must include: scene_number, narration, image_prompt, "
                        "duration_seconds, on_screen_text, transition."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Concept:\n{concept}\n\n"
                        f"Visual context:\n{visual_context_text}\n\n"
                        f"Style weights:\n{weights_text}\n\n"
                        f"{user_style_desc} "
                        "Keep narration short, clear, and easy to read for TTS. "
                        f"CRITICAL: Write the narration, title, hook, and on_screen_text strictly in {target_lang_name}. "
                        "CRITICAL: The 'image_prompt' MUST ALWAYS BE WRITTEN IN ENGLISH. Write highly detailed, artistic, and cinematic prompts for FLUX.1 image generator, describing camera angles, lighting, styles, and avoiding abstract text."
                    ),
                },
            ],
        )

    result = json.loads(raw_content)
    variants = result.get("variants", [])
    if len(variants) != 4:
        raise ValueError("The model did not return exactly 4 script variants.")

    return result


def generate_image(prompt: str, aspect_ratio: str, api_key: str) -> str:
    import urllib.parse
    width, height = 768, 1344
    if aspect_ratio == "1344x768" or "16:9" in aspect_ratio:
        width, height = 1344, 768
    elif "1:1" in aspect_ratio:
        width, height = 1024, 1024

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&private=true"
    
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    output_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.jpg"
    output_path.write_bytes(response.content)
    return str(output_path)


async def synthesize_speech(text: str, lang: str, output_path: str) -> str:
    import asyncio
    from gtts import gTTS
    
    tts = gTTS(text=text, lang=lang)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, tts.save, output_path)
    return output_path


def draw_subtitles(image_path: str, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    
    if not text:
        return
        
    try:
        img = Image.open(image_path)
        w, h = img.size
        
        # Load Arial font which supports Cyrillic. Windows path fallback
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        font_size = int(h * 0.045) # 4.5% of image height
        if font_size < 16:
            font_size = 16
            
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        
        # Temporarily create a draw context to measure text sizes
        test_draw = ImageDraw.Draw(img)
        
        # Split text into lines wrapping appropriately
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = test_draw.textbbox((0, 0), test_line, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w > w * 0.85:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
            
        # Calculate box sizes
        total_h = 0
        line_dims = []
        for line in lines:
            bbox = test_draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            line_dims.append((line_w, line_h))
            total_h += line_h + 12
            
        max_line_w = max([d[0] for d in line_dims])
        
        padding_x = 22
        padding_y = 16
        box_w = max_line_w + padding_x * 2
        box_h = total_h + padding_y * 2
        
        box_x1 = (w - box_w) // 2
        box_y1 = h - 100 - box_h
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        
        # Draw translucent black panel
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=(0, 0, 0, 160)) # 60% opacity
        
        # Blend overlay onto image
        img_rgba = img.convert("RGBA")
        blended = Image.alpha_composite(img_rgba, overlay)
        
        # Draw text lines
        draw = ImageDraw.Draw(blended)
        y_offset = box_y1 + padding_y
        for i, line in enumerate(lines):
            line_w, line_h = line_dims[i]
            x = (w - line_w) // 2
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += line_h + 12
            
        blended.convert("RGB").save(image_path, "JPEG")
    except Exception as e:
        print(f"Error drawing subtitles: {e}")


def compile_video(scenes: Sequence[Dict[str, Any]], output_path: str) -> str:
    if not scenes:
        raise ValueError("No scenes provided for video compilation.")

    clips = []
    audio_clips = []
    final_clip = None

    try:
        for scene in scenes:
            image_path = scene["image_path"]
            audio_path = scene["audio_path"]
            on_screen_text = scene.get("on_screen_text", "")
            
            draw_subtitles(image_path, on_screen_text)

            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio_clip = AudioFileClip(audio_path)
            
            # Make sure clip duration matches the audio track duration to prevent overlaps
            duration = float(audio_clip.duration or 5.0)
            if duration <= 0:
                duration = float(scene.get("duration_seconds") or 5.0)

            clip = (
                ImageClip(image_path)
                .set_duration(duration)
                .set_audio(audio_clip)
                .fadein(0.2)
                .fadeout(0.2)
            )

            clips.append(clip)
            audio_clips.append(audio_clip)

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=max(1, os.cpu_count() or 1),
        )
        return output_path
    finally:
        if final_clip is not None:
            final_clip.close()
        for clip in clips:
            clip.close()
        for audio_clip in audio_clips:
            audio_clip.close()


def assemble_revoice_audio_on_video(
    original_video_path: str,
    scenes: Sequence[Dict[str, Any]],
    output_path: str,
) -> str:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
    
    audio_clips = []
    try:
        for scene in scenes:
            audio_path = scene["audio_path"]
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_clip = AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
            
        final_audio = concatenate_audioclips(audio_clips)
        
        with VideoFileClip(original_video_path) as video:
            final_clip = video.set_audio(final_audio)
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=video.fps or 24,
                threads=max(1, os.cpu_count() or 1),
            )
            
        return output_path
    finally:
        for ac in audio_clips:
            ac.close()
