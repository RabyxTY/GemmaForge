from __future__ import annotations

import asyncio
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip

from pipeline import (
    analyze_scenes,
    compile_video,
    generate_image,
    generate_script,
    synthesize_speech,
    transcribe_audio,
    assemble_revoice_audio_on_video,
    describe_reference_image,
)


load_dotenv()


def detect_language(text: str) -> str:
    # Count cyrillic vs latin characters to determine language
    cyrillic_chars = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
    latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    if cyrillic_chars > latin_chars:
        return "ru"
    return "en"


def get_voice(text: str, gender: str) -> str:
    if not text or text.startswith("["):
        return "ru-RU-DmitryNeural" if gender == "Male" else "ru-RU-SvetlanaNeural"
    lang = detect_language(text)
    if lang == "ru":
        return "ru-RU-DmitryNeural" if gender == "Male" else "ru-RU-SvetlanaNeural"
    else:
        return "en-US-GuyNeural" if gender == "Male" else "en-US-JennyNeural"


ASPECT_OPTIONS = {
    "9:16": "768x1344",
    "16:9": "1344x768",
}


def run_async(coro):
    import queue
    import threading
    q = queue.Queue()
    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(coro)
            q.put((True, res))
        except Exception as e:
            q.put((False, e))
        finally:
            loop.close()
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    success, val = q.get()
    if success:
        return val
    else:
        raise val


def get_api_key() -> str:
    lang = st.session_state.get("app_lang_key", "en")
    def loc_t(k):
        return TRANSLATIONS[lang].get(k, k)

    api_key = st.sidebar.text_input(
        "Fireworks API Key",
        value=os.getenv("FIREWORKS_API_KEY", ""),
        type="password",
        help="Provide your Fireworks API key. Will fall back to .env if empty."
    ).strip()
    
    if not api_key:
        api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
        
    if not api_key:
        st.info(loc_t("api_key_prompt"))
        st.stop()
        
    return api_key


def build_style_weights(in_sidebar: bool = True) -> Dict[str, Any]:
    lang = st.session_state.get("app_lang_key", "en")
    def loc_t(k):
        return TRANSLATIONS[lang].get(k, k)

    parent = st.sidebar if in_sidebar else st
    with parent.expander(loc_t("fine_tuning_styles"), expanded=False):
        custom_style = st.text_input(loc_t("style_custom"), placeholder=loc_t("style_custom_placeholder"))
        return {
            "sarcasm": st.slider(loc_t("style_sarcasm"), 0, 100, 30),
            "humor": st.slider(loc_t("style_humor"), 0, 100, 40),
            "technical_slang": st.slider(loc_t("style_tech"), 0, 100, 20),
            "formal_tone": st.slider(loc_t("style_formal"), 0, 100, 10),
            "custom_style_desc": custom_style.strip() if custom_style else ""
        }


async def build_story_video(
    script: Dict[str, Any],
    aspect_ratio: str,
    lang_code: str,
    api_key: str,
    style_description: str = "",
    task_key: str = "story_gen",
) -> Dict[str, Any]:
    working_dir = Path(tempfile.mkdtemp(prefix="gemmaforge_story_"))
    rendered_scenes: List[Dict[str, Any]] = []

    def log(msg: str):
        if task_key:
            st.session_state[f"{task_key}_log"] = msg

    log("🎬 Инициализация генерации видео...")
    scenes = script["scenes"]
    for i, scene in enumerate(scenes):
        prompt = scene["image_prompt"]
        if style_description:
            prompt = f"{prompt}, styled as: {style_description}"
        
        log(f"🎨 Сцена {scene['scene_number']}/4: Генерация картинки...")
        image_path = generate_image(prompt, aspect_ratio, api_key)
        
        log(f"🎙️ Сцена {scene['scene_number']}/4: Синтез озвучки реплики...")
        audio_path = working_dir / f"scene_{scene['scene_number']}.mp3"
        await synthesize_speech(scene["narration"], lang_code, str(audio_path))

        rendered_scenes.append(
            {
                **scene,
                "image_path": image_path,
                "audio_path": str(audio_path),
            }
        )

    log("🎞️ Сведение звука и видеоряда в FFmpeg...")
    output_path = working_dir / "story_video.mp4"
    compile_video(rendered_scenes, str(output_path))
    
    log("✅ Ролик успешно скомпилирован!")
    return {
        "script": script,
        "video_path": str(output_path),
    }


def select_or_upload_video(key_prefix: str) -> Path | None:
    UPLOAD_DIR = Path("uploaded_videos")
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Автоматическая очистка старых загруженных файлов (старше 15 минут)
    now = time.time()
    for f in UPLOAD_DIR.glob("*.*"):
        if now - f.stat().st_mtime > 900:
            is_active = False
            for k in list(st.session_state.keys()):
                if k.endswith("_chosen_path") and st.session_state[k] == str(f):
                    is_active = True
                    break
            if not is_active:
                try:
                    f.unlink()
                except Exception:
                    pass
                    
    saved_videos = sorted(list(UPLOAD_DIR.glob("*.*")), key=lambda p: p.stat().st_mtime, reverse=True)
    
    lang = st.session_state.get("app_lang_key", "en")
    def loc_t(k):
        return TRANSLATIONS[lang].get(k, k)

    st.write(loc_t('history_title'))
    
    # Render layout with horizontal buttons
    if saved_videos:
        cols = st.columns(min(len(saved_videos) + 1, 4))
        with cols[0]:
            if st.button(loc_t('new_video'), key=f"{key_prefix}_btn_new", use_container_width=True):
                st.session_state[f"{key_prefix}_chosen_path"] = None
                
        for idx, p in enumerate(saved_videos):
            col_idx = (idx + 1) % 4
            with cols[col_idx]:
                short_name = p.name if len(p.name) <= 15 else p.name[:12] + "..."
                if st.button(f"🎬 {short_name}", key=f"{key_prefix}_btn_{p.name}", help=p.name, use_container_width=True):
                    st.session_state[f"{key_prefix}_chosen_path"] = str(p)
    else:
        st.info("History is empty. Upload a video below." if lang == "en" else "El historial está vacío. Suba un video a continuación.")
        st.session_state[f"{key_prefix}_chosen_path"] = None
        
    chosen_path = st.session_state.get(f"{key_prefix}_chosen_path")
    
    if chosen_path is None:
        uploaded_file = st.file_uploader(
            loc_t('upload_label'),
            type=["mp4", "mov", "avi", "mkv"],
            key=f"{key_prefix}_uploader"
        )
        if uploaded_file is not None:
            target_path = UPLOAD_DIR / uploaded_file.name
            target_path.write_bytes(uploaded_file.getvalue())
            st.success(f"Video {uploaded_file.name} saved!" if lang == "en" else f"¡Video {uploaded_file.name} guardado!")
            st.session_state[f"{key_prefix}_chosen_path"] = str(target_path)
            # Re-render to show updated history buttons
            st.rerun()
        return None
    else:
        st.markdown(f"{loc_t('chosen_video')} `{Path(chosen_path).name}`")
        if st.button(loc_t('reset_video'), key=f"{key_prefix}_btn_reset", type="secondary"):
            st.session_state[f"{key_prefix}_chosen_path"] = None
            st.rerun()
        return Path(chosen_path)


def analyze_revoice_video(
    video_path: str,
    api_key: str,
) -> Dict[str, Any]:
    source_video_path = Path(video_path)
    working_dir = Path(tempfile.mkdtemp(prefix="gemmaforge_revoice_"))

    status_text = st.empty()
    
    status_text.text("Анализируем видеоряд...")
    try:
        visual_context = analyze_scenes(str(source_video_path), api_key)
    except Exception as e:
        visual_context = [{"summary": f"[Не удалось проанализировать сцены: {str(e)}]", "frame_index": 1, "timestamp_seconds": 0.0}]

    status_text.text("Распознаем оригинальный звук...")
    transcript = ""
    extracted_audio_path = working_dir / "source_audio.mp3"
    
    try:
        with VideoFileClip(str(source_video_path)) as clip:
            if clip.audio is not None:
                clip.audio.write_audiofile(str(extracted_audio_path), logger=None)
                transcript = transcribe_audio(str(extracted_audio_path), api_key)
    except Exception as e:
        transcript = f"[Не удалось распознать оригинальный звук: {str(e)}]"

    status_text.empty()
    return {
        "visual_context": visual_context,
        "transcript": transcript,
        "source_video_path": str(source_video_path),
    }


async def assemble_revoice_video(
    script: Dict[str, Any],
    original_video_path: str,
    lang_code: str,
    api_key: str,
    task_key: str = "revoice_gen",
) -> Dict[str, Any]:
    working_dir = Path(tempfile.mkdtemp(prefix="gemmaforge_assemble_"))
    rendered_scenes: List[Dict[str, Any]] = []
    
    def log(msg: str):
        if task_key:
            st.session_state[f"{task_key}_log"] = msg

    log("🎬 Инициализация ре-озвучки...")
    for i, scene in enumerate(script["scenes"]):
        log(f"🎙️ Сцена {scene['scene_number']}/4: Синтез новой озвучки...")
        audio_path = working_dir / f"scene_{scene['scene_number']}.mp3"
        await synthesize_speech(scene["narration"], lang_code, str(audio_path))
        rendered_scenes.append(
            {
                **scene,
                "audio_path": str(audio_path),
            }
        )

    log("🎞️ Перенос новых аудиодорожек на исходный видеоряд...")
    output_path = working_dir / "revoiced_video.mp4"
    assemble_revoice_audio_on_video(original_video_path, rendered_scenes, str(output_path))
    
    log("🎵 Экспорт отдельного MP3 файла озвучки...")
    from moviepy.editor import concatenate_audioclips, AudioFileClip
    audio_clips = [AudioFileClip(s["audio_path"]) for s in rendered_scenes]
    final_audio = concatenate_audioclips(audio_clips)
    final_audio_path = working_dir / "revoiced_audio.mp3"
    final_audio.write_audiofile(str(final_audio_path), logger=None)
    for ac in audio_clips:
        ac.close()
        
    log("✅ Видео успешно переозвучено!")
    return {
        "script": script,
        "video_path": str(output_path),
        "audio_path": str(final_audio_path),
    }


def run_in_background(task_key: str, async_func, *args, **kwargs):
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    import threading
    
    st.session_state[f"{task_key}_status"] = "running"
    st.session_state[f"{task_key}_result"] = None
    st.session_state[f"{task_key}_error"] = None
    st.session_state[f"{task_key}_log"] = "Запуск фоновой задачи..."
    
    def worker():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(async_func(*args, **kwargs))
            st.session_state[f"{task_key}_result"] = res
            st.session_state[f"{task_key}_status"] = "completed"
        except Exception as e:
            st.session_state[f"{task_key}_error"] = str(e)
            st.session_state[f"{task_key}_status"] = "failed"
            
    t = threading.Thread(target=worker)
    add_script_run_ctx(t)
    t.start()


TRANSLATIONS = {
    "en": {
        "title": "GemmaForge",
        "subtitle": "AI Re-Voicer & Video Generator",
        "desc": "Powered by Kimi K2.6, FLUX.1 and Fireworks AI",
        "mode_selector": "Operating Mode",
        "mode_revoice": "Translate / Re-voice Video",
        "mode_story": "Create Video from Story (From Scratch)",
        "mode_analyzer": "🔍 Video Analyzer (Speech & Scenes)",
        "api_key_label": "Enter Fireworks API Key",
        "api_key_placeholder": "fw_...",
        "theme_label": "Theme",
        "theme_light": "☀️ Light",
        "theme_dark": "🌙 Dark",
        "lang_interface": "App Language / Idioma",
        "lang_voice": "Voice Language",
        "text_stylization": "🎛️ Text Stylization (Gemma)",
        "formal_tone": "Formal Tone",
        "custom_style": "Custom script style (e.g. comedic, horror)",
        "history_title": "📂 Uploaded Videos History (click to select):",
        "new_video": "➕ New Video",
        "reset_video": "❌ Reset Video Selection",
        "chosen_video": "👉 Selected video from history:",
        "upload_label": "Upload source video (MP4 / MOV / AVI)",
        "step1_btn": "🚀 Step 1: Analyze and write scripts",
        "step1_spinner": "Analyzing video and writing scripts...",
        "step2_btn": "🎭 Re-voice video with variant",
        "original_text": "🎙️ Original video text:",
        "script_options": "📜 Select one of 4 script variants:",
        "success_gen": "New video successfully generated!",
        "audio_download": "🎵 Download only re-voiced audio:",
        "script_text": "📜 New narration text:",
        "ai_metadata": "🔍 Detailed AI Metadata",
        "upload_style_img": "🖼️ Style Reference Image (Optional)",
        "style_desc_success": "🎨 Style read from image successfully!",
        "prompt_placeholder": "Describe your video idea... (e.g. Pixar style funny short about a cat)",
        "step1_story_btn": "🚀 Step 1: Write story scripts",
        "step1_story_spinner": "Gemma is writing story scripts...",
        "compile_story_btn": "🎬 Assemble video for variant",
        "video_success": "Video successfully created!",
        "chosen_script": "📜 Chosen Script",
        "original_video_header": "Original Video",
        "analyzer_header": "🔍 Mode: Video Analyzer",
        "analyzer_desc": "Upload a video. AI will analyze every visual frame and transcribe the original speech, producing a detailed timeline.",
        "analyzer_btn": "🚀 Start analysis",
        "analyzer_spinner": "Analyzing frames and transcribing speech...",
        "analyzer_success": "Video analysis completed successfully!",
        "analyzer_transcript": "🎙️ Original Speech Transcription (Whisper):",
        "analyzer_visual": "📸 Visual Scene Timeline (Vision-model):",
        "second": "Second",
        "frame": "Frame",
        "no_speech": "No speech detected in the original video.",
        "global_status_title": "⏳ Assembling video in background (you can switch tabs)...",
        "toast_revoice_success": "Video re-voiced successfully!",
        "toast_story_success": "Video created successfully!",
        "err_revoice": "Re-voice error",
        "err_story": "Story assembly error",
        "fine_tuning_styles": "🎛️ Fine-Tuning Styles",
        "style_custom": "Custom Style",
        "style_custom_placeholder": "e.g. Pirate slang, Gamer slang...",
        "style_sarcasm": "Sarcasm",
        "style_humor": "Humor",
        "style_tech": "Technical Slang",
        "style_formal": "Formal Tone",
        "api_key_prompt": "Please enter your Fireworks API Key (starts with fw_...) in the sidebar to start."
    },
    "es": {
        "title": "GemmaForge",
        "subtitle": "Doblador de IA y Generador de Video",
        "desc": "Desarrollado por Kimi K2.6, FLUX.1 y Fireworks AI",
        "mode_selector": "Modo de Funcionamiento",
        "mode_revoice": "Traducir / Re-doblaje de Video",
        "mode_story": "Crear Video desde Guión (Desde Cero)",
        "mode_analyzer": "🔍 Analizador de Video (Voz y Escenas)",
        "api_key_label": "Ingrese la clave API de Fireworks",
        "api_key_placeholder": "fw_...",
        "theme_label": "Tema",
        "theme_light": "☀️ Claro",
        "theme_dark": "🌙 Oscuro",
        "lang_interface": "Idioma de la App",
        "lang_voice": "Idioma de la Voz",
        "text_stylization": "🎛️ Estilización de Texto (Gemma)",
        "formal_tone": "Tono Formal",
        "custom_style": "Estilo de guión personalizado (ej. comedia, terror)",
        "history_title": "📂 Historial de videos subidos (clic para seleccionar):",
        "new_video": "➕ Nuevo Video",
        "reset_video": "❌ Restablecer Selección",
        "chosen_video": "👉 Video seleccionado del historial:",
        "upload_label": "Subir video original (MP4 / MOV / AVI)",
        "step1_btn": "🚀 Paso 1: Analizar y escribir guiones",
        "step1_spinner": "Analizando video y escribiendo guiones...",
        "step2_btn": "🎭 Re-doblar video con variante",
        "original_text": "🎙️ Texto del video original:",
        "script_options": "📜 Seleccione una de las 4 variantes de guión:",
        "success_gen": "¡Nuevo video generado con éxito!",
        "audio_download": "🎵 Descargar solo audio de re-doblaje:",
        "script_text": "📜 Nuevo texto de narración:",
        "ai_metadata": "🔍 Metadatos detallados de IA",
        "upload_style_img": "🖼️ Imagen de Referencia de Estilo (Opcional)",
        "style_desc_success": "¡Estilo de imagen leído con éxito!",
        "prompt_placeholder": "Describe la idea de tu video... (ej. corto divertido sobre un gato al estilo Pixar)",
        "step1_story_btn": "🚀 Paso 1: Escribir guiones de la historia",
        "step1_story_spinner": "Gemma está escribiendo los guiones...",
        "compile_story_btn": "🎬 Armar video para la variante",
        "video_success": "¡Video creado con éxito!",
        "chosen_script": "📜 Guión Elegido",
        "original_video_header": "Video Original",
        "analyzer_header": "🔍 Modo: Analizador de Video",
        "analyzer_desc": "Suba un video. La IA analizará cada fotograma visual y transcribirá el discurso original, produciendo una línea de tiempo detallada.",
        "analyzer_btn": "🚀 Iniciar análisis",
        "analyzer_spinner": "Analizando fotogramas y transcribiendo discurso...",
        "analyzer_success": "¡Análisis de video completado con éxito!",
        "analyzer_transcript": "🎙️ Transcripción del discurso original (Whisper):",
        "analyzer_visual": "📸 Línea de tiempo visual (Modelo de Visión):",
        "second": "Segundo",
        "frame": "Fotograma",
        "no_speech": "No se detectó voz en el video original.",
        "global_status_title": "⏳ Ensamblando video en segundo plano (puede cambiar de pestaña)...",
        "toast_revoice_success": "¡Video re-doblado con éxito!",
        "toast_story_success": "¡Video creado con éxito!",
        "err_revoice": "Error de re-doblaje",
        "err_story": "Error de ensamblaje de la historia",
        "fine_tuning_styles": "🎛️ Ajuste Fino de Estilos",
        "style_custom": "Estilo Personalizado",
        "style_custom_placeholder": "ej. Jerga de piratas, jerga de gamers...",
        "style_sarcasm": "Sarcasmo",
        "style_humor": "Humor",
        "style_tech": "Jerga Técnica",
        "style_formal": "Tono Formal",
        "api_key_prompt": "Por favor, introduzca su clave API de Fireworks (comienza con fw_...) en la barra lateral para comenzar."
    }
}


def main() -> None:
    st.set_page_config(
        page_title="GemmaForge: AI Re-Voicer & Video Generator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar preferences setup
    st.sidebar.header("⚙️ App Preferences")
    app_lang = st.sidebar.radio("Language / Idioma", ["English", "Español"], horizontal=True)
    lang_key = "en" if app_lang == "English" else "es"
    
    st.session_state["app_lang_key"] = lang_key
    
    def t(key: str) -> str:
        return TRANSLATIONS[lang_key].get(key, key)
        
    theme_choice = st.sidebar.radio("Theme / Tema", ["🌙 Dark / Oscuro", "☀️ Light / Claro"], horizontal=True)
    is_light = "Light" in theme_choice

    if is_light:
        bg_color = "#f9fafb"
        text_color = "#1f2937"
        card_bg = "#ffffff"
        border_color = "rgba(0, 0, 0, 0.1)"
        gpt_input_bg = "#ffffff"
        gpt_input_border = "#e5e7eb"
    else:
        bg_color = "#0f172a"
        text_color = "#f3f4f6"
        card_bg = "#1e293b"
        border_color = "rgba(255, 255, 255, 0.1)"
        gpt_input_bg = "#1e293b"
        gpt_input_border = "rgba(255, 255, 255, 0.1)"

    # Injecting native-friendly clean styling with theme configuration
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles override */
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        font-family: 'Inter', sans-serif;
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}

    /* Clean title gradient */
    .main-title {{
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}

    /* Safe styling for elements */
    div[data-testid="stExpander"], .stAlert {{
        border-radius: 8px !important;
        border: 1px solid {border_color} !important;
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}

    /* Primary Accent Button styling */
    div.stButton > button:first-child {{
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}

    /* ChatGPT styling for textarea */
    div[data-testid="stTextArea"] textarea {{
        border-radius: 12px !important;
        background-color: {gpt_input_bg} !important;
        border: 1px solid {gpt_input_border} !important;
        color: {text_color} !important;
        padding: 12px !important;
        font-size: 15px !important;
    }}

    /* Hide Streamlit Deploy button, header decoration and triple dots */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none !important;}}
    div[data-testid="stDecoration"] {{display: none !important;}}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🎬 GemmaForge</h1>', unsafe_allow_html=True)
    st.markdown(f"### {t('subtitle')}")
    st.caption(t('desc'))

    # API key setup
    api_key = get_api_key()

    # Глобальный фоновый трекер задач с анимацией (сбор информации)
    active_tasks = []
    if st.session_state.get("revoice_gen_status") == "running":
        active_tasks.append(("🔄 Ре-озвучка видео", "revoice_gen"))
    if st.session_state.get("story_gen_status") == "running":
        active_tasks.append(("✨ Создание видео с нуля", "story_gen"))

    # Сбор результатов по завершении задач
    if st.session_state.get("revoice_gen_status") == "completed":
        st.session_state["revoice_final_video"] = st.session_state.get("revoice_gen_result")
        st.session_state["revoice_gen_status"] = None
        st.toast("✅ Видео успешно переозвучено!", icon="🎉")

    if st.session_state.get("revoice_gen_status") == "failed":
        st.error(f"❌ Ошибка ре-озвучки: {st.session_state.get('revoice_gen_error')}")
        st.session_state["revoice_gen_status"] = None

    if st.session_state.get("story_gen_status") == "completed":
        st.session_state["story_final_video"] = st.session_state.get("story_gen_result")
        st.session_state["story_gen_status"] = None
        st.toast("✅ Видео успешно создано!", icon="🎉")

    if st.session_state.get("story_gen_status") == "failed":
        st.error(f"❌ Ошибка сборки ролика: {st.session_state.get('story_gen_error')}")
        st.session_state["story_gen_status"] = None

    # Sidebar settings
    st.sidebar.header(f"🛠️ {t('text_stylization')}")
    mode = st.sidebar.radio(
        t('mode_selector'),
        [
            t('mode_revoice'), 
            t('mode_story'),
            t('mode_analyzer')
        ]
    )
    
    LANG_MAPPING = {
        "🇷🇺 RU": "ru",
        "🇬🇧 EN": "en",
        "🇪🇸 ES": "es",
        "🇩🇪 DE": "de",
        "🇫🇷 FR": "fr"
    }
    lang_label = st.sidebar.radio(t('lang_voice'), list(LANG_MAPPING.keys()), horizontal=True)
    lang_code = LANG_MAPPING[lang_label]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(t('text_stylization'))
    weights = build_style_weights()

    st.session_state["current_mode"] = mode

    # Normalize mode comparison to localized mode titles
    if mode == t('mode_revoice'):
        st.header(f"🔄 {t('mode_revoice')}")
        st.write(t('analyzer_desc') if t('mode_revoice') == "Traducir / Re-doblaje de Video" else "Upload a source video. AI will transcribe the speech, analyze visual frames, and write 4 funny variant scripts for your re-voiced parody video.")
        
        video_path = select_or_upload_video("revoice")

        if video_path is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(t('original_video_header'))
                st.video(str(video_path))
                
            with col2:
                # Шаг 1: Анализ и генерация сценариев
                if st.button(t('step1_btn'), type="primary", use_container_width=True):
                    with st.spinner(t('step1_spinner')):
                        analysis = run_async(
                            asyncio.to_thread(
                                analyze_revoice_video,
                                video_path=str(video_path),
                                api_key=api_key
                            )
                        )
                        st.session_state["analysis"] = analysis
                        
                        concept = (
                            "Re-voice the uploaded video as a stronger short-form ad/parody. "
                            "Preserve the main visual beats from the source material."
                        )
                        if analysis["transcript"] and not analysis["transcript"].startswith("["):
                            concept += f"\n\nOriginal transcript:\n{analysis['transcript']}"
                            
                        revoice_script_variants = run_async(
                            asyncio.to_thread(
                                generate_script,
                                concept=concept,
                                visual_context=analysis["visual_context"],
                                weights=weights,
                                api_key=api_key
                            )
                        )
                        st.session_state["revoice_script_variants"] = revoice_script_variants
                        st.session_state["revoice_final_video"] = None

                # Выводим оригинальный транскрипт если он есть
                if st.session_state.get("analysis"):
                    st.info(f"🎙️ **{t('original_text')}** {st.session_state['analysis']['transcript']}")

                # Выводим 4 варианта сценария
                if st.session_state.get("revoice_script_variants"):
                    st.subheader(t('script_options'))
                    variants = st.session_state["revoice_script_variants"]["variants"]
                    
                    tabs = st.tabs([v["style"] for v in variants])
                    for idx, tab in enumerate(tabs):
                        with tab:
                            v = variants[idx]
                            st.markdown(f"### {v['title']}")
                            st.write(f"**Hook:** {v['hook']}")
                            st.write("---")
                            for s in v["scenes"]:
                                st.markdown(f"**{t('frame')} {s['scene_number']}**")
                                st.markdown(f"* Narration: *\"{s['narration']}\"*")
                                st.markdown(f"* Text: `{s['on_screen_text']}`")
                                st.markdown(f"* Frame description: *{s['image_prompt']}*")
                                st.write("")
                                
                            if st.button(f"{t('step2_btn')} '{v['style']}'", key=f"btn_rv_{idx}"):
                                run_in_background(
                                    "revoice_gen",
                                    assemble_revoice_video,
                                    script=v,
                                    original_video_path=st.session_state["analysis"]["source_video_path"],
                                    lang_code=lang_code,
                                    api_key=api_key
                                )
                                st.rerun()

                # Результат видео
                if st.session_state.get("revoice_final_video"):
                    st.success(t('success_gen'))
                    
                    # Center and scale down the video block for 9:16
                    v_col1, v_col2, v_col3 = st.columns([1, 1.2, 1])
                    with v_col2:
                        st.video(st.session_state["revoice_final_video"]["video_path"])
                    
                    # Скачивание дорожки озвучки отдельно
                    if "audio_path" in st.session_state["revoice_final_video"]:
                        st.subheader(t('audio_download'))
                        st.audio(st.session_state["revoice_final_video"]["audio_path"], format="audio/mp3")

                    # Красивый вывод текста
                    st.subheader(t('script_text'))
                    for s in st.session_state["revoice_final_video"]["script"]["scenes"]:
                        st.markdown(f"🎬 **{t('frame')} {s['scene_number']}:** *\"{s['narration']}\"*")

                    with st.expander(t('ai_metadata')):
                        st.json(st.session_state["revoice_final_video"]["script"])

    elif mode == t('mode_story'):
        st.header(f"✨ {t('mode_story')}")
        st.write("Write your story idea in GPT-style, choose video format, and upload an optional style reference image." if t('mode_story') == "Create Video from Story (From Scratch)" else "Escriba la idea de su historia en estilo GPT, elija el formato de video y suba una imagen de referencia de estilo opcional.")
        
        col_ref1, col_ref2 = st.columns([1, 1])
        with col_ref1:
            uploaded_style_img = st.file_uploader(
                f"{t('upload_style_img')}",
                type=["png", "jpg", "jpeg", "webp"],
                key="story_style_ref"
            )
        with col_ref2:
            aspect_label = st.radio(
                "Video Format (Aspect Ratio)" if lang_key == "en" else "Formato de Video (Relación de Aspecto)",
                ["9:16 (Shorts/TikTok/Reels)", "16:9 (YouTube)"],
                horizontal=True,
                key="aspect_radio"
            )
            aspect_ratio = ASPECT_OPTIONS[aspect_label.split(" ")[0]]

        # Analyze uploaded style reference image
        if uploaded_style_img is not None:
            if st.session_state.get("last_uploaded_img_name") != uploaded_style_img.name:
                with st.spinner("Analyzing image style..." if lang_key == "en" else "Analizando el estilo de la imagen..."):
                    try:
                        mime_type = "image/jpeg"
                        if uploaded_style_img.name.endswith(".png"):
                            mime_type = "image/png"
                        elif uploaded_style_img.name.endswith(".webp"):
                            mime_type = "image/webp"
                            
                        desc = describe_reference_image(
                            uploaded_style_img.getvalue(),
                            mime_type,
                            api_key
                        )
                        st.session_state["reference_style_description"] = desc
                        st.session_state["last_uploaded_img_name"] = uploaded_style_img.name
                        st.success(t('style_desc_success'))
                    except Exception as e:
                        st.error(f"Error reading image: {e}")
        else:
            st.session_state["reference_style_description"] = ""
            st.session_state["last_uploaded_img_name"] = None

        concept = st.text_area(
            "Describe your video idea (Prompt)" if lang_key == "en" else "Describa la idea de su video (Prompt)",
            placeholder=t('prompt_placeholder'),
            height=100,
            key="concept_input"
        )

        if concept.strip() and st.button(t('step1_story_btn'), type="primary", use_container_width=True):
            with st.spinner(t('step1_story_spinner')):
                final_concept = concept.strip()
                ref_style = st.session_state.get("reference_style_description", "")
                if ref_style:
                    final_concept += f"\n\nApply visual style and details from reference image:\n{ref_style}"
                    
                story_script_variants = run_async(
                    asyncio.to_thread(
                        generate_script,
                        concept=final_concept,
                        visual_context="No source video. Generate a fresh story-driven short-form video.",
                        weights=weights,
                        api_key=api_key
                    )
                )
                st.session_state["story_script_variants"] = story_script_variants
                st.session_state["story_final_video"] = None

        # Выводим 4 варианта сценария для режима с нуля
        if st.session_state.get("story_script_variants") and mode == t('mode_story'):
            st.subheader(t('script_options'))
            variants = st.session_state["story_script_variants"]["variants"]
            
            tabs = st.tabs([v["style"] for v in variants])
            for idx, tab in enumerate(tabs):
                with tab:
                    v = variants[idx]
                    st.markdown(f"### {v['title']}")
                    st.write(f"**Hook:** {v['hook']}")
                    st.write("---")
                    for s in v["scenes"]:
                        st.markdown(f"**{t('frame')} {s['scene_number']}**")
                        st.markdown(f"* Narration: *\"{s['narration']}\"*")
                        st.markdown(f"* Text: `{s['on_screen_text']}`")
                        st.markdown(f"* Frame description: *{s['image_prompt']}*")
                        st.write("")
                        
                    if st.button(f"{t('compile_story_btn')} '{v['style']}'", key=f"btn_story_{idx}"):
                        run_in_background(
                            "story_gen",
                            build_story_video,
                            script=v,
                            aspect_ratio=aspect_ratio,
                            lang_code=lang_code,
                            api_key=api_key,
                            style_description=st.session_state.get("reference_style_description", "")
                        )
                        st.rerun()

        # Результат видео
        if st.session_state.get("story_final_video") and mode == t('mode_story'):
            st.success(t('video_success'))
            
            # Reduce width of 9:16 vertical video players to look premium
            if aspect_ratio == "768x1344":
                v_col1, v_col2, v_col3 = st.columns([1, 1.2, 1])
                with v_col2:
                    st.video(st.session_state["story_final_video"]["video_path"])
            else:
                st.video(st.session_state["story_final_video"]["video_path"])
            
            # Красивый вывод текста
            st.subheader(t('script_text'))
            for s in st.session_state["story_final_video"]["script"]["scenes"]:
                st.markdown(f"🎬 **{t('frame')} {s['scene_number']}:** *\"{s['narration']}\"*")

            with st.expander(t('chosen_script')):
                st.json(st.session_state["story_final_video"]["script"])

    else:
        st.header(f"🔍 {t('mode_analyzer')}")
        st.write(t('analyzer_desc'))
        
        video_path = select_or_upload_video("analyzer")

        if video_path is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(t('original_video_header'))
                st.video(str(video_path))
                
            with col2:
                if st.button(t('analyzer_btn'), type="primary", use_container_width=True):
                    with st.spinner(t('analyzer_spinner')):
                        analysis = run_async(
                            asyncio.to_thread(
                                analyze_revoice_video,
                                video_path=str(video_path),
                                api_key=api_key
                            )
                        )
                        st.session_state["analysis"] = analysis
                
                if st.session_state.get("analysis"):
                    st.success(t('analyzer_success'))
                    
                    st.subheader(t('analyzer_transcript'))
                    transcript_text = st.session_state["analysis"]["transcript"]
                    if transcript_text and not transcript_text.startswith("["):
                        st.info(transcript_text)
                    else:
                        st.warning(t('no_speech'))
                        
                    st.subheader(t('analyzer_visual'))
                    for s in st.session_state["analysis"]["visual_context"]:
                        st.markdown(f"⏱️ **{t('second')} {s.get('timestamp_seconds', 0)} ({t('frame')} {s.get('frame_index', 0)}):**")
                        summary_data = s.get("summary", "")
                        if isinstance(summary_data, dict):
                            st.write(summary_data.get("summary", summary_data))
                        else:
                            st.write(summary_data)
                        st.markdown("---")

    # Отрисовка фонового трекера в самом конце страницы, чтобы не мешать загрузке контента
    active_tasks = []
    if st.session_state.get("revoice_gen_status") == "running":
        active_tasks.append((t('mode_revoice'), "revoice_gen"))
    if st.session_state.get("story_gen_status") == "running":
        active_tasks.append((t('mode_story'), "story_gen"))

    if active_tasks:
        st.markdown("---")
        with st.status(t('global_status_title'), expanded=True) as status:
            for task_name, task_key in active_tasks:
                log_msg = st.session_state.get(f"{task_key}_log", "Working...")
                status.write(f"**{task_name}**: {log_msg}")
            time.sleep(1.5)
            st.rerun()


if __name__ == "__main__":
    main()
