# main.py---tu-orquestador

import json
import time
import uuid
import requests
from PIL import Image
import io
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"

def queue_prompt(workflow, prompt_text):
    # Carga el workflow base
    with open("flux_hanon_workflow.json", "r") as f:
        wf = json.load(f)
    
    # Inyecta tu prompt de la transcripción
    full_prompt = f"clean piano sheet music page, Hanon exercise, {prompt_text}, treble and bass clef, 70 bpm, professional engraving, high contrast, white background"
    wf["6"]["inputs"]["text"] = full_prompt
    wf["3"]["inputs"]["seed"] = uuid.uuid4().int & 0xFFFFFFFF

    p = {"prompt": wf, "client_id": str(uuid.uuid4())}
    res = requests.post(f"{COMFY_URL}/prompt", json=p)
    res.raise_for_status()
    return res.json()["prompt_id"]

def get_image(prompt_id):
    # Espera a que termine
    while True:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in r and r[prompt_id]["status"]["completed"]:
            break
        time.sleep(1)
    
    # Descarga la última imagen generada
    history = r[prompt_id]
    img_data = history["outputs"]["9"]["images"][0]
    img_url = f"{COMFY_URL}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
    img_bytes = requests.get(img_url).content
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

def generate_hanon_image(prompt: str, logo_path: str, output: str):
    print(f"🎼 Generando partitura Flux: {prompt}")
    
    try:
        prompt_id = queue_prompt(None, prompt)
        base = get_image(prompt_id)
    except Exception as e:
        print(f"⚠️ ComfyUI no responde ({e}), uso placeholder")
        base = Image.new("RGBA", (1688, 2384), "white")

    # Pega tu logo
    try:
        logo = Image.open(logo_path).convert("RGBA")
        logo.thumbnail((380, 380))
        base.paste(logo, (base.width - logo.width - 40, base.height - logo.height - 40), logo)
    except FileNotFoundError:
        pass

    Path(output).parent.mkdir(exist_ok=True)
    base.save(output, dpi=(300, 300))
    print(f"✅ Partitura guardada: {output}")
    return output
    
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import re

# CONFIG HANON
SAMPLE_RATE = 44100

def parse_hanon_from_text(texto: str):
    """Saca tonalidad y ejercicio del texto transcrito"""
    texto = texto.lower()
    # Ejemplo: "hazme hanon 1 en do mayor mano derecha"
    ejercicio = 1
    m = re.search(r'hanon\s*(\d+)', texto)
    if m:
        ejercicio = int(m.group(1))
    
    # Por ahora todos en Do Mayor, luego lo mapeamos a más tonalidades
    return ejercicio

def create_click_track(bpm: int, duration_sec: int, beats_per_bar=4):
    """Click con acento en 1 y 3 - estilo Hanon"""
    total_beats = int((duration_sec / 60) * bpm)
    beat_interval = SAMPLE_RATE * 60 // bpm
    
    # Click agudo (acento) y grave (no acento)
    def make_click(freq, length_ms=80, volume=0.8):
        t = np.linspace(0, length_ms/1000, int(SAMPLE_RATE * length_ms/1000), False)
        click = np.sin(2 * np.pi * freq * t) * np.exp(-t*30) * volume
        return (click * 32767).astype(np.float32)

    click_accent = make_click(1200, 80, 1.0)  # TIEMPO 1 y 3
    click_normal = make_click(800, 50, 0.5)   # 2 y 4

    track = np.zeros(SAMPLE_RATE * duration_sec, dtype=np.float32)
    
    for i in range(total_beats):
        pos = i * beat_interval
        if pos + len(click_accent) >= len(track):
            break
        
        # Acento en 1 y 3
        is_accent = (i % beats_per_bar == 0) or (i % beats_per_bar == 2)
        click = click_accent if is_accent else click_normal
        
        track[pos:pos+len(click)] += click

    return track

def create_hanon_midi_pattern(exercise_num=1, bpm=70):
    """
    Aquí generamos el patrón Hanon 1 básico en Do.
    Patrón: Do-Re-Mi-Fa-Sol-Fa-Mi-Do...
    Lo exportamos como audio de piano simple con sine waves
    """
    # Notas en frecuencias (Do4 a Sol5)
    # C4=261Hz, D4=293, E4=329, F4=349, G4=392, A4=440, B4=493, C5=523
    hanon1 = [261, 293, 329, 349, 392, 349, 329, 261] * 4
    
    note_duration = 60 / bpm / 2  # Corcheas
    samples_per_note = int(SAMPLE_RATE * note_duration)
    
    audio = np.zeros(int(SAMPLE_RATE * 12), dtype=np.float32)
    idx = 0
    
    for freq in hanon1:
        if idx + samples_per_note >= len(audio):
            break
        t = np.linspace(0, note_duration, samples_per_note, False)
        # Piano simple: seno + armónico
        note = np.sin(2 * np.pi * freq * t) * 0.6 + np.sin(2 * np.pi * freq * 2 * t) * 0.2
        envelope = np.exp(-t * 3)  # decay piano
        note *= envelope
        audio[idx:idx+samples_per_note] += note * 0.4
        idx += samples_per_note
        
    return audio

def generate_practice_track(prompt: str, bpm: int, output: str):
    print(f"🎧 Generando práctica Hanon {bpm}bpm: {prompt}")
    
    exercise = parse_hanon_from_text(prompt)
    duration = 30  # 30 seg loop para práctica

    click = create_click_track(bpm, duration)
    piano = create_hanon_midi_pattern(exercise, bpm)

    # Mezcla: Click arriba, piano abajo
    # Alineamos largos
    min_len = min(len(click), len(piano))
    mix = np.zeros(min_len, dtype=np.float32)
    mix += click[:min_len] * 0.7
    mix += piano[:min_len] * 0.8

    # Normaliza
    mix = mix / np.max(np.abs(mix)) * 0.9
    mix_int16 = (mix * 32767).astype(np.int16)

    Path(output).parent.mkdir(exist_ok=True)
    write(output, SAMPLE_RATE, mix_int16)
    
    print(f"✅ Track listo con acento en 1 y 3: {output}")
    return output
