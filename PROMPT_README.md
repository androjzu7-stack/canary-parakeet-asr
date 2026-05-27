# Prompt para generar el README del proyecto

Pega este prompt en Claude (o cualquier LLM) para generar el README completo del repositorio.

---

## PROMPT

Eres un experto en documentación técnica de proyectos de machine learning. Genera un archivo `README.md` completo y profesional para el siguiente proyecto de Google Colab. Usa GitHub-Flavored Markdown.

### Contexto del proyecto

**Título:** Implementación de modelos Transformer encoder–decoder para ASR y AST  
**Curso:** Procesamiento de Datos Secuenciales  
**Modelos:** Canary-1B-v2 y Parakeet-TDT-0.6B-v3 (NVIDIA)  
**Artículo base:** "Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST" (arXiv:2509.14128v2)

### Lo que hace el notebook

1. Instala NeMo Toolkit y dependencias (librosa, jiwer, soundfile, matplotlib)
2. Importa librerías y verifica GPU disponible (requiere T4 o A100 en Colab)
3. Carga los dos modelos preentrenados desde Hugging Face:
   - `nvidia/canary-1b-v2`: FastConformer encoder + Transformer Decoder, 25 idiomas europeos, ASR + AST
   - `nvidia/parakeet-tdt-0.6b-v3`: FastConformer encoder + TDT/RNN-T Decoder, 25 idiomas europeos, ASR
4. Carga y preprocesa audio WAV a mono 16 kHz con librosa
5. Visualiza la señal de audio, espectrograma Mel y STFT
6. Ejecuta inferencia ASR con ambos modelos y mide tiempo de inferencia
7. Compara resultados en una tabla lado a lado
8. Calcula la métrica WER (Word Error Rate) con jiwer
9. Genera un gráfico de radar comparando características de ambos modelos
10. Genera `app.py` con interfaz interactiva en Streamlit (sube audio, elige modelo, transcribe, calcula WER)
11. Lanza Streamlit en Colab con túnel cloudflared

### Datos técnicos importantes (del paper)

- Canary-1B-v2 entrenado en **1.7M horas** (Granary + NeMo ASR Set 3.0 con 227K h anotadas)
- Parakeet-TDT-0.6B-v3 entrenado en ~660K horas (subconjunto ASR de Canary)
- Canary es **10× más rápido** que Whisper-large-v3 en benchmarks de inglés
- Ambos modelos soportan los mismos **25 idiomas europeos** (en, es, de, fr, it, pt, ru, pl, nl, da, sv, fi, cs, sk, sl, hr, bg, ro, hu, lt, lv, et, mt, uk, el)
- Innovaciones clave: FastConformer (8× subsampling), TDT decoder, nGPT encoder experimental, NeMo Forced Aligner (timestamps), 36K h de audio sin habla para reducir alucinaciones

### Requisitos del README

Incluye las siguientes secciones **en este orden**:

1. **Badge row** — badges de: Python 3.10+, PyTorch, NeMo Toolkit, Google Colab (con link directo al notebook), licencia MIT
2. **Descripción** — 2-3 párrafos explicando qué hace el proyecto, los modelos usados y la relevancia académica
3. **Demo** — menciona la interfaz Streamlit y cómo acceder a ella en Colab
4. **Arquitectura** — tabla o diagrama ASCII comparando Canary vs Parakeet (encoder, decoder, parámetros, idiomas, tarea, velocidad vs Whisper)
5. **Requisitos** — lista de dependencias con versiones mínimas
6. **Instalación y uso** — instrucciones paso a paso para ejecutar en Google Colab (incluyendo cómo cambiar a GPU runtime)
7. **Estructura del notebook** — tabla con cada sección del notebook (número, título, descripción breve)
8. **Métricas y benchmarks** — referencia a los resultados del paper (WER en FLEURS, HuggingFace Open ASR Leaderboard, velocidad vs Whisper)
9. **Innovaciones destacadas del paper** — lista con FastConformer, TDT, nGPT, NFA, Granary dataset, audio sin habla
10. **Integrantes** — tabla con columnas # | Nombre | Rol (dejar como placeholder: `[Nombre]`)
11. **Referencias** — formato IEEE, citar el paper, Vaswani 2017, Gulati 2020 (Conformer), y los repos HuggingFace de ambos modelos
12. **Licencia** — MIT

### Restricciones de formato

- El título principal debe ser: `# Canary-1B-v2 & Parakeet-TDT-0.6B-v3 — ASR/AST con NeMo en Google Colab`
- Usa emojis moderadamente (solo en encabezados de sección)
- El README debe funcionar bien en GitHub (no uses HTML excepto para badges)
- Longitud objetivo: 300-500 líneas

Genera el README completo ahora.
