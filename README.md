# Canary-1B-v2 & Parakeet-TDT-0.6B-v3 — ASR/AST con NeMo en Google Colab

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo_Toolkit-76B900?logo=nvidia)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/androjzu7-stack/canary-parakeet-asr/blob/main/notebook_final_canary_parakeet.ipynb)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Descripción

Proyecto final del curso **Procesamiento de Datos Secuenciales**, donde se implementa un pipeline completo de reconocimiento automático del habla (ASR) y traducción automática de voz (AST) usando dos modelos preentrenados de NVIDIA: **Canary-1B-v2** y **Parakeet-TDT-0.6B-v3**.

Ambos modelos están basados en la arquitectura **FastConformer encoder** y fueron presentados en el artículo *"Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST"* (arXiv:2509.14128v2, NVIDIA, 2025). Canary-1B-v2 combina un encoder FastConformer con un Transformer Decoder para soportar ASR y AST en 25 idiomas europeos, mientras que Parakeet-TDT-0.6B-v3 usa un decoder RNN-T/TDT optimizado para alta velocidad de transcripción multilingüe.

El notebook cubre desde la instalación del entorno hasta la comparación de resultados con la métrica WER, e incluye una interfaz interactiva en **Streamlit** para demostración en tiempo real.

---

## 🎬 Demo

La celda de Streamlit genera `app.py` y lo expone mediante un túnel **cloudflared** directamente en Google Colab. La interfaz permite:

- Subir un archivo de audio WAV
- Seleccionar el modelo (Canary o Parakeet) y el idioma
- Obtener la transcripción y calcular el WER respecto a una referencia manual

---

## 🏗️ Comparativa de arquitecturas

| Característica | Canary-1B-v2 | Parakeet-TDT-0.6B-v3 |
|---|---|---|
| **Encoder** | FastConformer (8× subsampling) | FastConformer (8× subsampling) |
| **Decoder** | Transformer Decoder (autoregresivo) | RNN-T / TDT |
| **Parámetros** | ~1 000 M | ~600 M |
| **Tareas** | ASR + AST | ASR |
| **Idiomas** | 25 idiomas europeos | 25 idiomas europeos |
| **Velocidad vs Whisper-large-v3** | **10× más rápido** | Más rápido aún |
| **VRAM recomendada** | ≥ 8 GB | ~4 GB |
| **Dataset entrenamiento** | 1.7 M horas (Granary + NeMo ASR Set 3.0) | ~660 K horas (subconjunto ASR) |

---

## ⚙️ Requisitos

```
torch >= 2.0
nemo_toolkit[asr] >= 1.23
librosa >= 0.10
soundfile >= 0.12
matplotlib >= 3.7
pandas >= 2.0
jiwer >= 3.0
streamlit >= 1.30
```

**Entorno recomendado:** Google Colab con GPU T4 o A100 (Runtime → Change runtime type → GPU)

---

## 🚀 Instalación y uso

### En Google Colab (recomendado)

1. Haz clic en el badge **Open in Colab** de arriba.
2. Cambia el runtime a GPU: `Runtime → Change runtime type → T4 GPU`.
3. Ejecuta las celdas en orden de arriba a abajo.
4. En la **celda 15**, sube o especifica la ruta de tu archivo de audio WAV (mono, 16 kHz).
5. En la **celda 25** (WER), ajusta la variable `referencia` con el texto real del audio.
6. Para la demo Streamlit, ejecuta la **celda 30** (genera `app.py`) y luego la **celda 31** (lanza el túnel cloudflared).

### Localmente

```bash
pip install nemo_toolkit[asr] librosa soundfile matplotlib pandas jiwer streamlit
jupyter notebook "notebook_final_canary_parakeet.ipynb"
```

> **Nota:** la inferencia en CPU es muy lenta. Se recomienda fuertemente usar GPU.

---

## 📂 Estructura del notebook

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | Introducción | Motivación, aplicaciones de ASR/AST, artículo base y repositorios |
| 2 | Marco Teórico | Transformer encoder-decoder, atención, FastConformer, nGPT, NFA, dataset Granary |
| 3 | Metodología | Flujo de implementación y entorno recomendado |
| 4.1 | Instalación | PyTorch, NeMo Toolkit, librosa, jiwer |
| 4.2 | Importación de librerías | Imports y configuración de warnings |
| 4.3 | Verificación de GPU | Detecta GPU disponible con torch.cuda |
| 4.4 | Carga de modelos | Descarga Canary-1B-v2 y Parakeet-TDT-0.6B-v3 desde Hugging Face |
| 4.5 | Preprocesamiento de audio | Carga WAV a 16 kHz mono con librosa |
| 4.6 | Visualización | Waveform, espectrograma Mel y STFT |
| 4.7 | Inferencia | ASR con ambos modelos + medición de tiempo |
| 4.8 | Comparación de resultados | Tabla lado a lado de transcripciones |
| 4.9 | Métrica WER | Cálculo de Word Error Rate con jiwer |
| 5 | Innovaciones | Tabla detallada de innovaciones de cada modelo según el paper |
| 6 | Limitaciones | Comparativa de limitaciones de ambos modelos |
| 7 | Interfaz Streamlit | Genera `app.py` y lo expone con cloudflared en Colab |
| 8 | Conclusiones | Aprendizajes, limitaciones observadas y posibles mejoras |
| 9 | Referencias | Bibliografía en formato IEEE |

---

## 📊 Métricas y benchmarks (del paper)

- **Hugging Face Open ASR Leaderboard:** Canary-1B-v2 supera a Whisper-large-v3 en inglés siendo **10× más rápido**.
- **FLEURS (25 idiomas):** WER competitivo frente a modelos más grandes como Seamless-M4T-v2-large.
- **CoVoST2:** Resultados comparables a sistemas basados en LLM en tareas de traducción X→En y En→X.
- **WER inglés < 5%** en benchmarks estándar para ambos modelos.

---

## 🔬 Innovaciones destacadas del paper

- **FastConformer:** subsampling 8× con convoluciones separables en profundidad — 2-3× más rápido que Conformer estándar.
- **TDT Decoder:** Token-and-Duration Transducer, genera token y duración simultáneamente, ideal para streaming.
- **nGPT encoder (experimental):** normalización hiperesférica + ALiBi simétrico — primera aplicación en ASR/AST.
- **NeMo Forced Aligner (NFA):** timestamps a nivel de segmento con modelo CTC auxiliar, sin reentrenar el decoder.
- **Granary dataset:** 1.7M horas con balanceo en dos etapas (corpus α=0.5 + idioma β=0.5).
- **Audio sin habla:** 36 000 h de audio no-lingüístico para reducir alucinaciones heredadas de pseudo-etiquetado Whisper.

---

## 👥 Integrantes

| # | Nombre | Código |
|---|--------|--------|
| 1 | Santiago Londoño Méndez | 22602902 |
| 2 | Andrés Rojas Zúñiga | 22507348 |
| 3 | Rubén Darío García Morales | 22507004 |
| 4 | David Ayala Caro | 22507570 |

---

## 📚 Referencias

[1] M. Sekoyan, N. R. Koluguri, N. Tadevosyan, P. Zelasko, T. Bartley, N. Karpov, J. Balam, y B. Ginsburg, "Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST," arXiv:2509.14128v2, Sep. 2025.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, e I. Polosukhin, "Attention Is All You Need," en *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[3] A. Gulati *et al.*, "Conformer: Convolution-augmented Transformer for Speech Recognition," en *Proc. Interspeech*, 2020, pp. 5036–5040.

[4] D. Rekesh *et al.*, "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition," en *Proc. ASRU*, 2023.

[5] NVIDIA, "canary-1b-v2," Hugging Face. [En línea]. Disponible en: https://huggingface.co/nvidia/canary-1b-v2

[6] NVIDIA, "parakeet-tdt-0.6b-v3," Hugging Face. [En línea]. Disponible en: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

[7] NVIDIA, "NeMo Toolkit," GitHub. [En línea]. Disponible en: https://github.com/NVIDIA/NeMo

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
