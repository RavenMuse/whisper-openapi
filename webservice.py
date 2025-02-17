from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

app = FastAPI(
    title="whisper-openapi",
    description="Whisper ASR OpenAI API Server",
)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(
        default="transcribe", enum=["transcribe", "translate"]
    ),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(
                True if CONFIG.ASR_ENGINE == "faster_whisper" else False
            ),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(
            True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False
        ),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(
        default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]
    ),
):
    result = asr_model.transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/audio/transcriptions", tags=["Endpoints"])
async def transcriptions(
    file: UploadFile = File(...),  # noqa: B008
    encode: bool = Form(default=True, description="Encode audio first through ffmpeg"),
    language: Union[str, None] = Form(default=None, enum=LANGUAGE_CODES),
    prompt: Union[str, None] = Form(default=None),
    vad_filter: Annotated[
        bool | None,
        Form(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(
                True if CONFIG.ASR_ENGINE == "faster_whisper" else False
            ),
        ),
    ] = False,
    word_timestamps: bool = Form(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Form(
        default=False,
        description="Diarize the input",
        include_in_schema=(
            True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False
        ),
    ),
    min_speakers: Union[int, None] = Form(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Form(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    model: Union[str, None] = Form(default="whisper-1"),
    response_format: Union[str, None] = Form(
        default="json", enum=["txt", "vtt", "srt", "tsv", "json"]
    ),
):
    result = asr_model.transcribe(
        load_audio(file.file, encode),
        "transcribe",
        language,
        prompt,
        vad_filter,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        response_format,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(file.filename)}.{response_format}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(
        load_audio(audio_file.file, encode)
    )
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
