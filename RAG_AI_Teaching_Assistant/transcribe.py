from faster_whisper import WhisperModel


def transcribe_video(video_path):

    print("Loading Whisper model...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("Transcribing video...")

    segments, info = model.transcribe(
        video_path,
        beam_size=5
    )

    transcript = ""

    for segment in segments:
        start = segment.start
        end = segment.end
        text = segment.text

        transcript += (
            f"[{start:.2f} - {end:.2f}] {text}\n"
        )

    return transcript