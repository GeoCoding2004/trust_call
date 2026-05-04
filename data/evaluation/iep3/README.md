# IEP3 Evaluation Dataset Layout

Place speaker-labeled evaluation audio under this folder when calibrating IEP3.

Expected layout:

```text
data/evaluation/iep3/
  speaker_001/
    sample_001.wav
    sample_002.wav
    sample_003.wav
  speaker_002/
    sample_001.wav
    sample_002.wav
    sample_003.wav
```

For VoxCeleb-style data, each top-level directory should be one speaker ID. Nested
folders under each speaker are fine; the evaluator searches recursively.

Supported audio extensions:

```text
.wav
.flac
.ogg
```

Run a quick subset evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_iep3.py `
  --dataset-root data\evaluation\iep3 `
  --max-speakers 20 `
  --max-files-per-speaker 5
```

Outputs are written to:

```text
trust_call_backend/state/evaluation/iep3/trials.csv
trust_call_backend/state/evaluation/iep3/summary.json
```

Do not commit downloaded datasets or generated evaluation outputs.
