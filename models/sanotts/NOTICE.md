# Amy Small model provenance

These assets derive from Ampixa's `amy-en-1p1m` voice in
<https://huggingface.co/ampixa/sanoTTS/tree/c532a5d21c078a16cb633718e9182bfd71a5b760/amy-en-1p1m>.
The pinned repository model card declares `license: gpl-3.0` and links to the
sanoTTS GPL license. The package-specific README supplies no separate license.
Copyright (C) 2026 Ampixa. The upstream GPL text is reproduced unchanged in
`LICENSE`.

Modifications: the duration/acoustic FP16 tensors are widened to FP32 and packed
for `snt_front_f32`; the decoder is expressed as two fixed-window TensorFlow
subgraphs, calibrated to int8, given symmetric scalar constants, and co-compiled
for the Edge TPU. `prepare.py` and `export.py` reproduce these steps from the
pinned source package. No trained weights are fitted or fine-tuned here.

The pinned sanoTTS source is commit
`3de9f37cbfedb8a1edcc28f4c8139979c6fcf889`. Its README describes an MIT runtime
boundary, but `LICENSE.MIT` says its file list is exhaustive and does not list
`snt_front_f32.c` or `snt_front_f32.h`. This notice does not extend that grant.
The complete upstream licenses and dictionary attribution remain in the
`sanoTTS` submodule; see `LICENSE`, `LICENSE.MIT`, and
`pypkg/sanotts/g2p_data/NOTICE.md` there.
