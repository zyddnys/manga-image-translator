mit_detect_text_default_params = dict(
    detector_key="default",
    # mostly defaults from manga-image-translator/args.py
    detect_size=2560,
    text_threshold=0.5,
    box_threshold=0.7,
    unclip_ratio=2.3,
    invert=False,
    # device="cpu",
    gamma_correct=False,
    rotate=False,
    verbose=True,
)
