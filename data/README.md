# External HOT3D data

Dataset files are intentionally excluded from Git.

For the current 3000-step tiny-overfit experiment, place the original archive
at:

```text
data/train_quest3/clip-000000.tar
```

Do not extract the tar. The dataset loader reads it directly. Other raw clips
belong in the same `data/train_quest3/` directory, while processed assets
belong under `data/train_quest3_processed/`.

Never commit `Hot3DAria_download_urls.json`; its signed download URLs are
account-specific and may be sensitive.
