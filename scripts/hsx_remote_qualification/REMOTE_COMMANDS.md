```bash
python scripts/hsx_remote_qualification/campaign.py verify-inputs --input-root .. --output work/hsx_clean_remote_v3
python scripts/hsx_remote_qualification/campaign.py preflight --input-root .. --output work/hsx_clean_remote_v3 --resolutions 32 48 64
python scripts/hsx_remote_qualification/campaign.py run --input-root .. --output work/hsx_clean_remote_v3 --resolutions 32 48 64 --workers 64
python scripts/hsx_remote_qualification/campaign.py validate --input-root .. --output work/hsx_clean_remote_v3 --resolutions 32 48 64
```
