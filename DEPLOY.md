# Deploy MedRisk

This folder IS the repository. Do not nest it inside another folder.

## 1. Test locally (optional, 1 min)

    pip install -r requirements.txt
    streamlit run app/app.py

Opens at http://localhost:8501

## 2. GitHub repo

Already created and pushed: https://github.com/RihamFouad/Prostackhup_medrisk
(main branch, remote "origin" already set up in this folder)

If you ever need to push again after local changes:

    git add .
    git commit -m "your message"
    git push

## 3. Deploy

1. share.streamlit.io
2. Sign in with GitHub, authorise access
3. New app -> Deploy a public app from GitHub
4. Repository:      RihamFouad/Prostackhup_medrisk
5. Branch:          main
6. Main file path:  app/app.py      <- change this, it defaults to streamlit_app.py
7. Deploy

First build takes 3-5 minutes. You get a URL like
https://prostackhup-medrisk.streamlit.app

## What the app needs at runtime

  app/app.py                      the interface
  models/medrisk_model.joblib     the trained pipeline + chosen threshold
  requirements.txt                dependencies
  .streamlit/config.toml          theme

If the build fails with FileNotFoundError, the .joblib did not get committed.
Check with: git ls-files models/

## Verified

Launched with streamlit before packaging: served HTTP 200.
