# Deploy MedRisk

This folder IS the repository. Do not nest it inside another folder.

## 1. Test locally (optional, 1 min)

    pip install -r requirements.txt
    streamlit run app/app.py

Opens at http://localhost:8501

## 2. Create the GitHub repo

github.com/new -> name it exactly: ProStackHub_MedRisk
Set to PUBLIC. Do not tick "Add a README" (this folder has one).

## 3. Push

Open a terminal INSIDE this folder, then:

    git init
    git add .
    git commit -m "MedRisk Streamlit app"
    git branch -M main
    git remote add origin https://github.com/YOUR-USERNAME/ProStackHub_MedRisk.git
    git push -u origin main

## 4. Deploy

1. share.streamlit.io
2. Sign in with GitHub, authorise access
3. New app -> Deploy a public app from GitHub
4. Repository:      YOUR-USERNAME/ProStackHub_MedRisk
5. Branch:          main
6. Main file path:  app/app.py      <- change this, it defaults to streamlit_app.py
7. Deploy

First build takes 3-5 minutes. You get a URL like
https://prostackhub-medrisk.streamlit.app

## What the app needs at runtime

  app/app.py                      the interface
  models/medrisk_model.joblib     the trained pipeline + chosen threshold
  requirements.txt                dependencies
  .streamlit/config.toml          theme

If the build fails with FileNotFoundError, the .joblib did not get committed.
Check with: git ls-files models/

## Verified

Launched with streamlit before packaging: served HTTP 200.
