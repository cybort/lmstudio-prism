# Publishing

The local Cursor GitHub token cannot create repositories (403). Create empty public repos once in the browser, then push:

1. https://github.com/new → `lmstudio-prism` (empty, no README)
2. https://github.com/new → `lmstudio-js` (empty)
3. Optional: `lms` for the CLI submodule branch

```powershell
cd K:\project\lmstudio-prism
git remote add origin https://github.com/cybort/lmstudio-prism.git   # if missing
git push -u origin main

cd K:\project\lmstudio-js
git remote add origin https://github.com/cybort/lmstudio-js.git
git push -u origin main

cd K:\project\lmstudio-js\packages\lms-cli
git remote rename origin upstream
git remote add origin https://github.com/cybort/lms.git
git push -u origin cybort-prism-register-custom
```
