# Running PaperScope on Windows (VS Code + PowerShell)

1. Install Python 3.9 or newer from https://www.python.org/downloads/ and tick "Add Python to PATH".
2. Extract `paper-search.zip`. Open the `paper-search` folder in VS Code (File > Open Folder).
3. Open a terminal: Terminal > New Terminal.
4. Create and turn on a virtual environment (the first line only the first time):
   ```
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.
5. Install the libraries (first time only): Flask, numpy, scipy and scikit-learn.
   ```
   py -m pip install -r requirements.txt
   ```
6. Start the app. The trained models are already in `data\models`, so there is nothing to download or train first.
   ```
   py app.py
   ```
   Wait for `Running on http://127.0.0.1:5000`, then open that address. Stop it with Ctrl + C.
   You need an internet connection: every new topic is looked up online (OpenAlex, then Semantic Scholar, then Crossref).
   OpenAlex needs a free API key once its shared budget is used up (https://help.openalex.org/api/authentication/).
   Set it, and your email, once per terminal before starting:
   `$env:OPENALEX_API_KEY = "your_key"` and `$env:OPENALEX_EMAIL = "you@example.com"`.
   Or, once: copy `.env.example` to `.env` in the paper-search folder and write your key after `OPENALEX_API_KEY=`.
   Without a key the search still works through Crossref.

## Other commands

| What | Command |
|---|---|
| Search from the terminal | `py scripts\search.py "graph neural networks for molecules"` |
| Two searches in one session (the second is answered from memory) | `py scripts\search.py "transformer attention machine translation" "attention transformers for translation"` |
| Run the tests (no internet needed) | `py -m unittest -v` |
| Retrain the models from the OpenAlex API (needs `OPENALEX_API_KEY`) | `py -m ml.train --online` |
| Evaluate the saved models on fresh OpenAlex queries | `py -m ml.evaluate --online --queries 300` |
| Evaluate on your own judgments (TSV: query, paper id, grade) | `py -m ml.evaluate --qrels my_judgments.tsv` |

Next time: open the folder in VS Code, run `.venv\Scripts\Activate.ps1`, then `py app.py`.
