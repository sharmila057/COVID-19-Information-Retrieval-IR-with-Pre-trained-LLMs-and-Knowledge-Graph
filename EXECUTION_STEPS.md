# Execution Steps — COVID-19 Information Retrieval System

Follow these steps in order. Steps 1–3 are one-time setup. Steps 4–7 are also one-time
(per machine). Step 8 is what you run every time you want to use the app.

---

## 1. Unzip and open in VS Code
```bash
unzip covid19-ir-system.zip
cd covid19-ir-system
code .
```

## 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate            # Windows
```
In VS Code: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `./venv/bin/python`.

## 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 4. Set up MongoDB Atlas (required — the app won't start without this)
1. Free account: https://www.mongodb.com/cloud/atlas/register
2. Build a database → free **M0** tier → any region → Create
3. **Database Access** → Add New Database User → username + password → "Read and write to any database"
4. **Network Access** → Add IP Address → "Allow Access from Anywhere" (`0.0.0.0/0`) for local dev
5. **Database → Connect → Drivers → Python** → copy the connection string:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

## 5. Set up Gemini API (optional — only needed for the AI Assistant page)
1. https://aistudio.google.com/app/apikey → sign in → Create API Key
2. Copy the key

## 6. Create your `.env` file
```bash
cp .env.example .env
```
Edit `.env`:
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=covid_ir_system
GEMINI_API_KEY=your_key_here
```
Leave `GEMINI_API_KEY` blank if you're skipping step 5 — everything except the AI Assistant
still works.

**Never commit `.env` or paste its contents anywhere** — it's already excluded via `.gitignore`.

## 7. Build the FAISS index (one-time, slow step)
```bash
python build_index.py
```
Downloads the TREC-COVID dataset (~171k papers), cleans it, embeds it, and builds the FAISS
index. Takes 20–40+ minutes on a laptop CPU for the full corpus.

To test quickly first with a smaller subset:
```bash
python build_index.py --max-docs 5000
```

## 8. Launch the app (run this every time)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

### First-time use in the browser
1. **Register** tab → your name, email, password (typed twice) → Create Account
   *(the `users` collection starts empty — see `TEAM_ACCOUNTS.md` if multiple people are registering)*
2. **Login** tab → sign in
3. **Dashboard** → navigate via the cards or the sidebar:
   - 🔍 Search Research Papers
   - 📜 Search History
   - ⭐ Saved Papers
   - 🤖 AI Assistant
   - 👤 Profile
4. Run a search → results auto-save to history, each result has a **⭐ Save** button
5. Visit **AI Assistant** afterward to summarize / compare / explain / ask questions about
   the papers you just retrieved (requires `GEMINI_API_KEY`)

## 9. (Optional) Run the evaluation
```bash
python run_evaluation.py
```
Prints Precision@10, Recall@10, MAP, MRR, NDCG@10 and saves results to
`results/evaluation_results.json`.

---

## Troubleshooting Quick Reference

| Symptom | Fix |
|---|---|
| Error about `MONGO_URI` on login | `.env` missing/wrong, or your IP isn't allow-listed in Atlas (step 4.4) |
| `FileNotFoundError: FAISS index not found` | Run step 7 before step 8 |
| `OSError: en_core_web_sm not found` | Re-run `python -m spacy download en_core_web_sm` |
| `ImportError: faiss` | `pip install faiss-cpu` (not `faiss`) |
| AI Assistant shows a warning instead of working | `GEMINI_API_KEY` missing from `.env` |
| `BuilderConfig 'qrels' not found` | Make sure `requirements.txt` was reinstalled — this project uses the official `beir` library, not Hugging Face `datasets` |
| Slow embedding step | Use `--max-docs 5000` while developing |
