# Fieldnote sample: a small RAG pipeline

This project is a readable retrieval-augmented generation pipeline. It answers questions from a company policy that is split across two documents: a written billing page, and a spoken account-access policy. The first four stages run once, when the process starts. Every question runs the last four.

The store is a Python list on the `Rag` object. It is gone when the process exits. Nothing here is a database.

## What you need

- Python 3.11 or newer
- An OpenAI API key

The key is billed for three calls: transcription of the recording (Whisper), embeddings (`text-embedding-3-large`), and the answer (`gpt-6-astra`).

## Setup

From this directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```

Open `.env` and set `OPEN_AI_API_KEY`. Do not commit that file.

## Sample sources

`sources/` holds the only documents the pipeline is allowed to read. Every HTML file and every audio file in that folder is gathered. Audio is transcribed on startup.

| File | What it contains |
|---|---|
| `fieldnote_billing_and_cancellation.html` | Plans, invoices, cancellation, and refunds |
| `fieldnote_account_access.mp3` | Password reset, a missing reset email, a locked sign-in, and phone hours |

A question about a refund is answered from the page. A question about a reset link is answered from the recording. A question that mixes both has to use both.

Include `sources/` when you publish the repository. A clone with an empty `sources/` folder gathers nothing.

## Run

```bash
python main.py
```

The process gathers, splits, embeds, and stores before it accepts a question. Ask in the terminal. Type `exit()` to quit.

Startup transcribes the recording and embeds every chunk. That is the spend. Later questions embed the question and call the chat model. They do not transcribe the file again until you start the process again.
