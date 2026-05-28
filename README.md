#### Worker

## Start worker

- Imperative to use WSL or Linux
- Must have a working redis server
- Put stuff into an .env file following the ./app/core/settings names

**Create virtual env**
python -m venv .venv

source .venv/bin/activate

**Install requirements**
pip install -r requirements.txt

**Start worker**
python3 -m app.main