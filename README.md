# Robot TicTacToe

Aplikace pro hraní piškvorek s robotickým ramenem uArm Swift Pro pomocí počítačového vidění a AI.

## Požadavky

- Python 3.8+
- uArm Swift Pro (volitelné)
- Webkamera (volitelné)

## Instalace a spuštění

### Bash (Linux/Mac)

```bash
# Stáhnout z GitHub
git clone https://github.com/michalprusek/TicTacToe.git
cd TicTacToe

# Nainstalovat závislosti
pip install -r requirements.txt

# Spustit aplikaci
python -m app.main.main_pyqt

# nebo
python run.py
```

### PowerShell (Windows)

```powershell
# Stáhnout z GitHub
git clone https://github.com/michalprusek/TicTacToe.git
cd TicTacToe

# Nainstalovat závislosti
pip install -r requirements.txt

# Spustit aplikaci
python -m app.main.main_pyqt
```

## Volitelné parametry

```bash
# S volbou kamery a debug módem
python -m app.main.main_pyqt --camera 1 --debug --difficulty 7

# Kalibrace robota (pouze s uArm)
python -m app.calibration.calibration

# Spuštění testů
pytest

# coverage
pytest --cov=app --cov-report=term-missing

# spuštění lintingu
pylint app
```

## Funkce

- Detekce herní desky pomocí počítačového vidění
- AI protivník s nastavitelnou obtížností (1-10)
- Ovládání robotického ramene uArm Swift Pro (volitelné)
- PyQt5 grafické rozhraní

## Licence

MIT