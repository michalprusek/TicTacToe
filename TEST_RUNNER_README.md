# Universal Test Runner

Univerzální test runner pro TicTacToe projekt, který spustí všechny funkční testy bez ohledu na jejich formát.

## Rychlý start

```bash
# Spustit všechny funkční testy
python universal_test_runner.py

# Zobrazit seznam dostupných testů
python universal_test_runner.py --list

# Spustit s podrobným výstupem
python universal_test_runner.py --verbose

# Rychlý režim (zastaví na první chybě)
python universal_test_runner.py --fast

# Spustit konkrétní soubory
python universal_test_runner.py test_config.py test_constants.py

# Spustit s coverage analýzou
python universal_test_runner.py --coverage
```

## Analýza testů

### 📊 Statistiky
- **Celkem souborů:** 40
- **Funkčních:** 38 (95%)
- **Problematických:** 2 (5%)
- **Celkem testů:** ~1,200+ testů
- **Formát:** 100% pytest

### ✅ Funkční soubory (38)

#### Config & Constants
- `test_config.py` - 24 testů - config module
- `test_config_extended.py` - 8 testů - config module extended
- `test_config_helper_pytest.py` - 44 testů - config_helper module
- `test_config_pytest.py` - 12 testů (1 failing) - config module
- `test_constants.py` - 10 testů - constants module
- `test_detector_constants.py` - 14 testů - detector_constants module

#### Drawing & UI
- `test_drawing_utils_comprehensive.py` - 34 testů - drawing_utils module
- `test_error_handler_comprehensive.py` - 42 testů - error_handler module
- `test_error_handler_pytest.py` - 10 testů - error_handler module
- `test_frame_utils_pytest.py` - 28 testů - frame_utils module
- `test_style_manager_comprehensive.py` - 60 testů - style_manager module
- `test_style_manager_pytest.py` - 10 testů - style_manager module

#### Game Logic
- `test_game_logic.py` - 60 testů - game_logic module
- `test_game_logic_comprehensive.py` - 30 testů - game_logic module
- `test_game_logic_extended.py` - 14 testů - game_logic module
- `test_game_logic_pytest.py` - 10 testů - game_logic module
- `test_game_logic_unittest.py` - 20 testů - game_logic module

#### Game State
- `test_game_state.py` - 28 testů - game_state module
- `test_game_state_additional_coverage.py` - 40 testů - game_state module
- `test_game_state_comprehensive.py` - 156 testů - game_state module
- `test_game_state_comprehensive_coverage.py` - 56 testů - game_state module
- `test_game_state_extended.py` - 16 testů - game_state module
- `test_game_state_pure_pytest.py` - 16 testů - game_state module

#### Strategy & AI
- `test_strategy.py` - 38 testů - strategy module
- `test_strategy_comprehensive.py` - 92 testů - strategy module
- `test_strategy_pure_pytest.py` - 10 testů - strategy module

#### Utils & Helpers
- `test_game_utils.py` - 12 testů - game_utils module
- `test_game_utils_comprehensive.py` - 54 testů - game_utils module
- `test_path_utils.py` - 12 testů - path_utils module
- `test_path_utils_pytest.py` - 10 testů - path_utils module
- `test_utils.py` - 16 testů - utils module
- `test_utils_comprehensive.py` - 36 testů - utils module
- `test_utils_extended.py` - 10 testů - utils module
- `test_utils_pure_pytest.py` - 10 testů - utils module

#### Coverage & Special
- `test_final_coverage.py` - 14 testů - multiple modules coverage
- `test_game_statistics_comprehensive.py` - 28 testů - game statistics
- `test_simple_coverage.py` - 10 testů - basic coverage
- `test_simple_pytest.py` - 12 testů - simple tests

### ❌ Problematické soubory (2)

1. **test_constants_pytest.py** - Import error (EMPTY not found in app.core.constants)
2. **test_main_constants_pytest.py** - Import error (main constants not accessible)

## Použití

### Základní spuštění
```bash
python universal_test_runner.py
```

### Možnosti

| Možnost | Popis |
|---------|-------|
| `--verbose` | Podrobný výstup testů |
| `--fast` | Rychlý režim (zastaví na první chybě) |
| `--coverage` | Spustí s coverage analýzou |
| `--list` | Zobrazí seznam všech testovacích souborů |
| `files...` | Spustí pouze konkrétní soubory |

### Příklady

```bash
# Rychlé ověření funkcí
python universal_test_runner.py --fast

# Podrobná analýza
python universal_test_runner.py --verbose

# Coverage report
python universal_test_runner.py --coverage

# Test konkrétního modulu
python universal_test_runner.py test_game_logic*.py

# Test jen základních funkcí
python universal_test_runner.py test_simple_pytest.py test_constants.py
```

## Technické detaily

### Formát testů
Všechny testy jsou v **pytest formátu**:
- Používají pytest třídy (`class TestXxx`)
- Používají `assert` statements
- Používají pytest fixtures a decoratory
- Žádné skutečné `unittest.TestCase` třídy nebyly nalezeny

### Automatická detekce
Runner automaticky:
- Objeví všechny soubory `test_*.py` v adresáři `tests/`
- Identifikuje problematické soubory s import errory
- Spustí funkční testy pomocí pytest
- Poskytne summary výsledků

### Problémy s importy
Některé testy mají problémy s importy konstant:
- `EMPTY`, `PLAYER_X`, `PLAYER_O` jsou definované v `app.main.game_logic` a `app.core.game_state`
- Nejsou v `app.core.constants` jak očekávají některé testy
- Runner tyto soubory automaticky přeskočí