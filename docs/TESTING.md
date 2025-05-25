# Testování TicTacToe Aplikace

Dokumentace pro testování pomocí pytest frameworku.

## Přehled

Aplikace používá **čistý pytest framework** bez unittest závislostí.

## Požadavky

```bash
pip install pytest pytest-cov pytest-mock
```

## Spuštění testů

```bash
# Základní spuštění
python run_pytest.py

# S verbose výstupem
python run_pytest.py --verbose

# S coverage reportem  
python run_pytest.py --coverage

# Přímo přes pytest
python -m pytest tests/
```

## Struktura testů

```
tests/
├── test_detector_constants.py      # Detection konstanty
├── test_game_logic_pytest.py       # Herní logika  
├── test_frame_utils_pytest.py      # Frame utils
├── test_main_constants_pytest.py   # Main konstanty
├── test_config_helper_pytest.py    # Config helper
├── test_style_manager_pytest.py    # Style manager
└── test_utils_pure_pytest.py       # Utilities
```## Pytest formát

```python
"""Pure pytest tests for module."""
import pytest

class TestModuleName:
    """Pure pytest test class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create test data."""
        return {"test": "data"}
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        assert sample_data["test"] == "data"
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2), (2, 4), (3, 6)
    ])
    def test_parametrized(self, input, expected):
        """Test with parameters."""
        assert input * 2 == expected
    
    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ValueError):
            raise ValueError("Test error")
```

## Pytest vlastnosti

### Assertions
```python
assert value == expected
assert value is True
assert "substring" in text
assert 0.1 + 0.2 == pytest.approx(0.3)
```

### Mocking
```python
def test_with_mock(self, mocker):
    """Test using pytest-mock."""
    mock_func = mocker.patch('module.function')
    mock_func.return_value = "mocked"
    assert mock_func() == "mocked"
```

## Coverage Report

```
Name                              Stmts   Miss  Cover
-------------------------------------------------------
app/core/detector_constants.py      19      0   100%
app/core/utils.py                    26      1    96%
-------------------------------------------------------
TOTAL                             2719   2446    10%
```