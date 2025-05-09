#!/bin/bash

# Momentum Trading Application - Project Setup Script

echo "Setting up Momentum Trading Application..."

# Create main project structure
mkdir -p ./{config/strategies,src/{core,data,analysis,execution,monitoring,utils},tests,scripts,notebooks/research,data/{raw,processed,cache},logs,docs}

#cd momentum-trading

# Create Python package files
touch src/__init__.py
touch src/core/__init__.py
touch src/data/__init__.py
touch src/analysis/__init__.py
touch src/execution/__init__.py
touch src/monitoring/__init__.py
touch src/utils/__init__.py

# Create configuration files
cat > config/logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(timestamp)s %(level)s %(name)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: logs/momentum_trading.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  momentum_trading:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
EOF

# Create environment file template
cat > .env.example << 'EOF'
# Environment variables for Momentum Trading Application

# Environment
ENVIRONMENT=

# Database
DATABASE_URL=sqlite:///data/momentum_trading.db

# Redis (if using)
REDIS_URL=redis://localhost:6379/0

# Email settings
SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password

# Slack
SLACK_WEBHOOK_URL=your_slack_webhook

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Dashboard
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=secure_password
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
*.log
logs/
data/raw/*
data/processed/*
data/cache/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/cache/.gitkeep

# Jupyter Notebook
.ipynb_checkpoints/

# Testing
.coverage
.pytest_cache/
htmlcov/

# Documentation
docs/_build/
EOF

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/cache/.gitkeep
touch logs/.gitkeep

# Create README.md
cat > README.md << 'EOF'
# Momentum Trading Application

A comprehensive Python application for executing medium and long-term momentum trading strategies.

## Features

- **Multiple Data Sources**: Support for various financial data APIs
- **Flexible Strategy Framework**: Easily implement and test new momentum strategies
- **Risk Management**: Built-in risk controls and position sizing
- **Real-time Monitoring**: Dashboard and alert system
- **Backtesting**: Comprehensive backtesting with walk-forward analysis
- **Paper Trading**: Test strategies with simulated trading
- **Live Trading**: Integration with popular brokers

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/momentum-trading.git
   cd momentum-trading
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .[dev]
   ```

4. Copy and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Quick Start

1. Update market data:
   ```bash
   momentum-update --symbols SPY,QQQ,IWM
   ```

2. Run a backtest:
   ```bash
   momentum-backtest --strategy classic_momentum --start 2020-01-01 --end 2023-12-31
   ```

3. Start paper trading:
   ```bash
   momentum-trade --paper --strategy dual_momentum
   ```

4. Launch the monitoring dashboard:
   ```bash
   streamlit run src/monitoring/dashboard.py
   ```

## Project Structure

```
momentum-trading/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── core/           # Core components
│   ├── data/           # Data handling
│   ├── analysis/       # Analysis and indicators
│   ├── execution/      # Trade execution
│   └── monitoring/     # Monitoring and alerts
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── notebooks/          # Research notebooks
└── data/              # Data storage
```

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOF

# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203"]
EOF

# Make the script executable
chmod +x setup_project.sh

echo "Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. cd momentum-trading"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. pip install -e .[dev]"
echo "5. cp .env.example .env"
echo "6. Edit .env with your API keys"
echo "7. pre-commit install"
echo ""
echo "Happy coding!"
EOF
