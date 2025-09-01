# API Keys Setup Guide

## Security First 🔒

**NEVER commit API keys to GitHub repositories.** This guide shows you how to securely configure API keys locally.

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual API keys:**
   ```bash
   # Edit this file and replace placeholders with your real keys
   nano .env
   ```

3. **Required API Keys:**

   ### OpenAI API (Required for PriceRe Chat)
   - Get key at: https://platform.openai.com/api-keys
   - Replace: `OPENAI_API_KEY=your-openai-api-key-here`
   - With: `OPENAI_API_KEY=sk-your-actual-openai-key`

   ### Optional APIs (for enhanced features)
   - **FRED API** (economic data): https://fred.stlouisfed.org/docs/api/api_key.html
   - **Alpha Vantage** (financial data): https://www.alphavantage.co/support/#api-key

## Verification

Test your OpenAI integration:
```bash
python3 test_openai_integration.py
```

You should see:
- ✅ API Key configured
- ✅ OpenAI package available  
- ✅ Chat engine created with model: gpt-4o-mini
- ✅ GPT-4o-mini connection successful

## Security Notes

- The `.env` file is automatically ignored by git (.gitignore)
- Never commit real API keys to version control
- Keep your API keys private and secure
- Rotate keys regularly for security

## Troubleshooting

**401 Unauthorized Error:**
- Check your OpenAI API key is correct
- Ensure you have sufficient OpenAI credits
- Verify the key has the right permissions

**Import Errors:**
- Install dependencies: `pip install -r requirements.txt`
- Make sure you're using Python 3.8+