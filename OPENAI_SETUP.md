# OpenAI API Setup for PriceRe Chat

## Quick Setup

1. **Get your OpenAI API key**:
   - Go to [platform.openai.com](https://platform.openai.com)
   - Sign up/login and navigate to **API Keys**
   - Click **"Create new secret key"**
   - Copy the key (starts with `sk-...`)

2. **Add to environment file**:
   ```bash
   # Copy example file (if first time setup)
   cp .env.example .env
   
   # Edit the .env file
   nano .env
   
   # Replace the placeholder with your actual key:
   OPENAI_API_KEY=sk-your-actual-api-key-here
   
   # Save and exit
   ```

3. **Start PriceRe**:
   ```bash
   streamlit run ui/comprehensive_pricing_platform.py --server.port 8501
   ```

## How It Works

- **With API Key**: Full GPT-4o mini powered conversations about reinsurance, data cleaning, and platform guidance
- **Without API Key**: Basic pattern matching responses (still functional)

## Security

- ✅ API key stored in `.env` file (excluded from git)
- ✅ No frontend input required
- ✅ Environment variables loaded automatically
- ✅ Secure server-side processing

## Cost

- GPT-4o mini is very affordable (~$0.15 per 1M input tokens)
- Typical chat session costs less than $0.01
- No charges when API key not provided (falls back to basic mode)

## Chat Features (with API key)

- **Reinsurance Expertise**: Treaty pricing, cat modeling, risk assessment
- **Data Analysis**: Intelligent insights about your specific datasets  
- **Platform Guidance**: Help with PriceRe workflow and features
- **Context Awareness**: Knows about your cleaning results and data structure