"""
Llama-powered Actuarial Intelligence Layer
Provides natural language explanations and insights for reinsurance pricing
"""

import json
import subprocess
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class LlamaActuarialIntelligence:
    """LLM-powered explanations for reinsurance pricing decisions"""
    
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        self.ollama_available = self._check_ollama()
        
    def _check_ollama(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def explain_premium_prediction(
        self, 
        premium: float,
        features: Dict[str, Any],
        feature_importance: pd.DataFrame
    ) -> str:
        """Generate natural language explanation for premium prediction"""
        
        if not self.ollama_available:
            return self._fallback_explanation(premium, features, feature_importance)
        
        # Prepare context for Llama
        top_features = feature_importance.head(5)
        
        prompt = f"""You are an expert reinsurance actuary. Explain this pricing decision in business terms:

PREDICTED PREMIUM: ${premium:,.0f}

KEY DRIVERS (by importance):
{self._format_features(top_features)}

TREATY DETAILS:
- Type: {features.get('treaty_type', 'Unknown')}
- Business Line: {features.get('business_line', 'Unknown')}
- Territory: {features.get('territory', 'Unknown')}
- Loss Ratio: {features.get('loss_ratio', 0):.1%}
- Combined Ratio: {features.get('combined_ratio', 0):.1%}

Provide a 3-4 sentence executive summary explaining:
1. Why this premium level is appropriate
2. The main risk factors driving the price
3. Any concerns or opportunities

Use clear business language, not technical jargon."""

        try:
            response = self._query_ollama(prompt)
            return response
        except Exception as e:
            return self._fallback_explanation(premium, features, feature_importance)
    
    def generate_underwriting_report(
        self,
        treaty_data: Dict,
        model_results: Dict,
        risk_factors: List[str]
    ) -> str:
        """Generate comprehensive underwriting report"""
        
        if not self.ollama_available:
            return self._generate_basic_report(treaty_data, model_results)
        
        prompt = f"""Generate a professional reinsurance underwriting report:

TREATY INFORMATION:
{json.dumps(treaty_data, indent=2)}

MODEL PREDICTIONS:
- Recommended Premium: ${model_results['premium']:,.0f}
- Expected Loss Ratio: {model_results['loss_ratio']:.1%}
- Confidence Interval: {model_results['confidence']:.1%}

KEY RISK FACTORS:
{chr(10).join(f'• {risk}' for risk in risk_factors)}

Create a formal underwriting memorandum with:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. PRICING RECOMMENDATION with rationale
3. KEY RISKS & MITIGANTS
4. TERMS & CONDITIONS recommendations
5. PORTFOLIO FIT assessment

Use professional reinsurance terminology."""

        try:
            response = self._query_ollama(prompt)
            return response
        except:
            return self._generate_basic_report(treaty_data, model_results)
    
    def answer_actuarial_question(self, question: str, context: Dict = None) -> str:
        """Answer actuarial questions about the data or models"""
        
        if not self.ollama_available:
            return "LLM not available. Please install Ollama for intelligent Q&A."
        
        context_str = json.dumps(context, indent=2) if context else "No specific context"
        
        prompt = f"""You are a senior reinsurance actuary. Answer this question accurately:

QUESTION: {question}

CONTEXT:
{context_str}

Provide a clear, concise answer using reinsurance industry best practices.
If you need more information, specify what data would help."""

        try:
            response = self._query_ollama(prompt)
            return response
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def explain_feature_importance(
        self, 
        feature_importance: pd.DataFrame,
        business_context: Dict
    ) -> str:
        """Explain why certain features matter for pricing"""
        
        if not self.ollama_available:
            return self._basic_feature_explanation(feature_importance)
        
        prompt = f"""Explain why these features are important for reinsurance pricing:

TOP FEATURES BY IMPORTANCE:
{feature_importance.head(10).to_string()}

BUSINESS CONTEXT:
- Industry: {business_context.get('industry', 'General')}
- Region: {business_context.get('region', 'Global')}
- Treaty Type: {business_context.get('treaty_type', 'Various')}

Explain in business terms:
1. Why each top feature matters for pricing
2. How they interact with each other
3. What patterns the model has learned
4. Actionable insights for underwriters

Use clear language that a business executive would understand."""

        try:
            response = self._query_ollama(prompt)
            return response
        except:
            return self._basic_feature_explanation(feature_importance)
    
    def suggest_portfolio_optimization(
        self,
        portfolio_metrics: Dict,
        risk_appetite: str = "moderate"
    ) -> str:
        """Suggest portfolio optimization strategies"""
        
        if not self.ollama_available:
            return "Portfolio optimization requires LLM. Install Ollama for recommendations."
        
        prompt = f"""As a reinsurance portfolio manager, analyze this portfolio and suggest optimizations:

PORTFOLIO METRICS:
{json.dumps(portfolio_metrics, indent=2)}

RISK APPETITE: {risk_appetite}

Provide specific recommendations for:
1. DIVERSIFICATION opportunities
2. PRICING adjustments needed  
3. CAPACITY allocation changes
4. RISK MITIGATION strategies
5. GROWTH opportunities

Be specific and actionable. Include estimated impact where possible."""

        try:
            response = self._query_ollama(prompt)
            return response
        except:
            return "Unable to generate optimization suggestions."
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama with the prompt"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"Ollama error: {result.stderr}")
        except subprocess.TimeoutExpired:
            return "LLM response timeout. Try a simpler query."
        except Exception as e:
            raise Exception(f"Failed to query Ollama: {str(e)}")
    
    def _fallback_explanation(
        self, 
        premium: float, 
        features: Dict,
        feature_importance: pd.DataFrame
    ) -> str:
        """Basic explanation when LLM not available"""
        top_feature = feature_importance.iloc[0]['feature'] if len(feature_importance) > 0 else "loss_ratio"
        
        return f"""
PREMIUM ANALYSIS:
• Recommended Premium: ${premium:,.0f}
• Primary Driver: {top_feature} ({feature_importance.iloc[0]['importance_pct']:.1f}% influence)
• Risk Level: {'High' if features.get('combined_ratio', 0) > 1.0 else 'Moderate'}
• Treaty Type: {features.get('treaty_type', 'Standard')}

Key Factors:
- Loss Ratio: {features.get('loss_ratio', 0):.1%}
- Business Line: {features.get('business_line', 'Unknown')}
- Territory: {features.get('territory', 'Unknown')}

Note: Install Ollama for detailed AI-powered explanations.
"""
    
    def _format_features(self, features_df: pd.DataFrame) -> str:
        """Format features for prompt"""
        lines = []
        for _, row in features_df.iterrows():
            lines.append(f"• {row['feature']}: {row['importance_pct']:.1f}% importance")
        return "\n".join(lines)
    
    def _generate_basic_report(self, treaty_data: Dict, model_results: Dict) -> str:
        """Generate basic report without LLM"""
        return f"""
UNDERWRITING REPORT
==================

Treaty: {treaty_data.get('treaty_id', 'Unknown')}
Type: {treaty_data.get('treaty_type', 'Unknown')}

PRICING RECOMMENDATION
• Premium: ${model_results['premium']:,.0f}
• Expected Loss Ratio: {model_results.get('loss_ratio', 0):.1%}
• Confidence: {model_results.get('confidence', 0):.1%}

RISK ASSESSMENT
• Combined Ratio: {treaty_data.get('combined_ratio', 0):.1%}
• Territory: {treaty_data.get('territory', 'Unknown')}
• Business Line: {treaty_data.get('business_line', 'Unknown')}

Note: Install Ollama for comprehensive AI-generated reports.
"""
    
    def _basic_feature_explanation(self, feature_importance: pd.DataFrame) -> str:
        """Basic feature explanation without LLM"""
        top_3 = feature_importance.head(3)
        
        explanation = "TOP PRICING FACTORS:\n\n"
        for _, row in top_3.iterrows():
            explanation += f"• {row['feature']}: {row['importance_pct']:.1f}% influence\n"
        
        explanation += "\nInstall Ollama for detailed explanations of feature interactions."
        return explanation


class StreamlitLlamaInterface:
    """Streamlit UI components for Llama integration"""
    
    @staticmethod
    def show_explanation_panel(st, llama_ai, premium, features, importance):
        """Show AI explanation panel in Streamlit"""
        with st.expander("🤖 AI Explanation (Powered by Llama)", expanded=True):
            if llama_ai.ollama_available:
                with st.spinner("Generating explanation..."):
                    explanation = llama_ai.explain_premium_prediction(
                        premium, features, importance
                    )
                st.write(explanation)
            else:
                st.warning("🔧 Install Ollama for AI explanations: `brew install ollama && ollama pull llama3.2`")
                st.write(llama_ai._fallback_explanation(premium, features, importance))
    
    @staticmethod
    def show_qa_interface(st, llama_ai, context=None):
        """Show Q&A interface in Streamlit"""
        st.subheader("🎓 Ask the Actuarial AI")
        
        if not llama_ai.ollama_available:
            st.info("💡 Install Ollama to enable AI Q&A: `brew install ollama && ollama pull llama3.2`")
            return
        
        # Sample questions
        sample_questions = [
            "Why is the loss ratio the most important feature?",
            "How should we price catastrophe treaties differently?",
            "What's driving the high premium for California property treaties?",
            "How do retention levels affect pricing?",
            "What portfolio optimizations would you recommend?"
        ]
        
        question = st.text_input(
            "Ask any actuarial question:",
            placeholder="e.g., How do catastrophe models affect pricing?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("Thinking..."):
                        answer = llama_ai.answer_actuarial_question(question, context)
                    st.markdown("### Answer:")
                    st.write(answer)
        
        with col2:
            if st.button("📝 Sample Questions"):
                st.write("**Try these:**")
                for q in sample_questions:
                    st.write(f"• {q}")
    
    @staticmethod
    def show_report_generator(st, llama_ai, treaty_data, model_results):
        """Generate and show underwriting report"""
        st.subheader("📄 AI-Generated Underwriting Report")
        
        if st.button("Generate Professional Report", type="primary"):
            if llama_ai.ollama_available:
                with st.spinner("Generating comprehensive report..."):
                    risk_factors = [
                        "High catastrophe exposure in California",
                        "Increasing frequency of weather events",
                        "Inflation impact on claims costs"
                    ]
                    
                    report = llama_ai.generate_underwriting_report(
                        treaty_data, model_results, risk_factors
                    )
                
                st.markdown(report)
                
                # Download button
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report,
                    file_name=f"underwriting_report_{treaty_data.get('treaty_id', 'unknown')}.md",
                    mime="text/markdown"
                )
            else:
                st.warning("Install Ollama for AI-generated reports")
                basic_report = llama_ai._generate_basic_report(treaty_data, model_results)
                st.text(basic_report)