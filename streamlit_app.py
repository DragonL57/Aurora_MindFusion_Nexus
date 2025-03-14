import os
import asyncio
import nest_asyncio
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
from rich.markdown import Markdown
from google import genai
from google.genai import types
from datetime import datetime

# -------------------------------------------------
# Configuration and Constants
# -------------------------------------------------
VERSION = "3.0.0"
AGENT_COLORS = {
    "decomposer": "#00BFFF",  # Deep Sky Blue
    "analyzer": "#32CD32",    # Lime Green
    "synthesizer": "#FF00FF"  # Magenta
}

# -------------------------------------------------
# Load environment variables and initialize Gemini API client
# -------------------------------------------------
load_dotenv()

def initialize_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Error: GEMINI_API_KEY not found in .env")
        st.stop()
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {str(e)}")
        st.stop()

# -------------------------------------------------
# API Interaction Functions
# -------------------------------------------------
async def async_generate_text_fast(
    client,
    prompt: str,
    system_instruction: str,
    max_tokens: int = 8192,
    temperature: float = 0.1,
    max_retries: int = 5,
    backoff_factor: float = 1.5
) -> str:
    """Generate text using the standard model (fast but less deep thinking)."""
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system_instruction
    )
    loop = asyncio.get_event_loop()
    for attempt in range(1, max_retries + 1):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",  # Fast model
                    contents=[prompt],
                    config=config
                )
            )
            return response.text.strip()
        except Exception as e:
            if attempt == max_retries:
                st.error(f"Error generating content after {attempt} attempts: {str(e)}")
                return f"An error occurred while processing your request: {str(e)}"
            else:
                wait_time = backoff_factor * attempt
                await asyncio.sleep(wait_time)
    return "An unknown error occurred while processing your request."

async def async_generate_text_thinking(
    client,
    prompt: str,
    system_instruction: str,
    progress_bar=None,
    status_text=None,
    max_tokens: int = 65536,
    temperature: float = 0.1,
    max_retries: int = 5,
    backoff_factor: float = 1.5
) -> str:
    """Generate text using the thinking model for deeper analysis."""
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system_instruction
    )
    loop = asyncio.get_event_loop()
    
    # Update UI to show thinking is happening
    if progress_bar and status_text:
        status_text.text("🧠 Thinking deeply...")
        progress_bar.progress(0.2)
    
    for attempt in range(1, max_retries + 1):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",  # Thinking model
                    contents=[prompt],
                    config=config
                )
            )
            # Update UI to show thinking is complete
            if progress_bar and status_text:
                status_text.text("✅ Thinking complete")
                progress_bar.progress(1.0)
            return response.text.strip()
        except Exception as e:
            if attempt == max_retries:
                st.error(f"Error generating content after {attempt} attempts: {str(e)}")
                return f"An error occurred: {str(e)}"
            else:
                wait_time = backoff_factor * attempt
                if progress_bar and status_text:
                    status_text.text(f"Attempt {attempt}/{max_retries} failed. Retrying in {wait_time} seconds...")
                    progress_bar.progress(0.1 * attempt)
                await asyncio.sleep(wait_time)
    return "An unknown error occurred while processing your request."

def get_current_datetime() -> str:
    """Return the current date and time as a formatted string."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------------------------
# Agent Classes
# -------------------------------------------------
class QueryDecomposer:
    """Agent responsible for breaking down the user query into fundamental sub-questions."""
    
    def __init__(self):
        self.name = "Query Decomposer"
        self.role = "Breaking queries into fundamental components"
    
    async def decompose(self, client, query: str, conversation_history: str, progress_bar=None, status_text=None) -> List[str]:
        """Decompose a query into its fundamental sub-questions using first principles."""
        prompt = (
            f"Current date and time: {get_current_datetime()}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"# ORIGINAL QUERY FROM USER:\n{query}\n\n"
            "# TASK:\n"
            "Decompose this query into its most fundamental sub-questions that must be answered to fully address the original query. "
            "Use first principles thinking to break it down into core components. Ensure that each sub-question includes sufficient context "
            "so that someone reading it later understands the background of the query.\n\n"
            "Rules:\n"
            "1. Identify 2-7 distinct sub-questions that are essential to answer the main query.\n"
            "2. Each sub-question should address a fundamental, distinct aspect of the query.\n"
            "3. The sub-questions should collectively cover all aspects needed to comprehensively answer the original query.\n"
            "4. Focus on the underlying principles, assumptions, and core components.\n"
            "5. Format your response as a numbered list of sub-questions only.\n"
            "6. Do not explain your reasoning, just provide the sub-questions.\n"
            "7. Use language consistent with the original query.\n"
            "8. Include enough context in each sub-question so that the analysis agent can understand the original query without additional information."
        )
        
        system_instruction = (
            "You are the Query Decomposer, an expert at breaking complex queries into their most fundamental components "
            "using first principles thinking. Your goal is to identify the essential sub-questions that must be answered "
            "to fully address the original query."
        )
        
        if status_text:
            status_text.text("🔄 Decomposing query into fundamental components...")
        
        result = await async_generate_text_thinking(
            client,
            prompt, 
            system_instruction, 
            temperature=0.1,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        sub_questions = []
        for line in result.strip().split('\n'):
            cleaned_line = line.strip()
            if cleaned_line and (cleaned_line[0].isdigit() and cleaned_line[1:3] in ['. ', ') '] or 
                                 '?' in cleaned_line or 
                                 cleaned_line.startswith("What") or
                                 cleaned_line.startswith("How") or
                                 cleaned_line.startswith("Why") or
                                 cleaned_line.startswith("When") or
                                 cleaned_line.startswith("Where") or
                                 cleaned_line.startswith("Which") or
                                 cleaned_line.startswith("Who")):
                if cleaned_line[0].isdigit() and cleaned_line[1:3] in ['. ', ') ']:
                    cleaned_line = cleaned_line[3:].strip()
                sub_questions.append(cleaned_line)
        
        if not sub_questions:
            sub_questions = [s.strip() for s in result.strip().split('\n') if s.strip()]
        
        return sub_questions

class FirstPrinciplesAnalyzer:
    """Agent responsible for deeply analyzing each sub-question using first principles reasoning."""
    
    def __init__(self):
        self.name = "Deep Analysis Agent"
        self.role = "Analyzing sub-questions using first principles"
    
    async def analyze(self, client, original_query: str, sub_question: str, conversation_history: str, progress_bar=None, status_text=None) -> str:
        """Analyze a sub-question using first principles thinking."""
        prompt = (
            f"Current date and time: {get_current_datetime()}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"# ORIGINAL QUERY FROM USER:\n{original_query}\n\n"
            f"# SUB-QUESTION TO ANALYZE (with context):\n{sub_question}\n\n"
            "# TASK:\n"
            "Perform a deep, first-principles analysis of this sub-question, taking into account the full conversation context. "
            "Start from fundamental truths and build up a comprehensive understanding step by step.\n\n"
            "Your analysis should:\n"
            "1. Identify the core principles, concepts, and assumptions relevant to this sub-question.\n"
            "2. Examine the sub-question from multiple perspectives in the context of the conversation history.\n"
            "3. Consider the underlying mechanisms and causal relationships.\n"
            "4. Build logical connections between fundamental concepts.\n"
            "5. Provide concrete examples or evidence when relevant.\n"
            "6. Reach a clear conclusion that directly addresses the sub-question.\n\n"
            "Format your response as a clear, logical analysis that builds from first principles."
        )
        
        system_instruction = (
            "You are the First Principles Analyzer, an expert at examining questions by breaking them down into their most "
            "fundamental components and building understanding from basic truths. Your analysis should be clear, logical, "
            "and directly address the specific sub-question while considering the full conversation context."
        )
        
        if status_text:
            status_text.text(f"🔍 Analyzing: {sub_question}")
        
        result = await async_generate_text_thinking(
            client,
            prompt, 
            system_instruction, 
            temperature=0.1,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        return result

class SynthesisAgent:
    """Agent responsible for integrating all analyses into a coherent final answer."""
    
    def __init__(self):
        self.name = "Synthesis Agent"
        self.role = "Integrating analyses into a coherent answer"
    
    async def synthesize(self, client, original_query: str, sub_questions: List[str], analyses: List[str], progress_bar=None, status_text=None) -> str:
        """Synthesize the analyses of all sub-questions into a unified answer."""
        # Combine the sub-questions and their analyses
        analysis_sections = []
        for i, (sub_q, analysis) in enumerate(zip(sub_questions, analyses)):
            analysis_sections.append(f"## Analysis {i+1}: {sub_q}\n{analysis}")
        
        all_analyses = "\n\n".join(analysis_sections)
        
        prompt = (
            f"Current date and time: {get_current_datetime()}\n\n"
            f"# ORIGINAL QUERY FROM USER:\n{original_query}\n\n"
            "# ANALYSES OF SUB-QUESTIONS:\n"
            f"{all_analyses}\n\n"
            "# TASK:\n"
            "Synthesize these analyses into a single, coherent answer that directly addresses the original query. "
            "Your synthesis should:\n"
            "1. Integrate insights from all the analyses\n"
            "2. Connect the different perspectives and components in a logical way\n"
            "3. Resolve any contradictions or tensions between different analyses\n"
            "4. Build a comprehensive understanding that goes beyond the individual analyses\n"
            "5. Directly and explicitly answer the original query\n"
            "6. Be presented in a clear, well-structured format with appropriate sections and headings\n\n"
            "The final answer should be comprehensive yet unified, avoiding unnecessary repetition of the analyses.\n"
            "Your response MUST be in the same language as the user original query."
        )
        
        system_instruction = (
            "You are the Synthesis Agent, an expert at integrating multiple analyses into unified, coherent answers. "
            "Your goal is to create a synthesis that directly addresses the original query by building on the analyses "
            "of its fundamental components."
        )
        
        if status_text:
            status_text.text(f"🧩 Synthesizing all analyses into final answer...")
        
        result = await async_generate_text_thinking(
            client,
            prompt, 
            system_instruction, 
            temperature=0.1,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        return result

class MetaCognitiveEvaluatorAgent:
    """Agent for evaluating the reasoning quality of other agents' outputs."""
    
    def __init__(self):
        self.name = "Meta-Cognitive Evaluator Agent"
    
    async def evaluate(self, client, contributions: dict, conversation_history: str, progress_bar=None, status_text=None) -> str:
        """
        Critically assess whether each agent's reasoning strictly adheres to first principles,
        and provide actionable feedback for improvement.
        """
        contributions_text = "\n".join([f"{agent}: {output}" for agent, output in contributions.items()])
        prompt = (
            f"Current date and time: {get_current_datetime()}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"# AGENT CONTRIBUTIONS:\n{contributions_text}\n\n"
            "# TASK:\n"
            "Evaluate the above contributions for adherence to first principles. Identify any unchallenged assumptions or weaknesses, "
            "and provide actionable feedback for improvement."
        )
        system_instruction = (
            "You are the Meta-Cognitive Evaluator Agent. Critically analyze the outputs of other agents and suggest ways to refine their reasoning."
        )
        if status_text:
            status_text.text("🔍 Evaluating agent contributions for first principles adherence...")
        result = await async_generate_text_thinking(
            client,
            prompt, 
            system_instruction, 
            temperature=0.1,
            progress_bar=progress_bar,
            status_text=status_text
        )
        return result

# -------------------------------------------------
# Orchestrator
# -------------------------------------------------
class FirstPrinciplesOrchestrator:
    def __init__(self, client):
        self.client = client
        self.decomposer = QueryDecomposer()
        self.analyzer = FirstPrinciplesAnalyzer()
        self.synthesizer = SynthesisAgent()
        self.evaluator = MetaCognitiveEvaluatorAgent()
        self.conversation_history = ""

    async def process_query(self, query: str):
        """Process a user query with streamlit UI elements for progress tracking."""
        self.conversation_history += f"User: {query}\n"

        # Evaluate query complexity
        with st.spinner("Evaluating query complexity..."):
            evaluation_prompt = (
                f"Determine if the following query is simple or complex. "
                f"Answer only with 'simple' if it is a straightforward greeting or a very simple query, "
                f"otherwise respond with 'complex'.\n\nQuery: {query}"
            )
            evaluation_system_instruction = (
                "You are a query complexity evaluator. Analyze the query and respond with either 'simple' or 'complex'. "
                "Do not include any additional text."
            )
            evaluation_result = await async_generate_text_fast(
                self.client,
                evaluation_prompt, 
                evaluation_system_instruction, 
                temperature=0.1
            )
            evaluation_result = evaluation_result.strip().lower()

        # If simple, answer directly
        if evaluation_result == "simple":
            st.info("This appears to be a simple query. Generating a direct response...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            system_instruction = (
                "You are a helpful assistant. Provide a concise, friendly answer to the user's simple query."
            )
            answer = await async_generate_text_thinking(
                self.client,
                query, 
                system_instruction, 
                temperature=0.1,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            self.conversation_history += f"Assistant: {answer}\n"
            return answer, None, None, None  # Return only the answer for simple queries

        # For complex queries, run the full framework
        st.info("This is a complex query. Utilizing first principles thinking framework...")
        
        # Step 1: Decomposition
        with st.expander("Step 1: Query Decomposition", expanded=True):
            st.markdown("### Breaking query into fundamental components")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            sub_questions = await self.decomposer.decompose(
                self.client, query, self.conversation_history, progress_bar, status_text
            )
            
            st.markdown("#### Sub-questions:")
            for i, sub_q in enumerate(sub_questions):
                st.markdown(f"{i+1}. {sub_q}")
        
        # Step 2: Analysis (launching all API calls concurrently)
        analyses = []
        with st.expander("Step 2: Deep Analysis", expanded=True):
            st.markdown("### Analyzing each component using first principles")
            
            analysis_tasks = []
            progress_and_status = []
            for i, sub_q in enumerate(sub_questions):
                st.markdown(f"#### Analyzing sub-question {i+1}:")
                st.markdown(f"*{sub_q}*")
                analysis_progress = st.progress(0)
                analysis_status = st.empty()
                progress_and_status.append((analysis_progress, analysis_status))
                analysis_tasks.append(
                    self.analyzer.analyze(
                        self.client, query, sub_q, self.conversation_history, analysis_progress, analysis_status
                    )
                )
            # Run all analysis tasks concurrently
            analyses = await asyncio.gather(*analysis_tasks)
            
            for i, analysis in enumerate(analyses):
                st.markdown(f"**Analysis {i+1} Results:**")
                st.markdown(analysis)
        
        # Step 3: Synthesis
        with st.expander("Step 3: Synthesis", expanded=True):
            st.markdown("### Integrating analyses into a coherent answer")
            
            synthesis_progress = st.progress(0)
            synthesis_status = st.empty()
            
            synthesized_answer = await self.synthesizer.synthesize(
                self.client, query, sub_questions, analyses, synthesis_progress, synthesis_status
            )
            
            st.success("Preliminary synthesis complete!")
        
        # Step 4: Meta-cognitive evaluation
        with st.expander("Step 4: Meta-Cognitive Evaluation", expanded=True):
            st.markdown("### Evaluating reasoning quality and refining the answer")
            
            evaluation_progress = st.progress(0)
            evaluation_status = st.empty()
            
            contributions = {
                "Decomposer": "\n".join(sub_questions),
                "Analyzer": "\n\n".join(analyses),
                "Synthesis": synthesized_answer
            }
            
            evaluator_feedback = await self.evaluator.evaluate(
                self.client, contributions, self.conversation_history, evaluation_progress, evaluation_status
            )
            
            st.markdown("**Evaluator Feedback:**")
            st.markdown(evaluator_feedback)
        
        # Final Integration
        with st.expander("Step 5: Final Integration", expanded=True):
            st.markdown("### Creating the final answer")
            
            integration_progress = st.progress(0)
            integration_status = st.empty()
            
            integration_prompt = (
                f"Below is the preliminary synthesized answer:\n\n{synthesized_answer}\n\n"
                f"And here are additional insights from a meta-cognitive evaluation:\n\n{evaluator_feedback}\n\n"
                "Please integrate these insights into a single, coherent final answer that flows naturally and directly addresses the original query.\n\n"
                "You MUST use the same language as the user query, or follow the language that the user chose."
            )
            
            integration_instruction = (
                "You are the Final Integration Agent. Your task is to merge the preliminary answer with the evaluator insights "
                "into a unified final answer that is natural, coherent, and fully addresses the user's query."
            )
            
            final_answer = await async_generate_text_thinking(
                self.client,
                integration_prompt,
                integration_instruction,
                temperature=0.1,
                progress_bar=integration_progress,
                status_text=integration_status
            )
            
            self.conversation_history += f"Assistant: {final_answer}\n"
            
        return final_answer, sub_questions, analyses, evaluator_feedback

# -------------------------------------------------
# Streamlit App
# -------------------------------------------------
def main():
    # Apply nested asyncio to support async operations
    nest_asyncio.apply()
    
    # Set page config
    st.set_page_config(
        page_title="First Principles Thinking Framework",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize client as a session state variable if it doesn't exist
    if 'client' not in st.session_state:
        st.session_state.client = initialize_client()
    
    # Initialize conversation history as a list of message dictionaries
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Initialize orchestrator if it doesn't exist
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = FirstPrinciplesOrchestrator(st.session_state.client)
    
    # Sidebar with application info
    with st.sidebar:
        st.title("First Principles Thinking")
        st.markdown(f"**Version:** {VERSION}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This application uses a multi-step process to analyze queries using first principles thinking. "
            "It decomposes queries, analyzes sub-questions concurrently, synthesizes the results, and then refines the answer through meta-cognitive evaluation."
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        if st.button("Update API Key"):
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                st.session_state.client = initialize_client()
                st.session_state.orchestrator = FirstPrinciplesOrchestrator(st.session_state.client)
                st.success("API key updated successfully!")
            else:
                st.error("Please enter a valid API key")
        
        st.markdown("---")
        if st.button("Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.orchestrator.conversation_history = ""
            st.success("Conversation history cleared!")
    
    st.title("First Principles Thinking Framework")
    st.markdown("Chat with the assistant using first principles thinking. Type your query below.")
    
    # Chat input using st.chat_input if available; fallback to st.text_input otherwise
    if hasattr(st, "chat_input"):
        user_query = st.chat_input("Your message")
    else:
        user_query = st.text_input("Your message")
    
    if user_query:
        # Display the user's message in a chat bubble
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Processing your query..."):
            try:
                final_answer, sub_questions, analyses, evaluator_feedback = asyncio.run(
                    st.session_state.orchestrator.process_query(user_query)
                )
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "metadata": {
                        "sub_questions": sub_questions,
                        "analyses": analyses,
                        "evaluation": evaluator_feedback
                    } if sub_questions else None
                })
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display conversation history using chat messages
    if st.session_state.conversation_history:
        st.markdown("## Conversation History")
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    if message.get("metadata"):
                        with st.expander("See analysis details", expanded=False):
                            st.markdown("#### Sub-Questions")
                            for i, sub_q in enumerate(message["metadata"]["sub_questions"]):
                                st.markdown(f"{i+1}. {sub_q}")
                            
                            st.markdown("#### Analyses")
                            tabs = st.tabs([f"Analysis {i+1}" for i in range(len(message["metadata"]["analyses"]))])
                            for i, (tab, analysis) in enumerate(zip(tabs, message["metadata"]["analyses"])):
                                with tab:
                                    st.markdown(analysis)

if __name__ == "__main__":
    main()
