import os
import time
from typing import Dict, List, Tuple, Callable, Optional
import streamlit as st
from dotenv import load_dotenv
import markdown
from google import genai
from google.genai import types

# -------------------------------------------------
# Configuration and Constants
# -------------------------------------------------
VERSION = "2.2.0"
AGENT_COLORS = {
    "critical": "#FF5555",
    "creative": "#FF55FF",
    "formal": "#5555FF",
    "counterfactual": "#FFFF55",
    "evidence": "#55FF55",
    "epistemological": "#55FFFF",
    "mathematical": "#55AAFF"
}

# Set page config early
st.set_page_config(
    page_title="Aurora MindFusion Nexus",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load environment variables and initialize Gemini API client
# -------------------------------------------------
load_dotenv()

def initialize_client():
    """Initialize the Gemini API client with proper error handling."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Error: GEMINI_API_KEY not found in .env")
        st.stop()
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {str(e)}")
        st.stop()

@st.cache_resource
def get_client():
    """Cached client initialization to avoid recreating on each rerun"""
    return initialize_client()

client = get_client()

# -------------------------------------------------
# Helper function: Call Gemini API to generate text
# -------------------------------------------------
def generate_text(prompt: str, system_instruction: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.1, progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> str:
    """Generate text using the Gemini API with enhanced error handling and live updates."""
    # If max_tokens is None, use a high limit (effectively removing truncation)
    if max_tokens is None:
        max_tokens = 10000

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system_instruction
    )
    
    try:
        if progress_bar:
            with progress_bar:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt],
                    config=config
                )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=config
            )
        
        return response.text.strip()
    except Exception as e:
        error_message = f"Error generating content: {str(e)}"
        st.error(error_message)
        return f"An error occurred while processing your request: {str(e)}"

# -------------------------------------------------
# New: Hierarchical Agent Teams and Dynamic Pairing
# -------------------------------------------------
class AgentGroup:
    def __init__(self, group_name: str, agents: List['Agent'], team_lead: 'Agent'):
        """
        group_name: Name of the group (e.g. "Logical Analysis Team")
        agents: List of agents in this group
        team_lead: Meta-agent responsible for consolidating the group's outputs
        """
        self.group_name = group_name
        self.agents = agents
        self.team_lead = team_lead

    def group_synthesize(self, contributions: Dict[str, str], query: str, 
                         progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> str:
        """
        Consolidate the group's contributions into a unified output.
        """
        bullet_points = "\n".join([
            f"- {agent.name}: {contributions.get(agent.name, '')}"
            for agent in self.agents
        ])
        prompt = (
            f"Group '{self.group_name}' contributions for the query '{query}':\n{bullet_points}\n\n"
            "Provide a consolidated response that reflects the team's consensus and key insights."
        )
        system_instr = (
            f"You are {self.team_lead.name}, the team lead for '{self.group_name}'. "
            "Consolidate your team's contributions into a coherent summary that addresses the user's query."
        )
        group_output = generate_text(prompt, system_instr, temperature=0.1, progress_bar=progress_bar)
        return group_output

def pair_agents(agents: List['Agent']) -> List[Tuple['Agent', 'Agent']]:
    """
    Pair agents dynamically. For simplicity, pair the first half with the second half.
    If an odd number, the last agent remains unpaired.
    """
    pairs = []
    n = len(agents)
    mid = n // 2
    for i in range(mid):
        pairs.append((agents[i], agents[n - 1 - i]))
    return pairs

# -------------------------------------------------
# New: Real-Time Collaboration - Agent Discussion
# -------------------------------------------------
def agent_discussion(contributions: Dict[str, str], query: str, agents: List['Agent'], 
                     progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> Dict[str, str]:
    """
    Have each agent "discuss" (vote on) their contribution.
    Returns a dict mapping agent names to their vote and brief comment.
    """
    votes = {}
    for agent in agents:
        prompt = (
            f"Based on your contribution:\n{contributions.get(agent.name, '')}\n"
            f"and considering the query '{query}', do you agree with this analysis? "
            "Answer 'yes' or 'no' and provide a one-sentence justification."
        )
        system_instr = (
            f"You are {agent.name}. Provide a brief vote ('yes' or 'no') and one sentence explanation."
        )
        vote = generate_text(prompt, system_instr, temperature=0.0, progress_bar=progress_bar)
        votes[agent.name] = vote
    return votes

# -------------------------------------------------
# Global Agent Templates
# -------------------------------------------------
# Mapping agent keys to a tuple of: (AgentClass, display name, role, domain, base_instruction)
AGENT_TEMPLATES = {
    "critical": (
        lambda **kwargs: Agent(**kwargs),
        "Logical Consistency Auditor",
        "Critical Reasoning Agent",
        "logical analysis and error detection",
        "Scrutinize the analysis for inconsistencies, fallacies, or contradictions. Focus directly on the user's query and address it specifically."
    ),
    "creative": (
        lambda **kwargs: Agent(**kwargs),
        "Imaginative Connector",
        "Creative Synthesis Agent",
        "creative and divergent thinking",
        "Generate novel insights, metaphors, and creative connections relevant to the query. Make sure to directly address what the user is asking."
    ),
    "formal": (
        lambda **kwargs: Agent(**kwargs),
        "Formal Proof Engineer",
        "Formal Logic Agent",
        "symbolic reasoning and formal derivations",
        "Formalize arguments and derive logical implications rigorously. Stay focused on the user's specific question and provide relevant reasoning."
    ),
    "counterfactual": (
        lambda **kwargs: Agent(**kwargs),
        "Hypothesis Expander",
        "Counterfactual Exploration Agent",
        "alternative scenario generation",
        "Propose 'what if' scenarios, alternative premises, and potential outcome variations that directly address the user's query."
    ),
    "evidence": (
        lambda **kwargs: Agent(**kwargs),
        "Evidence Weigher",
        "Evidence Evaluation Agent",
        "empirical validation and source critique",
        "Assess the strength, reliability, and relevance of evidence or assumptions. Focus on evidence that directly pertains to the user's specific question."
    ),
    "epistemological": (
        lambda **kwargs: Agent(**kwargs),
        "Perspective Shifter",
        "Epistemological Framing Agent",
        "multidisciplinary framing",
        "Re-contextualize the analysis from diverse philosophical, scientific, and social perspectives. Ensure these perspectives directly address what the user is asking about."
    ),
    "mathematical": (
        lambda **kwargs: Agent(**kwargs),
        "Mathematical Analyst",
        "Mathematical Reasoning Agent",
        "quantitative and symbolic reasoning",
        "Provide precise mathematical or quantitative reasoning where applicable. Focus on calculations and formalisms that directly answer the user's question."
    )
}

# -------------------------------------------------
# Base Agent Class
# -------------------------------------------------
class Agent:
    def __init__(self, name: str, role: str, domain: str, base_instruction: str):
        """
        name: Human-friendly name (e.g., "Logical Consistency Auditor")
        role: Agent type (e.g., "Critical Reasoning Agent")
        domain: Area of expertise (e.g., "logical analysis")
        base_instruction: Detailed instructions guiding agent behavior
        """
        self.name = name
        self.role = role
        self.domain = domain
        self.base_instruction = base_instruction
        self.last_contribution = ""
        self.agent_type = role.split()[0].lower()  # Extract first word of role for styling

    def produce_initial(self, query: str, progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> str:
        prompt = f"Question: {query}\nYour task: Analyze this query from your specialized perspective as {self.role}. Focus on directly answering what the user is asking."
        system_instr = (
            f"You are {self.name}, a {self.role} with expertise in {self.domain}. "
            f"{self.base_instruction} Keep your response focused on the user's query and provide specific, relevant insights."
        )
        
        self.last_contribution = generate_text(prompt, system_instr, temperature=0.1, progress_bar=progress_bar)
        return self.last_contribution

    def evaluate_peer(self, peer_output: str, query: str, 
                      progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> str:
        prompt = (
            f"Question: {query}\nPeer output: {peer_output}\n"
            "Evaluate this analysis and suggest improvements or highlight contradictions. "
            "Focus on whether the analysis directly addresses what the user is asking."
        )
        system_instr = (
            f"You are {self.name}, a {self.role} with expertise in {self.domain}. "
            f"{self.base_instruction} Evaluate specifically whether the peer's response adequately addresses the user's query."
        )
        
        evaluation = generate_text(prompt, system_instr, temperature=0.1, progress_bar=progress_bar)
        return evaluation

# -------------------------------------------------
# Final Integration Agent: Always used for synthesis
# -------------------------------------------------
class FinalIntegrationAgent(Agent):
    def synthesize(self, contributions: Dict[str, str], query: str, 
                   progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> str:
        bullet_points = "\n".join([f"- {agent_name}: {text}" for agent_name, text in contributions.items()])
        prompt = (
            f"Based on the following agent contributions for the query '{query}':\n{bullet_points}\n\n"
            "Produce a single, coherent, and unified final answer that integrates all perspectives and resolves conflicts. "
            "Most importantly, make sure your response directly addresses what the user is asking in their query."
        )
        system_instr = (
            f"You are {self.name}, a {self.role} with expertise in {self.domain}. "
            f"{self.base_instruction} Your primary responsibility is to ensure the final answer specifically addresses the user's query."
        )
        
        synthesis = generate_text(prompt, system_instr, temperature=0.1, progress_bar=progress_bar)
        return synthesis

# -------------------------------------------------
# Meta-Cognitive Orchestrator: Dynamically selects agents and orchestrates responses
# -------------------------------------------------
class MetaCognitiveOrchestrator:
    SIMPLE_QUERIES = {"hello", "hi", "hey", "good morning", "good evening"}

    def __init__(self):
        self.final_agent = FinalIntegrationAgent(
            name="Master Synthesizer",
            role="Final Integration Agent",
            domain="narrative fusion and final synthesis",
            base_instruction="Integrate all contributions into a coherent, unified answer reflecting deep multi-agent insights. Ensure the final response directly addresses the user's specific query."
        )
        # Initialize conversation history
        self.conversation_history = ""

    def is_simple_query(self, query: str) -> bool:
        normalized = query.strip().lower()
        if normalized in self.SIMPLE_QUERIES or len(normalized.split()) < 2:
            return True
        return False

    def dynamic_agent_selector(self, query: str, 
                              progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> List[str]:
        prompt = (
            f"Given the following query: '{query}', choose from the following agent types: "
            "critical, creative, formal, counterfactual, evidence, epistemological, mathematical. "
            "List the keys (comma separated) that are most relevant to answer this query. "
            "If the query is simple, respond with 'direct'. "
            "Provide only the comma-separated list of keys."
        )
        system_instr = (
            "You are the Dynamic Agent Selector. Analyze the query and output only the agent keys (from the available list) that are needed."
        )
        
        response = generate_text(prompt, system_instr, max_tokens=500, temperature=0.0, progress_bar=progress_bar)
        selected = [key.strip().lower() for key in response.split(",") if key.strip()]
        return selected

    def run(self, query: str, st_container):
        # Append new query to history
        self.conversation_history += f"User: {query}\n"
        full_query = f"{self.conversation_history}\nUser: {query}"
        
        # Create status placeholder
        status = st_container.empty()
        status.info("✨ Analyzing your query...")
        
        # Create placeholders for various components
        agent_selection_container = st_container.empty()
        contributions_container = st_container.empty()
        evaluations_container = st_container.empty()
        discussion_container = st_container.empty()
        final_answer_container = st_container.empty()
        
        # Check if simple query
        if self.is_simple_query(query):
            status.info("✨ Simple query detected; using direct response mode.")
            progress_bar = st_container.progress(0.0)
            direct_prompt = f"User said: {full_query}\nProduce a warm and helpful response."
            direct_system_instr = (
                "You are the Meta-Cognitive Orchestrator. Provide a clear and helpful answer for a simple query."
            )
            result = generate_text(direct_prompt, direct_system_instr, temperature=0.1, progress_bar=progress_bar)
            progress_bar.progress(1.0)
            final_answer_container.success("📝 Final Response")
            final_answer_container.markdown(result)
            status.empty()
            self.conversation_history += f"Assistant: {result}\n"
            return result

        # Select agents
        status.info("🔄 Selecting most relevant agents for your query...")
        progress_bar = st_container.progress(0.0)
        selected_keys = self.dynamic_agent_selector(full_query, progress_bar=progress_bar)
        agent_list = ", ".join(selected_keys) if selected_keys else "None"
        agent_selection_container.info(f"Selected agents: {agent_list}")
        progress_bar.progress(0.2)

        if "direct" in selected_keys:
            status.info("✨ 'direct' selected; using direct response mode.")
            direct_prompt = f"User said: {full_query}\nProduce a clear answer."
            direct_system_instr = (
                "You are the Meta-Cognitive Orchestrator. Provide a direct answer that specifically addresses the user's query."
            )
            result = generate_text(direct_prompt, direct_system_instr, temperature=0.1, progress_bar=progress_bar)
            progress_bar.progress(1.0)
            final_answer_container.success("📝 Final Response")
            final_answer_container.markdown(result)
            status.empty()
            self.conversation_history += f"Assistant: {result}\n"
            return result

        # Create the selected agents
        agents = []
        agent_types = {}
        for key in selected_keys:
            if key in AGENT_TEMPLATES:
                constructor, disp_name, role, domain, base_instr = AGENT_TEMPLATES[key]
                agent = constructor(
                    name=disp_name,
                    role=role,
                    domain=domain,
                    base_instruction=base_instr
                )
                agents.append(agent)
                agent_types[disp_name] = key
                
        if not agents:
            status.warning("⚠️ No agents selected; defaulting to direct response.")
            direct_prompt = f"User said: {full_query}\nProduce a clear, concise answer."
            direct_system_instr = "You are the Meta-Cognitive Orchestrator. Provide a direct answer."
            result = generate_text(direct_prompt, direct_system_instr, temperature=0.1, progress_bar=progress_bar)
            progress_bar.progress(1.0)
            final_answer_container.success("📝 Final Response")
            final_answer_container.markdown(result)
            status.empty()
            self.conversation_history += f"Assistant: {result}\n"
            return result

        # Initial agent contributions
        status.info("🔄 Divergent Phase - Gathering initial agent perspectives...")
        contributions = {}
        
        with st.expander("Agent Contributions", expanded=False):
            contribution_cols = st.columns(min(3, len(agents)))
            contribution_placeholders = [col.empty() for col in contribution_cols]
        
        for i, agent in enumerate(agents):
            agent_key = agent_types.get(agent.name, "default")
            progress_bar.progress(0.2 + (0.3 * (i / len(agents))))
            
            sub_progress = st_container.progress(0.0)
            sub_status = st_container.empty()
            sub_status.info(f"🔄 Agent {i+1}/{len(agents)}: {agent.name} is generating analysis...")
            
            contribution = agent.produce_initial(full_query, progress_bar=sub_progress)
            contributions[agent.name] = contribution
            
            # Update contribution display
            placeholder_index = i % len(contribution_placeholders)
            with contribution_placeholders[placeholder_index].container():
                st.subheader(agent.name)
                st.markdown(contribution)
            
            sub_status.empty()
            sub_progress.empty()
        
        progress_bar.progress(0.5)
        
        # Display contribution summary
        contributions_table = []
        for agent_name, content in contributions.items():
            agent_type = agent_types.get(agent_name, "default")
            color = AGENT_COLORS.get(agent_type.lower().split()[0], "white")
            contributions_table.append({
                "Agent": agent_name,
                "Type": agent_type,
                "Contribution": content[:100] + "..." if len(content) > 100 else content,
                "Color": color
            })
        
        contributions_container.success("✅ Initial agent contributions complete!")
        
        # Peer evaluations (if multiple agents)
        evaluations = {}
        if len(agents) > 1:
            status.info("🔄 Convergence Phase - Agents are evaluating each other's work...")
            agent_names = list(contributions.keys())
            num_agents = len(agents)
            
            with st.expander("Peer Evaluations", expanded=False):
                eval_cols = st.columns(min(2, num_agents))
                eval_placeholders = [col.empty() for col in eval_cols]
            
            for i, agent in enumerate(agents):
                peer_index = (i + 1) % num_agents
                peer_name = agent_names[peer_index]
                peer_output = contributions[peer_name]
                agent_key = agent_types.get(agent.name, "default")
                
                progress_bar.progress(0.5 + (0.2 * (i / num_agents)))
                sub_progress = st_container.progress(0.0)
                sub_status = st_container.empty()
                sub_status.info(f"🔄 Evaluation {i+1}/{num_agents}: {agent.name} evaluating {peer_name}...")
                
                evaluation = agent.evaluate_peer(peer_output, full_query, progress_bar=sub_progress)
                evaluations[f"{agent.name} → {peer_name}"] = evaluation
                
                # Update evaluation display
                placeholder_index = i % len(eval_placeholders)
                with eval_placeholders[placeholder_index].container():
                    st.subheader(f"{agent.name} evaluates {peer_name}")
                    st.markdown(evaluation)
                
                sub_status.empty()
                sub_progress.empty()
            
            evaluations_container.success("✅ Peer evaluations complete!")
        
        progress_bar.progress(0.7)
        
        # Agent discussion
        status.info("🔄 Agents are discussing their findings...")
        discussion_votes = agent_discussion(contributions, full_query, agents, progress_bar=st_container.progress(0.0))
        
        with discussion_container.expander("Agent Discussion", expanded=True):
            for agent, vote in discussion_votes.items():
                agent_type = agent_types.get(agent, "default").split()[0].lower()
                color = AGENT_COLORS.get(agent_type, "#FFFFFF")
                st.markdown(f"**{agent}:** {vote}")
        
        progress_bar.progress(0.8)
        
        # Final synthesis
        status.info("🔄 Final Synthesis Phase - Creating unified response...")
        synthesis_progress = st_container.progress(0.0)
        final_answer = self.final_agent.synthesize(contributions, full_query, progress_bar=synthesis_progress)
        progress_bar.progress(1.0)
        
        # Display final answer
        final_answer_container.success("📝 Final Synthesized Answer")
        final_answer_container.markdown(final_answer)
        
        # Clear status and progress
        status.empty()
        progress_bar.empty()
        
        # Append assistant's reply to conversation history
        self.conversation_history += f"Assistant: {final_answer}\n"
        return final_answer

# -------------------------------------------------
# Initialize session state if not already set
# -------------------------------------------------
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MetaCognitiveOrchestrator()
    st.session_state.conversation = []
    st.session_state.show_advanced = False

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("✨ Aurora MindFusion Nexus (AMN)")
st.caption(f"Dynamic Multi-Agent Orchestrator v{VERSION}")

# Sidebar with advanced options
with st.sidebar:
    st.header("About AMN")
    st.markdown("""
    Aurora MindFusion Nexus dynamically orchestrates multiple AI agents to provide 
    comprehensive and thoughtful responses to complex queries.
    
    Each agent brings unique perspectives and expertise to your questions.
    """)
    
    st.subheader("Agent Types")
    for key, (_, name, role, domain, _) in AGENT_TEMPLATES.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Role:** {role}")
            st.markdown(f"**Domain:** {domain}")
    
    st.divider()
    
    # API Key input
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", help="Enter your API key if you want to use your own key")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.session_state.orchestrator = MetaCognitiveOrchestrator()  # Reinitialize with new key
        st.success("API Key updated!")
    
    st.divider()
    
    # Advanced options toggle
    st.session_state.show_advanced = st.checkbox("Show Advanced Options", value=st.session_state.show_advanced)
    
    if st.session_state.show_advanced:
        st.subheader("Advanced Options")
        if st.button("Clear Conversation History"):
            st.session_state.conversation = []
            st.session_state.orchestrator.conversation_history = ""
            st.success("Conversation history cleared!")

# Display conversation history
for i, (role, text) in enumerate(st.session_state.conversation):
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Query input
query = st.chat_input("Ask anything...")
if query:
    # Add user message to conversation
    st.session_state.conversation.append(("user", query))
    st.chat_message("user").write(query)
    
    # Create a container for the assistant's response
    assistant_msg = st.chat_message("assistant")
    
    # Process the query
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.orchestrator.run(query, assistant_msg)
            # Response is displayed by the orchestrator directly in the assistant_msg container
            # We don't need to display it again
            st.session_state.conversation.append(("assistant", response))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display info at the bottom
st.divider()
st.caption("Aurora MindFusion Nexus | Powered by Gemini AI")
