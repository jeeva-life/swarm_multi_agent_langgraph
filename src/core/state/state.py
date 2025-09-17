"""
State schema definition for LangGraph multi-agent system.
Defines the shared state that travels with conversations and enables agent coordination.
"""

from typing import TypedDict, Literal, Any, Optional, Dict, List
from typing_extensions import Annotated
from langgraph.graph import add_messages
# RemainingSteps is not available in current langgraph version


class State(TypedDict):
    """
    Holds the shared state for the multi-agent swarm system.
    
    This state object travels with the conversation and enables:
    - Context sharing between agents
    - Conversation history preservation
    - Agent coordination and handoffs
    - Customer session tracking
    - Loop prevention in cyclic workflows
    """
    
    # Unique identifier for the current customer session
    customer_id: str
    
    # Conversation history as a list of messages.
    # Annotated with `add_messages` so new messages are appended (not overwritten).
    messages: Annotated[List[Any], add_messages]
    
    # Tracks the remaining steps to avoid infinite loops 
    # when executing cyclic graphs in LangGraph.
    remaining_steps: int
    
    # Indicates which agent is currently active in the swarm.
    # This is required by `langgraph_swarm` for proper agent switching.
    active_agent: Literal[
        "rag_agent", 
        "nl2sql_agent", 
        "invoice_agent"
    ]
    
    # Additional context and metadata for enhanced coordination
    context: Dict[str, Any]
    
    # Current query being processed
    current_query: str
    
    # Agent-specific data that can be passed between agents
    agent_data: Dict[str, Any]
    
    # Session metadata
    session_metadata: Dict[str, Any]
    
    # Error handling and recovery information
    error_info: Optional[Dict[str, Any]]
    
    # Performance metrics for the current conversation
    performance_metrics: Dict[str, Any]


class StateManager:
    """
    Manages state operations and provides utilities for state manipulation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("state_manager")
    
    def create_initial_state(
        self, 
        customer_id: str, 
        initial_query: str,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> State:
        """
        Create initial state for a new conversation.
        
        Args:
            customer_id: Unique identifier for the customer
            initial_query: The initial user query
            session_metadata: Optional session metadata
            
        Returns:
            Initial state object
        """
        return State(
            customer_id=customer_id,
            messages=[],
            remaining_steps=10,  # Default step limit
            active_agent="rag_agent",  # Start with RAG agent
            context={},
            current_query=initial_query,
            agent_data={},
            session_metadata=session_metadata or {},
            error_info=None,
            performance_metrics={}
        )
    
    def update_active_agent(self, state: State, new_agent: str) -> State:
        """
        Update the active agent in the state.
        
        Args:
            state: Current state
            new_agent: New active agent identifier
            
        Returns:
            Updated state
        """
        state["active_agent"] = new_agent
        return state
    
    def add_message(self, state: State, message: Dict[str, Any]) -> State:
        """
        Add a message to the conversation history.
        
        Args:
            state: Current state
            message: Message to add
            
        Returns:
            Updated state
        """
        # The add_messages annotation will handle this automatically
        # This method is for explicit control if needed
        return state
    
    def update_context(self, state: State, key: str, value: Any) -> State:
        """
        Update context information in the state.
        
        Args:
            state: Current state
            key: Context key
            value: Context value
            
        Returns:
            Updated state
        """
        state["context"][key] = value
        return state
    
    def update_agent_data(self, state: State, agent_name: str, data: Dict[str, Any]) -> State:
        """
        Update agent-specific data in the state.
        
        Args:
            state: Current state
            agent_name: Name of the agent
            data: Data to store for the agent
            
        Returns:
            Updated state
        """
        state["agent_data"][agent_name] = data
        return state
    
    def set_error(self, state: State, error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> State:
        """
        Set error information in the state.
        
        Args:
            state: Current state
            error_type: Type of error
            error_message: Error message
            details: Additional error details
            
        Returns:
            Updated state
        """
        state["error_info"] = {
            "type": error_type,
            "message": error_message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        return state
    
    def clear_error(self, state: State) -> State:
        """
        Clear error information from the state.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        state["error_info"] = None
        return state
    
    def update_performance_metrics(self, state: State, metrics: Dict[str, Any]) -> State:
        """
        Update performance metrics in the state.
        
        Args:
            state: Current state
            metrics: Performance metrics to add
            
        Returns:
            Updated state
        """
        state["performance_metrics"].update(metrics)
        return state
    
    def get_conversation_history(self, state: State, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history from the state.
        
        Args:
            state: Current state
            limit: Optional limit on number of messages
            
        Returns:
            List of conversation messages
        """
        messages = state.get("messages", [])
        if limit:
            return messages[-limit:]
        return messages
    
    def get_agent_context(self, state: State, agent_name: str) -> Dict[str, Any]:
        """
        Get context specific to an agent.
        
        Args:
            state: Current state
            agent_name: Name of the agent
            
        Returns:
            Agent-specific context
        """
        return state.get("agent_data", {}).get(agent_name, {})
    
    def is_agent_active(self, state: State, agent_name: str) -> bool:
        """
        Check if a specific agent is currently active.
        
        Args:
            state: Current state
            agent_name: Name of the agent to check
            
        Returns:
            True if the agent is active
        """
        return state.get("active_agent") == agent_name
    
    def has_error(self, state: State) -> bool:
        """
        Check if the state contains error information.
        
        Args:
            state: Current state
            
        Returns:
            True if there is an error
        """
        return state.get("error_info") is not None
    
    def get_remaining_steps(self, state: State) -> int:
        """
        Get the remaining steps for the current conversation.
        
        Args:
            state: Current state
            
        Returns:
            Number of remaining steps
        """
        return state.get("remaining_steps", 0)
    
    def decrement_remaining_steps(self, state: State) -> State:
        """
        Decrement the remaining steps counter.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        current_steps = state.get("remaining_steps", 0)
        state["remaining_steps"] = max(0, current_steps - 1)
        return state
    
    def should_continue(self, state: State) -> bool:
        """
        Check if the conversation should continue based on remaining steps.
        
        Args:
            state: Current state
            
        Returns:
            True if conversation should continue
        """
        return state.get("remaining_steps", 0) > 0
    
    def get_state_summary(self, state: State) -> Dict[str, Any]:
        """
        Get a summary of the current state.
        
        Args:
            state: Current state
            
        Returns:
            State summary
        """
        return {
            "customer_id": state.get("customer_id"),
            "active_agent": state.get("active_agent"),
            "message_count": len(state.get("messages", [])),
            "remaining_steps": state.get("remaining_steps", 0),
            "has_error": self.has_error(state),
            "context_keys": list(state.get("context", {}).keys()),
            "agent_data_keys": list(state.get("agent_data", {}).keys()),
            "session_metadata": state.get("session_metadata", {})
        }


