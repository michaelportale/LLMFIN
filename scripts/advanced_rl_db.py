import os
import uuid
import json
import logging
from typing import Dict, Any, Optional, List

from scripts.database import DatabaseManager
from scripts.advanced_rl import HierarchicalRL, MultiAgentMarketSystem, ImitationLearning

logger = logging.getLogger(__name__)

class DatabaseIntegration:
    """Helper class for integrating RL algorithms with database storage."""
    
    def __init__(self, connection_uri: Optional[str] = None):
        """Initialize database integration.
        
        Args:
            connection_uri: MongoDB connection URI. If None, will use environment variable.
        """
        self.db_manager = DatabaseManager(connection_uri)
        self.db_manager.connect()
        
    def save_hierarchical_rl(self, agent: HierarchicalRL, model_id: Optional[str] = None) -> str:
        """Save HierarchicalRL agent metadata and history to database.
        
        Args:
            agent: HierarchicalRL agent
            model_id: Optional model ID (generated if not provided)
            
        Returns:
            str: Model ID
        """
        model_id = model_id or f"hierarchical_rl_{uuid.uuid4().hex[:8]}"
        
        # Extract parameters
        params = {
            'num_options': agent.num_options,
            'meta_model_type': agent.meta_model_type,
            'option_model_type': agent.option_model_type,
            'option_termination_proba': agent.option_termination_proba,
            'option_duration': agent.option_duration
        }
        
        # Extract metrics
        metrics = {}
        if agent.history['episode_rewards']:
            metrics['mean_reward'] = sum(agent.history['episode_rewards'][-10:]) / min(10, len(agent.history['episode_rewards']))
            metrics['max_reward'] = max(agent.history['episode_rewards']) if agent.history['episode_rewards'] else 0
            metrics['episodes'] = len(agent.history['episode_rewards'])
            
        # Store model metadata
        self.db_manager.store_model_metadata(
            model_id=model_id,
            model_type='HierarchicalRL',
            params=params,
            metrics=metrics
        )
        
        # Store training history
        self.db_manager.store_training_history(
            model_id=model_id,
            history=agent.history
        )
        
        logger.info(f"Saved HierarchicalRL agent with ID: {model_id}")
        return model_id
    
    def save_multi_agent_system(self, system: MultiAgentMarketSystem, model_id: Optional[str] = None) -> str:
        """Save MultiAgentMarketSystem metadata and history to database.
        
        Args:
            system: MultiAgentMarketSystem
            model_id: Optional model ID (generated if not provided)
            
        Returns:
            str: Model ID
        """
        model_id = model_id or f"multi_agent_{uuid.uuid4().hex[:8]}"
        
        # Extract parameters
        params = {
            'num_agents': system.num_agents,
            'agent_types': system.agent_types,
            'competitive_rewards': system.competitive_rewards,
            'shared_observations': system.shared_observations,
            'market_impact': system.market_impact
        }
        
        # Extract metrics
        metrics = {}
        for i in range(system.num_agents):
            if system.history['episode_rewards'][i]:
                metrics[f'agent_{i}_mean_reward'] = sum(system.history['episode_rewards'][i][-10:]) / min(10, len(system.history['episode_rewards'][i]))
                metrics[f'agent_{i}_max_reward'] = max(system.history['episode_rewards'][i]) if system.history['episode_rewards'][i] else 0
        
        # Store model metadata
        self.db_manager.store_model_metadata(
            model_id=model_id,
            model_type='MultiAgentMarketSystem',
            params=params,
            metrics=metrics
        )
        
        # Store training history
        # Convert history to appropriate format for db storage
        history = {
            'episode_rewards': system.history['episode_rewards'],
            'market_impacts': system.history['market_impacts']
        }
        self.db_manager.store_training_history(
            model_id=model_id,
            history=history
        )
        
        logger.info(f"Saved MultiAgentMarketSystem with ID: {model_id}")
        return model_id
    
    def save_imitation_learning(self, model: ImitationLearning, model_id: Optional[str] = None) -> str:
        """Save ImitationLearning model metadata and history to database.
        
        Args:
            model: ImitationLearning model
            model_id: Optional model ID (generated if not provided)
            
        Returns:
            str: Model ID
        """
        model_id = model_id or f"imitation_learning_{uuid.uuid4().hex[:8]}"
        
        # Extract parameters
        params = {
            'method': model.method,
            'model_type': model.model_type,
            'hidden_sizes': model.hidden_sizes,
            'learning_rate': model.learning_rate
        }
        
        # Extract metrics
        metrics = {}
        if model.history['validation_accuracy']:
            metrics['final_val_accuracy'] = model.history['validation_accuracy'][-1]
        if model.history['loss']:
            metrics['final_loss'] = model.history['loss'][-1]
        if model.history['episode_rewards']:
            metrics['mean_reward'] = sum(model.history['episode_rewards'][-10:]) / min(10, len(model.history['episode_rewards']))
            
        # Store model metadata
        self.db_manager.store_model_metadata(
            model_id=model_id,
            model_type='ImitationLearning',
            params=params,
            metrics=metrics
        )
        
        # Store training history
        self.db_manager.store_training_history(
            model_id=model_id,
            history=model.history
        )
        
        logger.info(f"Saved ImitationLearning model with ID: {model_id}")
        return model_id
    
    def store_evaluation_results(self, 
                               model_id: str, 
                               environment: str, 
                               results: Dict[str, Any]) -> str:
        """Store model evaluation results.
        
        Args:
            model_id: Model ID
            environment: Environment description
            results: Evaluation results
            
        Returns:
            str: ID of the stored evaluation
        """
        return self.db_manager.store_evaluation_results(
            model_id=model_id,
            environment=environment,
            results=results
        )
    
    def load_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            dict: Model metadata or None if not found
        """
        return self.db_manager.get_model_metadata(model_id)
    
    def load_training_history(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load training history.
        
        Args:
            model_id: Model ID
            
        Returns:
            dict: Training history or None if not found
        """
        result = self.db_manager.get_training_history(model_id)
        if result:
            return result.get('history')
        return None
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Update model metrics.
        
        Args:
            model_id: Model ID
            metrics: Updated metrics
            
        Returns:
            bool: Success status
        """
        return self.db_manager.update_model_metrics(model_id, metrics)
    
    def close(self) -> None:
        """Close database connection."""
        self.db_manager.close() 