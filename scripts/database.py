import os
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Union
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for storing and retrieving reinforcement learning results."""
    
    def __init__(self, connection_uri: Optional[str] = None):
        """Initialize the database manager.
        
        Args:
            connection_uri: MongoDB connection URI. If None, will use MONGODB_URI env variable.
        """
        self.connection_uri = connection_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/rl_results')
        self.client = None
        self.db = None
        
    def connect(self) -> bool:
        """Connect to the MongoDB database.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            self.client = MongoClient(self.connection_uri)
            # Test the connection
            self.client.admin.command('ping')
            
            # Get database name from URI
            db_name = self.connection_uri.split('/')[-1]
            self.db = self.client[db_name]
            
            logger.info(f"Connected to MongoDB: {self.connection_uri}")
            return True
        except (ConnectionFailure, OperationFailure) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
            
    def close(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
            
    def store_model_metadata(self, 
                           model_id: str, 
                           model_type: str, 
                           params: Dict[str, Any],
                           metrics: Optional[Dict[str, float]] = None) -> str:
        """Store model metadata.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'PPO', 'HierarchicalRL')
            params: Model parameters
            metrics: Performance metrics
            
        Returns:
            str: ID of the inserted document
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.models
        
        document = {
            'model_id': model_id,
            'model_type': model_type,
            'params': params,
            'metrics': metrics or {},
            'created_at': datetime.datetime.utcnow(),
            'updated_at': datetime.datetime.utcnow()
        }
        
        result = collection.insert_one(document)
        logger.info(f"Stored model metadata with ID: {result.inserted_id}")
        
        return str(result.inserted_id)
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Update model metrics.
        
        Args:
            model_id: Unique identifier for the model
            metrics: Updated performance metrics
            
        Returns:
            bool: True if update was successful
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.models
        
        result = collection.update_one(
            {'model_id': model_id},
            {
                '$set': {
                    'metrics': metrics,
                    'updated_at': datetime.datetime.utcnow()
                }
            }
        )
        
        success = result.modified_count > 0
        if success:
            logger.info(f"Updated metrics for model {model_id}")
        else:
            logger.warning(f"Failed to update metrics for model {model_id}")
            
        return success
    
    def store_training_history(self, 
                             model_id: str, 
                             history: Dict[str, List[Any]]) -> str:
        """Store training history.
        
        Args:
            model_id: Unique identifier for the model
            history: Training history (e.g., rewards, losses)
            
        Returns:
            str: ID of the inserted document
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.training_history
        
        # Check if history already exists
        existing_doc = collection.find_one({'model_id': model_id})
        
        if existing_doc:
            # Update existing document
            result = collection.update_one(
                {'model_id': model_id},
                {
                    '$set': {
                        'history': history,
                        'updated_at': datetime.datetime.utcnow()
                    }
                }
            )
            return str(existing_doc['_id'])
        else:
            # Insert new document
            document = {
                'model_id': model_id,
                'history': history,
                'created_at': datetime.datetime.utcnow(),
                'updated_at': datetime.datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored training history for model {model_id}")
            
            return str(result.inserted_id)
    
    def store_evaluation_results(self, 
                               model_id: str,
                               environment: str,
                               results: Dict[str, Any]) -> str:
        """Store model evaluation results.
        
        Args:
            model_id: Unique identifier for the model
            environment: Environment name/configuration
            results: Evaluation results
            
        Returns:
            str: ID of the inserted document
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.evaluations
        
        document = {
            'model_id': model_id,
            'environment': environment,
            'results': results,
            'timestamp': datetime.datetime.utcnow()
        }
        
        result = collection.insert_one(document)
        logger.info(f"Stored evaluation results for model {model_id}")
        
        return str(result.inserted_id)
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            dict: Model metadata or None if not found
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.models
        
        document = collection.find_one({'model_id': model_id})
        
        if document:
            # Convert ObjectId to string for JSON serialization
            document['_id'] = str(document['_id'])
            return document
        
        return None
    
    def get_training_history(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get training history.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            dict: Training history or None if not found
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.training_history
        
        document = collection.find_one({'model_id': model_id})
        
        if document:
            # Convert ObjectId to string for JSON serialization
            document['_id'] = str(document['_id'])
            return document
        
        return None
    
    def get_evaluation_results(self, 
                             model_id: str, 
                             environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation results.
        
        Args:
            model_id: Unique identifier for the model
            environment: Filter by environment name/configuration
            
        Returns:
            list: List of evaluation results
        """
        if not self.db:
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
                
        collection = self.db.evaluations
        
        query = {'model_id': model_id}
        if environment:
            query['environment'] = environment
            
        documents = list(collection.find(query))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc['_id'] = str(doc['_id'])
            
        return documents 