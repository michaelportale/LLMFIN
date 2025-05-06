import numpy as np
import random
import copy
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.metrics import mean_squared_error
from datetime import datetime

class TradingStrategy:
    """
    Base class for trading strategies that can be evolved via genetic algorithms
    """
    
    def __init__(self, parameters=None):
        """
        Initialize a trading strategy with given parameters
        
        Args:
            parameters (dict): Dictionary of strategy parameters
        """
        self.parameters = parameters or {}
        self.fitness = 0.0
        
    def generate_random_parameters(self, param_ranges):
        """
        Generate random parameters within the specified ranges
        
        Args:
            param_ranges (dict): Dictionary of parameter ranges {param_name: (min, max)}
        """
        self.parameters = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                self.parameters[param] = random.randint(min_val, max_val)
            else:
                self.parameters[param] = random.uniform(min_val, max_val)
    
    def evaluate(self, data):
        """
        Evaluate the strategy on the provided data
        
        Args:
            data (pd.DataFrame): DataFrame with market data
            
        Returns:
            float: Strategy fitness/performance score
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def crossover(self, other):
        """
        Perform crossover with another strategy
        
        Args:
            other (TradingStrategy): Another strategy to crossover with
            
        Returns:
            TradingStrategy: A new strategy resulting from crossover
        """
        # Create a new strategy
        child = self.__class__()
        
        # Mix parameters from both parents
        child.parameters = {}
        for param in self.parameters:
            if random.random() < 0.5:
                child.parameters[param] = self.parameters[param]
            else:
                child.parameters[param] = other.parameters[param]
        
        return child
    
    def mutate(self, mutation_rate, param_ranges):
        """
        Mutate the strategy's parameters
        
        Args:
            mutation_rate (float): Probability of mutating each parameter
            param_ranges (dict): Dictionary of parameter ranges {param_name: (min, max)}
        """
        for param in self.parameters:
            if random.random() < mutation_rate:
                min_val, max_val = param_ranges[param]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    self.parameters[param] = random.randint(min_val, max_val)
                else:
                    self.parameters[param] = random.uniform(min_val, max_val)


class RLModelParameters(TradingStrategy):
    """
    Strategy that evolves reinforcement learning model hyperparameters
    """
    
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.model = None
        
    def create_model(self, env):
        """
        Create a RL model with the current parameters
        
        Args:
            env: Training environment
            
        Returns:
            model: The created model
        """
        # Create a PPO model with evolved parameters
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.parameters.get("learning_rate", 3e-4),
            n_steps=self.parameters.get("n_steps", 2048),
            batch_size=self.parameters.get("batch_size", 64),
            n_epochs=self.parameters.get("n_epochs", 10),
            gamma=self.parameters.get("gamma", 0.99),
            gae_lambda=self.parameters.get("gae_lambda", 0.95),
            clip_range=self.parameters.get("clip_range", 0.2),
            ent_coef=self.parameters.get("ent_coef", 0.0),
            verbose=0
        )
        
        return self.model
    
    def evaluate(self, env, timesteps=50000):
        """
        Evaluate by training a model with the current parameters
        
        Args:
            env: Training environment
            timesteps (int): Number of training timesteps
            
        Returns:
            float: Fitness score (average reward)
        """
        if self.model is None:
            self.create_model(env)
        
        # Train the model
        self.model.learn(total_timesteps=timesteps)
        
        # Evaluate the trained model
        rewards = []
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
        
        self.fitness = sum(rewards)
        return self.fitness


class MovingAverageCrossover(TradingStrategy):
    """
    Simple moving average crossover strategy with evolvable parameters
    """
    
    def evaluate(self, data):
        """
        Evaluate the MA crossover strategy on market data
        
        Args:
            data (pd.DataFrame): DataFrame with market data (must contain 'Close' prices)
            
        Returns:
            float: Strategy returns
        """
        # Extract parameters
        short_window = self.parameters.get("short_window", 10)
        long_window = self.parameters.get("long_window", 50)
        
        # Ensure long window is actually longer than short window
        if short_window >= long_window:
            short_window, long_window = sorted([short_window, long_window])
            self.parameters["short_window"] = short_window
            self.parameters["long_window"] = long_window
        
        # Compute moving averages
        data = data.copy()
        data["short_ma"] = data["Close"].rolling(window=short_window).mean()
        data["long_ma"] = data["Close"].rolling(window=long_window).mean()
        
        # Generate signals
        data["signal"] = 0
        data.loc[data["short_ma"] > data["long_ma"], "signal"] = 1
        data.loc[data["short_ma"] < data["long_ma"], "signal"] = -1
        
        # Calculate returns
        data["returns"] = data["Close"].pct_change()
        data["strategy_returns"] = data["signal"].shift(1) * data["returns"]
        
        # Calculate total return
        cumulative_return = (1 + data["strategy_returns"].fillna(0)).prod() - 1
        
        # Calculate Sharpe ratio (simple approximation)
        sharpe = cumulative_return / (data["strategy_returns"].std() + 1e-9)
        
        # Set fitness (combination of return and Sharpe)
        self.fitness = cumulative_return * (1 + sharpe)
        
        return self.fitness


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving trading strategies
    """
    
    def __init__(self, 
                 strategy_class,
                 population_size=50, 
                 generations=20, 
                 crossover_prob=0.7, 
                 mutation_rate=0.1,
                 param_ranges=None,
                 elite_size=2):
        """
        Initialize the genetic algorithm.
        
        Args:
            strategy_class: Class of trading strategy to evolve
            population_size (int): Size of the population
            generations (int): Number of generations to evolve
            crossover_prob (float): Probability of crossover
            mutation_rate (float): Probability of mutation per parameter
            param_ranges (dict): Dictionary of parameter ranges {param_name: (min, max)}
            elite_size (int): Number of top strategies to keep unchanged
        """
        self.strategy_class = strategy_class
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.param_ranges = param_ranges or {}
        self.elite_size = elite_size
        
        # Initialize population
        self.population = []
        
        # Track best strategies and fitnesses over generations
        self.best_strategies = []
        self.generation_best_fitness = []
        self.generation_avg_fitness = []
    
    def initialize_population(self):
        """
        Initialize the population with random strategies
        """
        self.population = []
        for _ in range(self.population_size):
            strategy = self.strategy_class()
            strategy.generate_random_parameters(self.param_ranges)
            self.population.append(strategy)
    
    def evaluate_population(self, evaluation_data):
        """
        Evaluate all strategies in the population
        
        Args:
            evaluation_data: Data to evaluate strategies on (DataFrame or environment)
        """
        for strategy in self.population:
            strategy.evaluate(evaluation_data)
    
    def select_parents(self):
        """
        Select parents for reproduction using tournament selection
        
        Returns:
            list: Selected parents
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(self.population, 
                                   key=lambda x: x.fitness, 
                                   reverse=True)
        
        # Keep the elite individuals
        elite = sorted_population[:self.elite_size]
        
        # Tournament selection for the rest
        parents = elite.copy()
        
        while len(parents) < self.population_size:
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def create_next_generation(self, parents):
        """
        Create the next generation through crossover and mutation
        
        Args:
            parents (list): List of parent strategies
            
        Returns:
            list: Next generation of strategies
        """
        # Keep elite strategies
        sorted_parents = sorted(parents, key=lambda x: x.fitness, reverse=True)
        next_generation = sorted_parents[:self.elite_size]
        
        # Create the rest through crossover and mutation
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            
            # Crossover
            if random.random() < self.crossover_prob:
                child = parent1.crossover(parent2)
            else:
                # If no crossover, clone the better parent
                if parent1.fitness > parent2.fitness:
                    child = copy.deepcopy(parent1)
                else:
                    child = copy.deepcopy(parent2)
            
            # Mutation
            child.mutate(self.mutation_rate, self.param_ranges)
            
            next_generation.append(child)
        
        return next_generation
    
    def evolve(self, evaluation_data, callback=None):
        """
        Run the genetic algorithm evolution process
        
        Args:
            evaluation_data: Data to evaluate strategies on
            callback: Optional callback function to call after each generation
            
        Returns:
            TradingStrategy: The best strategy found
        """
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate all strategies
            self.evaluate_population(evaluation_data)
            
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Record best strategy and fitness
            best_strategy = self.population[0]
            best_fitness = best_strategy.fitness
            avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
            
            self.best_strategies.append(copy.deepcopy(best_strategy))
            self.generation_best_fitness.append(best_fitness)
            self.generation_avg_fitness.append(avg_fitness)
            
            print(f"Generation {generation+1}/{self.generations}, Best Fitness: {best_fitness:.6f}, Avg Fitness: {avg_fitness:.6f}")
            print(f"Best Parameters: {best_strategy.parameters}")
            
            # Call the callback if provided
            if callback:
                callback(generation, self.population, best_strategy)
            
            # Stop if last generation
            if generation == self.generations - 1:
                break
            
            # Select parents for next generation
            parents = self.select_parents()
            
            # Create next generation
            self.population = self.create_next_generation(parents)
        
        # Return the best strategy found
        return self.best_strategies[-1]
    
    def plot_evolution(self, save_path=None):
        """
        Plot the evolution of fitness over generations
        
        Args:
            save_path (str): Optional path to save the plot
        """
        generations = range(1, len(self.generation_best_fitness) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.generation_best_fitness, 'b-', label='Best Fitness')
        plt.plot(generations, self.generation_avg_fitness, 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Evolution')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_best_strategy(self, path):
        """
        Save the best evolved strategy
        
        Args:
            path (str): Path to save the strategy
        """
        if not self.best_strategies:
            raise ValueError("No evolved strategies to save. Run evolve() first.")
        
        best_strategy = self.best_strategies[-1]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save strategy parameters
        strategy_data = {
            'strategy_class': self.strategy_class.__name__,
            'parameters': best_strategy.parameters,
            'fitness': best_strategy.fitness,
            'evolution_history': {
                'best_fitness': self.generation_best_fitness,
                'avg_fitness': self.generation_avg_fitness
            },
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(strategy_data, path)
        
        # If the strategy has a model, save it separately
        if hasattr(best_strategy, 'model') and best_strategy.model is not None:
            model_path = f"{path}_model"
            best_strategy.model.save(model_path)
        
        return path


def optimize_strategy(strategy_class, data, param_ranges, population_size=50, generations=20, 
                     crossover_prob=0.7, mutation_rate=0.1, elite_size=2):
    """
    Convenience function to evolve and optimize a trading strategy
    
    Args:
        strategy_class: Class of trading strategy to evolve
        data: Data to evaluate strategies on
        param_ranges (dict): Dictionary of parameter ranges {param_name: (min, max)}
        population_size (int): Size of the population
        generations (int): Number of generations to evolve
        crossover_prob (float): Probability of crossover
        mutation_rate (float): Probability of mutation per parameter
        elite_size (int): Number of top strategies to keep unchanged
        
    Returns:
        tuple: (best_strategy, evolution_history)
    """
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        strategy_class=strategy_class,
        population_size=population_size,
        generations=generations,
        crossover_prob=crossover_prob,
        mutation_rate=mutation_rate,
        param_ranges=param_ranges,
        elite_size=elite_size
    )
    
    best_strategy = ga.evolve(data)
    
    # Create results
    evolution_history = {
        'best_fitness': ga.generation_best_fitness,
        'avg_fitness': ga.generation_avg_fitness,
        'best_parameters': best_strategy.parameters
    }
    
    return best_strategy, evolution_history 