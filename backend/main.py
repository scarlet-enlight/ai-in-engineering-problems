from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import traceback

from test_functions import TEST_FUNCTIONS
from gtoa import GTOA
from algorithms import ALGORITHMS, ALGORITHM_NAMES

# Add GTOA to algorithms dictionary
ALGORITHMS['GTOA'] = GTOA

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI", "status": "running"}

@app.get("/api/algorithms")
def get_algorithms():
    """Return available optimization algorithms with full names"""
    return {
        "algorithms": [
            {"code": code, "name": ALGORITHM_NAMES[code]} 
            for code in ALGORITHMS.keys()
        ]
    }

class OptimizationRequest(BaseModel):
    algorithm: str = "GTOA"
    function_name: str
    dimension: int
    population_size: int = 50
    max_iterations: Optional[int] = None
    seed: Optional[int] = None

class BatchOptimizationRequest(BaseModel):
    algorithm: str = "GTOA"
    function_names: list[str]
    dimension: int
    population_size: int = 50
    max_iterations: Optional[int] = None
    seed: Optional[int] = None
    runs_per_function: int = 1

@app.get("/api/functions")
def get_functions():
    """Return available test functions with their configurations"""
    try:
        functions = {}
        for name, config in TEST_FUNCTIONS.items():
            bounds = config["bounds"]
            if isinstance(bounds, tuple) and np.isscalar(bounds[0]):
                bounds_info = {"type": "uniform", "min": float(bounds[0]), "max": float(bounds[1])}
            else:
                bounds_info = {
                    "type": "per_dimension",
                    "min": bounds[0].tolist() if hasattr(bounds[0], 'tolist') else list(bounds[0]),
                    "max": bounds[1].tolist() if hasattr(bounds[1], 'tolist') else list(bounds[1])
                }
            
            functions[name] = {
                "dims": config["dims"],
                "bounds": bounds_info
            }
        return functions
    except Exception as e:
        print(f"Error in get_functions: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def run_single_algorithm(algorithm_name: str, func_config: dict, dimension: int, 
                        population_size: int, max_iterations: Optional[int], seed: Optional[int]):
    """Helper function to run a single algorithm"""
    if algorithm_name not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm_name}")
    
    AlgorithmClass = ALGORITHMS[algorithm_name]
    
    # For GTOA, use Tmax parameter instead of max_iterations
    if algorithm_name == 'GTOA':
        algo = AlgorithmClass(
            func=func_config["fn"],
            dim=dimension,
            bounds=func_config["bounds"],
            population_size=population_size,
            Tmax=max_iterations,
            seed=seed
        )
    else:
        algo = AlgorithmClass(
            func=func_config["fn"],
            dim=dimension,
            bounds=func_config["bounds"],
            population_size=population_size,
            max_iterations=max_iterations,
            seed=seed
        )
    
    return algo.optimize(verbose=False)

@app.post("/api/optimize")
def run_optimization(request: OptimizationRequest):
    """Run optimization with specified algorithm and parameters"""
    try:
        if request.function_name not in TEST_FUNCTIONS:
            raise HTTPException(status_code=400, detail=f"Unknown function: {request.function_name}")
        
        func_config = TEST_FUNCTIONS[request.function_name]
        
        if request.dimension not in func_config["dims"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension {request.dimension} not supported for {request.function_name}"
            )
        
        # Run algorithm
        best_solution, best_fitness, total_evals, history_iter, history_T, history_best = run_single_algorithm(
            request.algorithm,
            func_config,
            request.dimension,
            request.population_size,
            request.max_iterations,
            request.seed
        )
        
        return {
            "success": True,
            "algorithm": request.algorithm,
            "best_solution": best_solution.tolist(),
            "best_fitness": float(best_fitness),
            "total_evaluations": int(total_evals),
            "history": {
                "iterations": history_iter,
                "evaluations": history_T,
                "best_fitness": history_best
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in run_optimization: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize/batch")
def run_batch_optimization(request: BatchOptimizationRequest):
    """Run optimization on multiple test functions"""
    try:
        results = {}
        
        for func_name in request.function_names:
            if func_name not in TEST_FUNCTIONS:
                results[func_name] = {"error": f"Unknown function: {func_name}"}
                continue
            
            func_config = TEST_FUNCTIONS[func_name]
            
            if request.dimension not in func_config["dims"]:
                results[func_name] = {
                    "error": f"Dimension {request.dimension} not supported for {func_name}"
                }
                continue
            
            # Run multiple times if requested
            run_results = []
            for run_idx in range(request.runs_per_function):
                run_seed = request.seed + run_idx if request.seed is not None else None
                
                best_solution, best_fitness, total_evals, history_iter, history_T, history_best = run_single_algorithm(
                    request.algorithm,
                    func_config,
                    request.dimension,
                    request.population_size,
                    request.max_iterations,
                    run_seed
                )
                
                run_results.append({
                    "run": run_idx + 1,
                    "best_solution": best_solution.tolist(),
                    "best_fitness": float(best_fitness),
                    "total_evaluations": int(total_evals),
                    "final_iteration": len(history_iter),
                    "history": {
                        "iterations": history_iter,
                        "evaluations": history_T,
                        "best_fitness": history_best
                    }
                })
            
            # Calculate statistics if multiple runs
            if request.runs_per_function > 1:
                fitness_values = [r["best_fitness"] for r in run_results]
                best_run_idx = int(np.argmin(fitness_values))
                results[func_name] = {
                    "success": True,
                    "runs": run_results,
                    "statistics": {
                        "mean_fitness": float(np.mean(fitness_values)),
                        "std_fitness": float(np.std(fitness_values)),
                        "min_fitness": float(np.min(fitness_values)),
                        "max_fitness": float(np.max(fitness_values)),
                        "median_fitness": float(np.median(fitness_values))
                    },
                    "best_run_history": run_results[best_run_idx]["history"]
                }
            else:
                results[func_name] = {
                    "success": True,
                    "runs": run_results
                }
        
        return {
            "success": True,
            "algorithm": request.algorithm,
            "results": results,
            "total_functions": len(request.function_names)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in run_batch_optimization: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))