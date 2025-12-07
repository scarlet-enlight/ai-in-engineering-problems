<script lang="ts">
	import { onMount } from 'svelte';

	interface AlgorithmInfo {
		code: string;
		name: string;
	}

	interface FunctionConfig {
		dims: number[];
		bounds: {
			type: string;
			min: number | number[];
			max: number | number[];
		};
	}

	interface OptimizationResult {
		success?: boolean;
		algorithm?: string;
		best_solution?: number[];
		best_fitness?: number;
		total_evaluations?: number;
		history?: {
			iterations: number[];
			evaluations: number[];
			best_fitness: number[];
		};
		error?: string;
	}

	interface BatchResult {
		success?: boolean;
		algorithm?: string;
		results?: Record<string, any>;
		total_functions?: number;
		error?: string;
	}

	let algorithms: AlgorithmInfo[] = $state([]);
	let selectedAlgorithm = $state('GTOA');
	let functions: Record<string, FunctionConfig> = $state({});
	let selectedFunction = $state('Sphere');
	let selectedFunctions = $state<Set<string>>(new Set(['Sphere']));
	let selectedDimension = $state(2);
	let populationSize = $state(50);
	let maxIterations = $state<number | null>(null);
	let runsPerFunction = $state(1);
	let testMode = $state<'single' | 'multiple'>('single');
	let loading = $state(false);
	let loadingFunctions = $state(true);
	let result: OptimizationResult | null = $state(null);
	let batchResult: BatchResult | null = $state(null);

	const API_BASE = 'http://localhost:8000';

	async function fetchAPI(endpoint: string, options?: RequestInit) {
		try {
			const response = await fetch(`${API_BASE}${endpoint}`, options);
			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}
			return await response.json();
		} catch (error) {
			console.error(`API error (${endpoint}):`, error);
			throw new Error(`Failed to connect to backend at ${API_BASE}. Make sure FastAPI server is running.`);
		}
	}

	onMount(async () => {
		try {
			const [funcsData, algoData] = await Promise.all([
				fetchAPI('/api/functions'),
				fetchAPI('/api/algorithms')
			]);
			functions = funcsData;
			algorithms = algoData.algorithms;
		} catch (error) {
			console.error('Error loading data:', error);
		} finally {
			loadingFunctions = false;
		}
	});

	function toggleFunction(funcName: string) {
		const newSet = new Set(selectedFunctions);
		if (newSet.has(funcName)) {
			newSet.delete(funcName);
		} else {
			newSet.add(funcName);
		}
		selectedFunctions = newSet;
	}

	async function runBatchOptimization() {
		loading = true;
		batchResult = null;
		result = null;

		try {
			batchResult = await fetchAPI('/api/optimize/batch', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					algorithm: selectedAlgorithm,
					function_names: Array.from(selectedFunctions),
					dimension: selectedDimension,
					population_size: populationSize,
					max_iterations: maxIterations,
					runs_per_function: runsPerFunction
				})
			});
		} catch (error) {
			batchResult = { error: error instanceof Error ? error.message : 'Unknown error' };
		} finally {
			loading = false;
		}
	}

	async function runOptimization() {
		if (testMode === 'multiple') {
			await runBatchOptimization();
		} else {
			loading = true;
			result = null;
			batchResult = null;

			try {
				result = await fetchAPI('/api/optimize', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						algorithm: selectedAlgorithm,
						function_name: selectedFunction,
						dimension: selectedDimension,
						population_size: populationSize,
						max_iterations: maxIterations
					})
				});
			} catch (error) {
				result = { error: error instanceof Error ? error.message : 'Unknown error' };
			} finally {
				loading = false;
			}
		}
	}
</script>

<div class="container">
	<h1>Optimization Algorithm Tester</h1>

	<div class="form-card">
		<h2>Configuration</h2>

		{#if loadingFunctions}
			<div class="loading-message">Loading...</div>
		{:else if Object.keys(functions).length === 0}
			<div class="error">No test functions available. Check backend connection.</div>
		{:else}
			<div class="form-group">
				<label for="algorithm">Algorithm:</label>
				<select id="algorithm" bind:value={selectedAlgorithm}>
					{#each algorithms as algo}
						<option value={algo.code}>{algo.name}</option>
					{/each}
				</select>
				<small>Select optimization algorithm to test</small>
			</div>

			<div class="form-group">
				<label>Test Mode:</label>
				<div class="radio-group">
					<label class="radio-label">
						<input type="radio" bind:group={testMode} value="single" />
						Single Function
					</label>
					<label class="radio-label">
						<input type="radio" bind:group={testMode} value="multiple" />
						Multiple Functions
					</label>
				</div>
			</div>

			{#if testMode === 'single'}
				<div class="form-group">
					<label for="function">Test Function:</label>
					<select id="function" bind:value={selectedFunction}>
						{#each Object.keys(functions) as funcName}
							<option value={funcName}>{funcName}</option>
						{/each}
					</select>
				</div>

				{#if functions[selectedFunction]}
					<div class="form-group">
						<label for="dimension">Dimension:</label>
						<select id="dimension" bind:value={selectedDimension}>
							{#each functions[selectedFunction].dims as dim}
								<option value={dim}>{dim}D</option>
							{/each}
						</select>
					</div>

					<div class="bounds-info">
						<strong>Bounds:</strong>
						{#if functions[selectedFunction].bounds.type === 'uniform'}
							[{functions[selectedFunction].bounds.min}, {functions[selectedFunction].bounds.max}]
						{:else}
							Per dimension
						{/if}
					</div>
				{/if}
			{:else}
				<div class="form-group">
					<label>Select Test Functions:</label>
					<div class="checkbox-group">
						{#each Object.keys(functions) as funcName}
							<label class="checkbox-label">
								<input
									type="checkbox"
									checked={selectedFunctions.has(funcName)}
									onchange={() => toggleFunction(funcName)}
								/>
								{funcName}
								<span class="dims-info">({functions[funcName].dims.join(', ')}D)</span>
							</label>
						{/each}
					</div>
					{#if selectedFunctions.size === 0}
						<small class="warning">Please select at least one function</small>
					{/if}
				</div>

				<div class="form-group">
					<label for="dimension-batch">Dimension (for all selected):</label>
					<input id="dimension-batch" type="number" bind:value={selectedDimension} min="2" />
					<small>Make sure selected functions support this dimension</small>
				</div>

				<div class="form-group">
					<label for="runs">Runs per Function:</label>
					<input id="runs" type="number" bind:value={runsPerFunction} min="1" max="100" />
					<small>Number of independent runs for each function</small>
				</div>
			{/if}

			<div class="form-group">
				<label for="population">Population Size:</label>
				<input id="population" type="number" bind:value={populationSize} min="2" step="2" />
				<small>Must be even number</small>
			</div>

			<div class="form-group">
				<label for="maxIter">Max Iterations (optional):</label>
				<input id="maxIter" type="number" bind:value={maxIterations} placeholder="Auto (100*D)" />
			</div>

			<button
				onclick={runOptimization}
				disabled={loading || loadingFunctions || (testMode === 'multiple' && selectedFunctions.size === 0)}
				class="run-button"
			>
				{loading ? 'Running...' : testMode === 'single' ? 'Run Optimization' : `Run Batch (${selectedFunctions.size} functions)`}
			</button>
		{/if}
	</div>

	{#if result && testMode === 'single'}
		<div class="result-card">
			<h2>Results - {result.algorithm || 'Unknown'}</h2>

			{#if result.error}
				<div class="error">{result.error}</div>
			{:else if result.success}
				<div class="result-grid">
					<div class="result-item">
						<strong>Best Fitness:</strong>
						<span>{result.best_fitness?.toExponential(6)}</span>
					</div>
					<div class="result-item">
						<strong>Total Evaluations:</strong>
						<span>{result.total_evaluations}</span>
					</div>
					<div class="result-item">
						<strong>Iterations:</strong>
						<span>{result.history?.iterations.length}</span>
					</div>
				</div>

				<div class="solution">
					<strong>Best Solution:</strong>
					<code>{result.best_solution?.map((x) => x.toFixed(6)).join(', ')}</code>
				</div>

				{#if result.history}
					<div class="convergence">
						<h3>Convergence History</h3>
						<div class="history-table">
							<table>
								<thead>
									<tr>
										<th>Iteration</th>
										<th>Evaluations</th>
										<th>Best Fitness</th>
									</tr>
								</thead>
								<tbody>
									{#each result.history.iterations.slice(-10) as iter, i}
										<tr>
											<td>{iter}</td>
											<td>{result.history?.evaluations[result.history.iterations.length - 10 + i]}</td>
											<td>{result.history?.best_fitness[result.history.iterations.length - 10 + i]?.toExponential(6)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
							<small>Showing last 10 iterations</small>
						</div>
					</div>
				{/if}
			{/if}
		</div>
	{/if}

	{#if batchResult && testMode === 'multiple'}
		<div class="result-card">
			<h2>Batch Results - {batchResult.algorithm || 'Unknown'}</h2>

			{#if batchResult.error}
				<div class="error">{batchResult.error}</div>
			{:else if batchResult.success && batchResult.results}
				<div class="batch-summary">
					<strong>Tested {batchResult.total_functions} function(s)</strong>
					{#if runsPerFunction > 1}
						<span>with {runsPerFunction} runs each</span>
					{/if}
				</div>

				{#each Object.entries(batchResult.results) as [funcName, funcResult]}
					<div class="function-result">
						<h3>{funcName}</h3>

						{#if funcResult.error}
							<div class="error">{funcResult.error}</div>
						{:else if funcResult.success}
							{#if funcResult.statistics}
								<div class="stats-grid">
									<div class="stat-item">
										<strong>Best:</strong>
										<span>{funcResult.statistics.min_fitness.toExponential(6)}</span>
									</div>
									<div class="stat-item">
										<strong>Worst:</strong>
										<span>{funcResult.statistics.max_fitness.toExponential(6)}</span>
									</div>
									<div class="stat-item">
										<strong>Mean:</strong>
										<span>{funcResult.statistics.mean_fitness.toExponential(6)}</span>
									</div>
									<div class="stat-item">
										<strong>Median:</strong>
										<span>{funcResult.statistics.median_fitness.toExponential(6)}</span>
									</div>
									<div class="stat-item">
										<strong>Std Dev:</strong>
										<span>{funcResult.statistics.std_fitness.toExponential(6)}</span>
									</div>
								</div>

								<details class="run-details">
									<summary>Show all {funcResult.runs.length} runs</summary>
									<table>
										<thead>
											<tr>
												<th>Run</th>
												<th>Best Fitness</th>
												<th>Evaluations</th>
												<th>Iterations</th>
											</tr>
										</thead>
										<tbody>
											{#each funcResult.runs as run}
												<tr>
													<td>#{run.run}</td>
													<td>{run.best_fitness.toExponential(6)}</td>
													<td>{run.total_evaluations}</td>
													<td>{run.final_iteration}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</details>

								{#if funcResult.best_run_history}
									<div class="convergence">
										<h4>Convergence History (Best Run)</h4>
										<div class="history-table">
											<table>
												<thead>
													<tr>
														<th>Iteration</th>
														<th>Evaluations</th>
														<th>Best Fitness</th>
													</tr>
												</thead>
												<tbody>
													{#each funcResult.best_run_history.iterations.slice(-10) as iter, i}
														<tr>
															<td>{iter}</td>
															<td>{funcResult.best_run_history.evaluations[funcResult.best_run_history.iterations.length - 10 + i]}</td>
															<td>{funcResult.best_run_history.best_fitness[funcResult.best_run_history.iterations.length - 10 + i]?.toExponential(6)}</td>
														</tr>
													{/each}
												</tbody>
											</table>
											<small>Showing last 10 iterations from the best run</small>
										</div>
									</div>
								{/if}
							{:else if funcResult.runs.length === 1}
								<div class="result-grid">
									<div class="result-item">
										<strong>Best Fitness:</strong>
										<span>{funcResult.runs[0].best_fitness.toExponential(6)}</span>
									</div>
									<div class="result-item">
										<strong>Evaluations:</strong>
										<span>{funcResult.runs[0].total_evaluations}</span>
									</div>
									<div class="result-item">
										<strong>Iterations:</strong>
										<span>{funcResult.runs[0].final_iteration}</span>
									</div>
								</div>

								{#if funcResult.runs[0].history}
									<div class="convergence">
										<h4>Convergence History</h4>
										<div class="history-table">
											<table>
												<thead>
													<tr>
														<th>Iteration</th>
														<th>Evaluations</th>
														<th>Best Fitness</th>
													</tr>
												</thead>
												<tbody>
													{#each funcResult.runs[0].history.iterations.slice(-10) as iter, i}
														<tr>
															<td>{iter}</td>
															<td>{funcResult.runs[0].history.evaluations[funcResult.runs[0].history.iterations.length - 10 + i]}</td>
															<td>{funcResult.runs[0].history.best_fitness[funcResult.runs[0].history.iterations.length - 10 + i]?.toExponential(6)}</td>
														</tr>
													{/each}
												</tbody>
											</table>
											<small>Showing last 10 iterations</small>
										</div>
									</div>
								{/if}
							{/if}
						{/if}
					</div>
				{/each}
			{/if}
		</div>
	{/if}
</div>