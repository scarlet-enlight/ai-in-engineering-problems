<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import '../styles/style.css';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<style>
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
			margin: 0;
			padding: 0;
			background: #f5f5f5;
		}
	</style>
</svelte:head>

<nav style="background: #2c3e50; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
	<div style="max-width: 1200px; margin: 0 auto; display: flex; gap: 1rem; align-items: center;">
		<h2 style="color: white; margin: 0; font-size: 1.2rem;">GTOA Tester</h2>
		<div style="display: flex; gap: 0.5rem; margin-left: auto;">
			<a href="/" style="background: #3498db; color: white; padding: 0.5rem 1rem; text-decoration: none; border-radius: 4px; transition: background 0.3s;">
				Strona główna
			</a>
			<a href="/raport" style="background: #27ae60; color: white; padding: 0.5rem 1rem; text-decoration: none; border-radius: 4px; transition: background 0.3s;">
				Wygeneruj Raport
			</a>
		</div>
	</div>
</nav>

<main style="max-width: 1200px; margin: 0 auto; padding: 20px;">
	{@render children()}
</main>
