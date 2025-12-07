<script>
	let title = "Generator Raportów";
</script>

<div class="container">
	<header>
		<h1>{title}</h1>
		<p class="subtitle">System do analizy i porównywania różnych implementacji algorytmów</p>
	</header>

	<section class="intro">
		<h2>Wprowadzenie</h2>
		<p>
			Ta aplikacja umożliwia generowanie szczegółowych raportów porównawczych
			dla różnych wariantów i implementacji algorytmu.
		</p>
		<p>
			System pozwala na analizę wydajności, złożoności oraz poprawności
			poszczególnych implementacji w kontrolowanych warunkach testowych.
		</p>
	</section>

	<section class="instructions">
		<h2>Instrukcja obsługi</h2>
		<ol>
			<li>
				<strong>Wybór algorytmu</strong> - Wybierz implementację algorytmu,
				którą chcesz przetestować
			</li>
			<li>
				<strong>Konfiguracja parametrów</strong> - Ustaw parametry testowe
				(liczbę populacji, liczbę iteracji, itp.)
			</li>
			<li>
				<strong>Generowanie raportu</strong> - Uruchom proces generowania raportu
				klikając odpowiedni przycisk
			</li>
			<li>
				<strong>Analiza wyników</strong> - Przeglądaj wygenerowane raporty
				zawierające wyniki testowania algorytmów.
			</li>
		</ol>
	</section>

	<section class="get-started">
		<h2>Rozpocznij pracę</h2>
		<p>Aby rozpocząć, przejdź do odpowiedniej sekcji używając menu nawigacji.</p>
	</section>
</div>

<style>
	.container {
		max-width: 900px;
		margin: 0 auto;
		padding: 2rem;
		font-family: system-ui, -apple-system, sans-serif;
	}

	header {
		text-align: center;
		margin-bottom: 3rem;
		padding-bottom: 2rem;
		border-bottom: 2px solid #e5e7eb;
	}

	h1 {
		color: #1f2937;
		font-size: 2.5rem;
		margin-bottom: 0.5rem;
	}

	.subtitle {
		color: #6b7280;
		font-size: 1.1rem;
	}

	section {
		margin-bottom: 2.5rem;
	}

	h2 {
		color: #374151;
		font-size: 1.8rem;
		margin-bottom: 1rem;
		border-left: 4px solid #3b82f6;
		padding-left: 1rem;
	}

	p {
		color: #4b5563;
		line-height: 1.7;
		margin-bottom: 1rem;
	}

	ol {
		color: #4b5563;
		line-height: 1.8;
		padding-left: 1.5rem;
	}

	li {
		margin-bottom: 0.8rem;
	}

	strong {
		color: #1f2937;
		font-weight: 600;
	}

	.get-started {
		background-color: #f3f4f6;
		padding: 1.5rem;
		border-radius: 8px;
		text-align: center;
	}

	.get-started p {
		margin: 0;
		font-size: 1.1rem;
	}
</style>
