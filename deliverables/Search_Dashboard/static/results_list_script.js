document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const countryFilter = document.getElementById('country-filter').value;
            if (countryFilter) {
                formData.append('country', countryFilter);
            }
            const url = this.action;
            fetch(url, {
                method: 'POST',
                body: formData
            }).then(response => response.json())
            .then(data => {
                const resultsTable = document.getElementById('results-list');
                resultsTable.innerHTML = '';
                if (Array.isArray(data)) {
                    data.forEach((item, index) => {
                        setTimeout(() => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${item.score.toFixed(2)}</td>
                                <td>${item.symbol}</td>
                                <td>${item.company}</td>`;
                            resultsTable.appendChild(row);
                        }, 100 * index); // Delays the addition of each row
                    });
                } else if (data.error) {
                    resultsTable.innerHTML = `<tr><td colspan="3">Error: ${data.error}</td></tr>`;
                }
            }).catch(error => {
                document.getElementById('results-list').innerHTML = `<tr><td colspan="3">Error: ${error}</td></tr>`;
            });
        });
    });
});