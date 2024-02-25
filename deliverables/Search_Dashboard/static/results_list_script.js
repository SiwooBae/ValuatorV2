function fetchAndDisplayDescription(symbol, row) {
    // Check if the description row already exists and is next to this row
    let descRow = row.nextElementSibling;
    if (descRow && descRow.classList.contains("description-row")) {
        toggleDescriptionVisibility(descRow);
    } else {
        // If no description row exists, fetch and display it
        fetch('/description', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'symbol=' + encodeURIComponent(symbol)
        }).then(response => response.json())
            .then(data => {
                if (data.description) {
                    // Create and insert the description row
                    descRow = createDescriptionRow(data.description);
                    row.parentNode.insertBefore(descRow, row.nextSibling);

                    // Start the slide-down animation
                    setTimeout(() => toggleDescriptionVisibility(descRow), 10);
                }
            }).catch(error => {
            console.error('Error:', error);
        });
    }
}

function createDescriptionRow(description) {
    const descCell = document.createElement('td');
    descCell.colSpan = 3;
    descCell.innerHTML = description;
    descCell.classList.add("description-cell");

    const descRow = document.createElement('tr');
    descRow.appendChild(descCell);
    descRow.classList.add("description-row");

    return descRow;
}

function toggleDescriptionVisibility(descRow) {
    if (descRow.classList.contains("visible")) {
        // Hide the description row
        descRow.classList.remove("visible");
        setTimeout(() => descRow.style.display = 'none', 500); // Wait for animation
    } else {
        // Show the description row
        descRow.style.display = 'table-row';
        setTimeout(() => descRow.classList.add("visible"), 10); // Start animation
    }
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('results-list').addEventListener('click', function (e) {
        if (e.target && e.target.nodeName === "TD") {
            const row = e.target.parentNode;
            const symbol = row.dataset.symbol;
            if (symbol) {
                fetchAndDisplayDescription(symbol, row);
            }
        }
    });
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function (e) {
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
                                row.setAttribute('data-symbol', item.symbol);
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