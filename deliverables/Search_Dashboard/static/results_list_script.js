// API module to handle all API requests
const API = {
    // Function to handle POST requests
    // url: endpoint URL, body: payload for the POST request
    post: async (url, body) => {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(body)
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
};

// Function to fetch and display company descriptions
// symbol: company symbol, row: HTML row element for the company
async function fetchAndDisplayDescription(symbol, row) {
    // Check if the description row already exists and is next to this row
    let descRow = row.nextElementSibling;
    if (descRow && descRow.classList.contains("description-row")) {
        // Toggle visibility of an existing description row
        toggleDescriptionVisibility(descRow);
    } else {
        try {
            row.style.cursor = 'progress'; // Show loading indicator
            // Fetch description data from server
            const data = await API.post('/description', { symbol });
            if (data.description) {
                // Create and insert the description row
                descRow = createDescriptionRow(data.description);
                row.parentNode.insertBefore(descRow, row.nextSibling);
                // Add a slight delay for toggling visibility for better UX
                setTimeout(() => toggleDescriptionVisibility(descRow), 100);
            }
        } catch (error) {
            console.error('Error:', error); // Log errors to the console
        } finally {
            row.style.cursor = 'pointer'; // Revert cursor back to pointer
        }
    }
}

// Creates a new description row in the table
// description: text to be displayed in the row
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

// Toggles the visibility of the description row
// descRow: HTML row element containing the description
function toggleDescriptionVisibility(descRow) {
    if (descRow.classList.contains("visible")) {
        descRow.classList.remove("visible");
        setTimeout(() => descRow.style.display = 'none', 500);
    } else {
        descRow.style.display = 'table-row';
        setTimeout(() => descRow.classList.add("visible"), 10);
    }
}

// Event listener for DOM content loaded
document.addEventListener('DOMContentLoaded', () => {
    const resultsList = document.getElementById('results-list');
    // Event delegation for click events in the results list
    resultsList.addEventListener('click', (e) => {
        if (e.target && e.target.nodeName === "TD") {
            const row = e.target.parentNode;
            const symbol = row.dataset.symbol;
            if (symbol) {
                // Fetch and display description for the clicked row
                fetchAndDisplayDescription(symbol, row);
            }
        }
    });

    // Event listeners for all forms for handling submissions
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault(); // Prevent default form submission
            try {
                const formData = new FormData(form);
                const countryFilter = document.getElementById('country-filter').value;
                if (countryFilter) {
                    formData.append('country', countryFilter);
                }
                // Post form data and update results table
                const data = await API.post(form.action, formData);
                updateResultsTable(data);
            } catch (error) {
                // Display errors in the results table
                resultsList.innerHTML = `<tr><td colspan="3">Error: ${error.message}</td></tr>`;
            }
        });
    });
});

// Updates the results table with data
// data: array of items to be displayed in the table
function updateResultsTable(data) {
    const resultsTable = document.getElementById('results-list');
    resultsTable.innerHTML = ''; // Clear existing results
    if (Array.isArray(data)) {
        // Populate the table with new results
        data.forEach((item, index) => {
            setTimeout(() => {
                const row = document.createElement('tr');
                row.setAttribute('data-symbol', item.symbol);
                row.innerHTML = `
                    <td>${item.score.toFixed(2)}</td>
                    <td>${item.symbol}</td>
                    <td>${item.company}</td>`;
                resultsTable.appendChild(row);
            }, 100 * index); // Add a delay for each row for a staggered appearance
        });
    } else if (data.error) {
        // Display any errors received
        resultsTable.innerHTML = `<tr><td colspan="3">Error: ${data.error}</td></tr>`;
    }
}
