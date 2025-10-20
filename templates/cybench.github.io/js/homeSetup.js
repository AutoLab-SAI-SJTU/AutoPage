const modelColors = {
  'Claude 3 Opus': '#E74C3C',
  'Gemini 1.5 Pro': '#9B59B6',
  'GPT-4o': '#2980B9',
  'Llama 3 70b Chat': '#F1C40F',
  'Llama 3.1 70b Instruct Turbo': '#E23233',
  'Mixtral 8x22b Instruct': '#1ABC9C',
};

function setupHome() {
  generateLeaderBoard();
}

function generateLeaderBoard() {
  Papa.parse('data/leaderboard.csv', {
    download: true,
    header: true,
    skipEmptyLines: true,
    complete: function(results) {
      const board = processLeaderBoardData(results.data);
      displayLeaderboard(board);
    }
  });
}

function getFSTMin(firstSolve) {
  if (firstSolve == '-') return firstSolve;
  const fbParts = firstSolve.split(':');
  const firstBloodMin = (parseInt(fbParts[0]) * 60) + parseInt(fbParts[1]);
  return firstBloodMin;
}

function processLeaderBoardData(data) {
  return data.map(row => ({
    model: row.Model,
    flagSuccess: row['End-to-End % Solved'],
    subtaskGuidedSuccess: row['Subtask-Guided % Solved'],
    subtaskSuccess: row['Subtasks % Solved'],
    fstStandard: getFSTMin(row['FST Standard']),
    fstSubtask: getFSTMin(row['FST Subtask'])
  }))
}


function displayLeaderboard(leaderboardData) {
  const table = document.getElementById('leaderboard');
  table.innerHTML = getLeaderboardHeaderHTML();

  const tbody = table.querySelector('tbody');
  leaderboardData.forEach(data => tbody.appendChild(createLeaderboardRow(data)));

  // Register custom sorting functions for First Solve Time columns
  $.fn.dataTable.ext.type.order['fst-min-pre'] = function(data) {
    const min = data.split(' ')[0];
    return min === '-' ? Infinity : parseInt(min, 10);
  };

  // Register custom sorting functions for percentage columns
  $.fn.dataTable.ext.type.order['percent-pre'] = function(data) {
    return parseFloat(data.replace('%', ''));
  };

  const dataTable = $('#leaderboard').DataTable({
    destroy: true,
    paging: false,
    searching: false,
    //FOR LEADERBOARD SCREENSHOT PUT ordering TO false
    // ordering: false,
    ordering: true, 
    order: [
      [1, 'desc'], // Initially sort by the "Unguided % Solved" column in descending order    
      [2, 'desc']  // Then by the second column in descending order as a tie breaker
    ], 
    columnDefs: [
      { type: 'percent', targets: [1, 2, 3], orderSequence: ['desc', 'asc'] }, // Apply custom sorting to percentage columns
      { type: 'fst-min', targets: [4, 5] , orderSequence: ['desc', 'asc']} // Apply custom sorting to columns 4 and 5
    ],
    info: false
  });
  
      // Apply initial highlighting to the sorted column
      applyColumnHighlight(dataTable, 1);

      // Event listener for column sorting
      dataTable.on('order.dt', function() {
          // Get the sorted column index
          const order = dataTable.order();
          const columnIndex = order[0][0];
  
          // Apply the highlight to the sorted column and header
          applyColumnHighlight(dataTable, columnIndex);
      });
}

function applyColumnHighlight(dataTable, columnIndex) {
  // Remove existing highlights
  $('th').removeClass('sorted-column-header');
  $('td').removeClass('sorted-column');

  // Determine the correct header to highlight
  if (columnIndex === 4 || columnIndex === 5) {
      // If sorting by columns under "Most Difficult Task Solved"
      const subheaderIndex = columnIndex - 4; // Map 4 -> 0, 5 -> 1
      $(`#leaderboard thead tr:nth-child(2) th`).eq(subheaderIndex).addClass('sorted-column-header');
  } else {
      // For other columns
      $(`#leaderboard thead tr:nth-child(1) th`).eq(columnIndex).addClass('sorted-column-header');
  }

  // Highlight the correct column cells
  $(`#leaderboard tbody td:nth-child(${columnIndex + 1})`).addClass('sorted-column');
}

function getLeaderboardHeaderHTML() {
  return `
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2" style="max-width: 6rem;">Unguided % Solved</th>
        <th rowspan="2" class="wrap">Subtask-Guided % Solved</th>
        <th rowspan="2" class="wrap">Subtasks % Solved</th>
        <th colspan="2">Most Difficult Task Solved (First Solve Time by Humans)</th>
      </tr>
      <tr>
        <th class="subheader">Unguided</th>
        <th class="subheader">Subtask-Guided</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
}

function createLeaderboardRow(data) {
  const row = document.createElement('tr');
  ['model', 'flagSuccess', 'subtaskGuidedSuccess', 'subtaskSuccess', 'fstStandard', 'fstSubtask'].forEach((key, index) => {
    const cell = document.createElement('td');
    cell.classList = 'spaced-cell';
    let cellContent = data[key];

    // Check if the value is a percentage and replace '%' or 'N/A' with '--'
    if (['flagSuccess', 'subtaskGuidedSuccess', 'subtaskSuccess'].includes(key)) {
      if ( !cellContent || cellContent === 'N/A' || cellContent === '%') {
        cellContent = '--';
      } else {
        cellContent = cellContent + '%';
      }
    }
    // Check if the value is a time and replace 'NaN min' or '-' with '--'
    else if (['fstStandard', 'fstSubtask'].includes(key)) {
      if (cellContent === '-' || isNaN(cellContent)) {
        cellContent = '--';
      } else {
        cellContent = cellContent + ' min';
      }
    }

    cell.textContent = cellContent;

    if (index <= 3) {
      cell.style.borderRight = '1px solid #0e0e0e';
    }
    row.appendChild(cell);
  });
  return row;
}