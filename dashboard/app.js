// LCS Fantasy Interactive Weekly Dashboard Logic

let rawData = null;
let filteredPlayers = [];
let currentPositionFilter = 'ALL';
let currentSortCol = 'total_pts';
let currentSortDir = 'desc';
let pointsMode = 'raw'; // 'raw' or 'adjusted'
let trendChart = null;

document.addEventListener('DOMContentLoaded', async () => {
  await loadDashboardData();
  setupEventListeners();
});

async function loadDashboardData() {
  try {
    const resp = await fetch('./dashboard_data.json');
    if (!resp.ok) throw new Error('Could not load dashboard_data.json');
    rawData = await resp.json();

    populateFilterDropdowns();
    applyFilters();
  } catch (err) {
    console.error('Error loading dashboard data:', err);
    document.getElementById('tableContainer').innerHTML = `
      <div style="padding: 40px; text-align: center; color: var(--badge-top);">
        ⚠️ Could not load data. Run <code>python data_pipeline/export_dashboard_data.py</code> first!
      </div>
    `;
  }
}

function populateFilterDropdowns() {
  const leagueSelect = document.getElementById('leagueSelect');
  const yearSelect = document.getElementById('yearSelect');
  const splitSelect = document.getElementById('splitSelect');

  // Populate Leagues
  const leagues = ['ALL', ...rawData.leagues];
  leagueSelect.innerHTML = leagues.map(l => `<option value="${l}">${l === 'ALL' ? 'All Leagues' : l}</option>`).join('');
  
  // Default to LCS if present
  if (rawData.leagues.includes('LCS')) {
    leagueSelect.value = 'LCS';
  }

  // Populate Years
  const years = ['ALL', ...rawData.years.sort().reverse()];
  yearSelect.innerHTML = years.map(y => `<option value="${y}">${y === 'ALL' ? 'All Years' : y}</option>`).join('');
}

function setupEventListeners() {
  document.getElementById('searchInput').addEventListener('input', applyFilters);
  document.getElementById('leagueSelect').addEventListener('change', applyFilters);
  document.getElementById('yearSelect').addEventListener('change', applyFilters);
  document.getElementById('splitSelect').addEventListener('change', applyFilters);
  document.getElementById('pointsModeSelect').addEventListener('change', (e) => {
    pointsMode = e.target.value;
    applyFilters();
  });

  // Position Buttons
  document.querySelectorAll('.pos-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.pos-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentPositionFilter = e.target.dataset.pos;
      applyFilters();
    });
  });

  // Export CSV
  document.getElementById('exportCsvBtn').addEventListener('click', exportToCSV);

  // Modal close
  document.getElementById('modalCloseBtn').addEventListener('click', closeModal);
  document.getElementById('modalOverlay').addEventListener('click', (e) => {
    if (e.target.id === 'modalOverlay') closeModal();
  });
}

function applyFilters() {
  if (!rawData || !rawData.players) return;

  const search = document.getElementById('searchInput').value.toLowerCase().trim();
  const league = document.getElementById('leagueSelect').value;
  const year = document.getElementById('yearSelect').value;
  const split = document.getElementById('splitSelect').value;

  filteredPlayers = rawData.players.filter(p => {
    if (search && !p.playername.toLowerCase().includes(search) && !p.teamname.toLowerCase().includes(search)) return false;
    if (league !== 'ALL' && p.league !== league) return false;
    if (year !== 'ALL' && p.year !== year) return false;
    if (split !== 'ALL' && p.split !== split) return false;
    if (currentPositionFilter !== 'ALL' && p.position !== currentPositionFilter) return false;
    return true;
  });

  // Calculate dynamic sort values
  filteredPlayers.forEach(p => {
    p._active_total = pointsMode === 'adjusted' ? p.total_adjusted_pts : p.total_fantasy_pts;
    p._active_avg = p.total_games > 0 ? (p._active_total / p.total_games) : 0;
  });

  // Sort
  filteredPlayers.sort((a, b) => {
    let valA = a._active_total;
    let valB = b._active_total;

    if (currentSortCol === 'playername') {
      valA = a.playername;
      valB = b.playername;
      return currentSortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
    } else if (currentSortCol === 'teamname') {
      valA = a.teamname;
      valB = b.teamname;
      return currentSortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
    } else if (currentSortCol === 'avg_pts') {
      valA = a._active_avg;
      valB = b._active_avg;
    } else if (currentSortCol.startsWith('W')) {
      const weekKey = currentSortCol;
      valA = a.weekly_stats[weekKey] ? (pointsMode === 'adjusted' ? a.weekly_stats[weekKey].adjusted_pts : a.weekly_stats[weekKey].fantasy_pts) : 0;
      valB = b.weekly_stats[weekKey] ? (pointsMode === 'adjusted' ? b.weekly_stats[weekKey].adjusted_pts : b.weekly_stats[weekKey].fantasy_pts) : 0;
    }

    return currentSortDir === 'asc' ? valA - valB : valB - valA;
  });

  updateKPICards();
  renderTable();
  renderTrendChart();
}

function updateKPICards() {
  document.getElementById('totalPlayersKpi').innerText = filteredPlayers.length;
  
  if (filteredPlayers.length > 0) {
    const topPlayer = filteredPlayers[0];
    document.getElementById('topPlayerKpi').innerText = topPlayer.playername;
    document.getElementById('topPlayerSub').innerText = `${topPlayer.teamname} • ${topPlayer._active_total.toFixed(1)} Pts`;

    const totalGames = filteredPlayers.reduce((acc, p) => acc + p.total_games, 0);
    const totalPts = filteredPlayers.reduce((acc, p) => acc + p._active_total, 0);
    const avgPts = totalGames > 0 ? (totalPts / totalGames).toFixed(2) : '0.00';
    document.getElementById('avgPtsKpi').innerText = avgPts;

    // Highest single week score
    let maxWeekScore = 0;
    let maxWeekPlayer = '-';
    filteredPlayers.forEach(p => {
      Object.values(p.weekly_stats).forEach(w => {
        const pts = pointsMode === 'adjusted' ? w.adjusted_pts : w.fantasy_pts;
        if (pts > maxWeekScore) {
          maxWeekScore = pts;
          maxWeekPlayer = `${p.playername} (W${w.week_num})`;
        }
      });
    });

    document.getElementById('maxWeekKpi').innerText = maxWeekScore.toFixed(1);
    document.getElementById('maxWeekSub').innerText = maxWeekPlayer;

  } else {
    document.getElementById('topPlayerKpi').innerText = 'N/A';
    document.getElementById('topPlayerSub').innerText = '-';
    document.getElementById('avgPtsKpi').innerText = '0.00';
    document.getElementById('maxWeekKpi').innerText = '0.0';
    document.getElementById('maxWeekSub').innerText = '-';
  }
}

function renderTable() {
  const container = document.getElementById('tableContainer');

  // Discover all distinct weeks in filtered dataset
  const weekSet = new Set();
  filteredPlayers.forEach(p => {
    Object.keys(p.weekly_stats).forEach(w => weekSet.add(w));
  });

  const sortedWeeks = Array.from(weekSet).sort((a, b) => {
    const numA = parseInt(a.replace('W', '')) || 0;
    const numB = parseInt(b.replace('W', '')) || 0;
    return numA - numB;
  });

  if (filteredPlayers.length === 0) {
    container.innerHTML = `<div style="padding: 40px; text-align: center; color: var(--text-muted);">No players found matching your criteria.</div>`;
    return;
  }

  let html = `
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th class="rank-cell">#</th>
            <th onclick="sortTable('playername')">Player</th>
            <th onclick="sortTable('teamname')">Team</th>
            <th>Pos</th>
            <th>Games</th>
            <th onclick="sortTable('total_pts')">Total Pts ⇳</th>
            <th onclick="sortTable('avg_pts')">Avg Pts ⇳</th>
  `;

  sortedWeeks.forEach(w => {
    html += `<th onclick="sortTable('${w}')">${w} ⇳</th>`;
  });

  html += `
          </tr>
        </thead>
        <tbody>
  `;

  filteredPlayers.forEach((p, idx) => {
    const avgPts = p.total_games > 0 ? (p._active_total / p.total_games).toFixed(2) : '0.00';

    html += `
      <tr onclick="openPlayerModal('${p.playername}', '${p.year}', '${p.league}', '${p.split}')">
        <td class="rank-cell">${idx + 1}</td>
        <td>
          <div class="player-name-cell">
            <span>${escapeHtml(p.playername)}</span>
          </div>
        </td>
        <td><span class="team-badge">${escapeHtml(p.teamname)}</span></td>
        <td><span class="pos-tag ${p.position}">${p.position}</span></td>
        <td style="color: var(--text-muted);">${p.total_games}</td>
        <td style="font-weight: 800; color: var(--accent-cyan);">${p._active_total.toFixed(2)}</td>
        <td style="font-weight: 700;">${avgPts}</td>
    `;

    sortedWeeks.forEach(w => {
      const stats = p.weekly_stats[w];
      if (stats) {
        const val = pointsMode === 'adjusted' ? stats.adjusted_pts : stats.fantasy_pts;
        let badgeClass = 'low-score';
        if (val >= 40) badgeClass = 'high-score';
        else if (val >= 25) badgeClass = 'med-score';

        html += `<td><span class="weekly-pts-badge ${badgeClass}">${val.toFixed(1)}</span></td>`;
      } else {
        html += `<td><span style="color: var(--border-color);">-</span></td>`;
      }
    });

    html += `</tr>`;
  });

  html += `
        </tbody>
      </table>
    </div>
  `;

  container.innerHTML = html;
}

function sortTable(column) {
  if (currentSortCol === column) {
    currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    currentSortCol = column;
    currentSortDir = 'desc';
  }
  applyFilters();
}

function renderTrendChart() {
  const ctx = document.getElementById('trendChart').getContext('2d');
  
  if (trendChart) {
    trendChart.destroy();
  }

  const top5 = filteredPlayers.slice(0, 5);
  if (top5.length === 0) return;

  // Discover weeks
  const weekSet = new Set();
  top5.forEach(p => Object.keys(p.weekly_stats).forEach(w => weekSet.add(w)));
  const weeks = Array.from(weekSet).sort((a, b) => (parseInt(a.replace('W', '')) || 0) - (parseInt(b.replace('W', '')) || 0));

  const colors = ['#00f2fe', '#ff4d6d', '#7209b7', '#ffb703', '#00e676'];

  const datasets = top5.map((p, i) => {
    const dataPoints = weeks.map(w => {
      const s = p.weekly_stats[w];
      return s ? (pointsMode === 'adjusted' ? s.adjusted_pts : s.fantasy_pts) : null;
    });

    return {
      label: `${p.playername} (${p.teamname})`,
      data: dataPoints,
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length] + '20',
      tension: 0.3,
      fill: false,
      pointRadius: 5,
      pointHoverRadius: 8
    };
  });

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: weeks,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#f0f4fc', font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          mode: 'index',
          intersect: false
        }
      },
      scales: {
        x: {
          ticks: { color: '#8a99ad' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          title: { display: true, text: 'Fantasy Points', color: '#8a99ad' },
          ticks: { color: '#8a99ad' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });
}

function openPlayerModal(pname, year, league, split) {
  const player = rawData.players.find(p => p.playername === pname && p.year === year && p.league === league && p.split === split);
  if (!player) return;

  const content = document.getElementById('modalDetails');
  
  let weeksHtml = Object.entries(player.weekly_stats).map(([wKey, w]) => {
    return `
      <div style="background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong style="color: var(--accent-cyan);">${wKey}</strong> • ${w.games} Game(s)
          <div style="font-size: 12px; color: var(--text-muted);">KDA: ${w.kills} / ${w.deaths} / ${w.assists}</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 16px; font-weight: 800; color: var(--text-main);">${w.fantasy_pts.toFixed(1)} Pts</div>
          <div style="font-size: 11px; color: var(--accent-purple);">Adj: ${w.adjusted_pts.toFixed(1)} Pts</div>
        </div>
      </div>
    `;
  }).join('');

  content.innerHTML = `
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 20px;">
      <div style="width: 50px; height: 50px; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 20px;">
        ${player.position}
      </div>
      <div>
        <h2 style="font-size: 24px; font-weight: 800;">${escapeHtml(player.playername)}</h2>
        <div style="color: var(--text-muted); font-size: 14px;">${escapeHtml(player.teamname)} • ${player.league} ${player.year} (${player.split})</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px;">
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; text-align: center;">
        <div style="font-size: 11px; color: var(--text-muted);">TOTAL POINTS</div>
        <div style="font-size: 20px; font-weight: 800; color: var(--accent-cyan);">${player.total_fantasy_pts}</div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; text-align: center;">
        <div style="font-size: 11px; color: var(--text-muted);">AVG PTS / GAME</div>
        <div style="font-size: 20px; font-weight: 800;">${player.avg_fantasy_pts}</div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; text-align: center;">
        <div style="font-size: 11px; color: var(--text-muted);">TOTAL K / D / A</div>
        <div style="font-size: 16px; font-weight: 700;">${player.total_kills} / ${player.total_deaths} / ${player.total_assists}</div>
      </div>
    </div>

    <h4 style="margin-bottom: 12px; color: var(--text-muted); text-transform: uppercase; font-size: 12px; letter-spacing: 0.8px;">Weekly Performance Breakdown</h4>
    ${weeksHtml}
  `;

  document.getElementById('modalOverlay').classList.add('active');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('active');
}

function exportToCSV() {
  if (filteredPlayers.length === 0) return;

  const headers = ['Player', 'Team', 'Position', 'League', 'Year', 'Split', 'Games', 'Total Fantasy Pts', 'Avg Pts Per Game'];
  const rows = filteredPlayers.map(p => [
    `"${p.playername}"`,
    `"${p.teamname}"`,
    p.position,
    p.league,
    p.year,
    `"${p.split}"`,
    p.total_games,
    p._active_total.toFixed(2),
    p._active_avg.toFixed(2)
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `lcs_fantasy_weekly_stats_${new Date().toISOString().slice(0,10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
