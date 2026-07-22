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

  updateSplitDropdown();
}

function updateSplitDropdown() {
  const league = document.getElementById('leagueSelect').value;
  const year = document.getElementById('yearSelect').value;
  const splitSelect = document.getElementById('splitSelect');

  const splitSet = new Set();
  if (rawData && rawData.players) {
    rawData.players.forEach(p => {
      if ((league === 'ALL' || p.league === league) && (year === 'ALL' || p.year === year)) {
        if (p.splits && Array.isArray(p.splits)) {
          p.splits.forEach(s => splitSet.add(s));
        } else if (p.split) {
          p.split.split(', ').forEach(s => splitSet.add(s));
        }
      }
    });
  }

  const sortedSplits = Array.from(splitSet).sort();
  const currentVal = splitSelect.value;
  splitSelect.innerHTML = `<option value="ALL">All Splits & Playoffs</option>` +
    sortedSplits.map(s => `<option value="${s}">${s}</option>`).join('');

  if (sortedSplits.includes(currentVal)) {
    splitSelect.value = currentVal;
  } else {
    splitSelect.value = 'ALL';
  }
}

function setupEventListeners() {
  document.getElementById('searchInput').addEventListener('input', applyFilters);
  document.getElementById('leagueSelect').addEventListener('change', () => {
    updateSplitDropdown();
    applyFilters();
  });
  document.getElementById('yearSelect').addEventListener('change', () => {
    updateSplitDropdown();
    applyFilters();
  });
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

  // View Switcher Buttons
  document.querySelectorAll('.view-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.view-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));

      const targetBtn = e.currentTarget;
      const targetViewId = targetBtn.dataset.view;
      targetBtn.classList.add('active');

      const viewEl = document.getElementById(targetViewId);
      if (viewEl) viewEl.classList.add('active');
    });
  });

  // Price Modal close
  const priceCloseBtn = document.getElementById('priceModalCloseBtn');
  const priceModal = document.getElementById('priceModalOverlay');
  if (priceCloseBtn && priceModal) {
    priceCloseBtn.addEventListener('click', closePriceModal);
    priceModal.addEventListener('click', (e) => {
      if (e.target.id === 'priceModalOverlay') closePriceModal();
    });
  }

  // Export CSV
  document.getElementById('exportCsvBtn').addEventListener('click', exportToCSV);

  // Rules Modal setup
  const rulesBtn = document.getElementById('rulesBtn');
  const rulesModal = document.getElementById('rulesModalOverlay');
  const rulesCloseBtn = document.getElementById('rulesModalCloseBtn');

  if (rulesBtn && rulesModal) {
    rulesBtn.addEventListener('click', () => rulesModal.classList.add('active'));
    rulesCloseBtn.addEventListener('click', () => rulesModal.classList.remove('active'));
    rulesModal.addEventListener('click', (e) => {
      if (e.target.id === 'rulesModalOverlay') rulesModal.classList.remove('active');
    });

    // Rules Tabs Switching
    document.querySelectorAll('.rules-tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.rules-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.rules-tab-content').forEach(c => c.classList.remove('active'));
        
        const targetTab = e.target.dataset.tab;
        e.target.classList.add('active');
        const contentEl = document.getElementById(targetTab);
        if (contentEl) contentEl.classList.add('active');
      });
    });
  }

  // Player detail modal close
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
    if (split !== 'ALL') {
      if (p.splits && !p.splits.includes(split)) return false;
      if (!p.splits && !p.split.includes(split)) return false;
    }
    if (currentPositionFilter !== 'ALL' && p.position !== currentPositionFilter) return false;
    return true;
  });

  // Calculate dynamic sort values (split-aware)
  filteredPlayers.forEach(p => {
    let totalPts = 0;
    let totalGames = 0;

    Object.entries(p.weekly_stats).forEach(([wKey, wVal]) => {
      if (split === 'ALL' || wKey.startsWith(split) || (wVal.split && wVal.split === split)) {
        totalPts += (pointsMode === 'adjusted' ? wVal.adjusted_pts : wVal.fantasy_pts) * wVal.games;
        totalGames += wVal.games;
      }
    });

    p._active_total = totalGames > 0 ? (totalPts / totalGames) : 0; // Average per game for fantasy ranking
    p._active_sum = totalPts;
    p._active_games = totalGames;
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
    } else if (currentSortCol === 'current_price') {
      valA = a.current_price || 15.0;
      valB = b.current_price || 15.0;
    } else if (currentSortCol === 'total_price_change') {
      valA = a.total_price_change || 0;
      valB = b.total_price_change || 0;
    } else if (currentSortCol === 'avg_pts' || currentSortCol === 'total_pts') {
      valA = a._active_total;
      valB = b._active_total;
    } else if (currentSortCol.length > 0) {
      const weekKey = currentSortCol;
      valA = a.weekly_stats[weekKey] ? (pointsMode === 'adjusted' ? a.weekly_stats[weekKey].adjusted_pts : a.weekly_stats[weekKey].fantasy_pts) : 0;
      valB = b.weekly_stats[weekKey] ? (pointsMode === 'adjusted' ? b.weekly_stats[weekKey].adjusted_pts : b.weekly_stats[weekKey].fantasy_pts) : 0;
    }

    return currentSortDir === 'asc' ? valA - valB : valB - valA;
  });

  updateKPICards();
  renderTable();
  renderPriceTable();
  renderTrendChart();
}

function updateKPICards() {
  document.getElementById('totalPlayersKpi').innerText = filteredPlayers.length;
  
  if (filteredPlayers.length > 0) {
    const topPlayer = filteredPlayers[0];
    document.getElementById('topPlayerKpi').innerText = topPlayer.playername;
    document.getElementById('topPlayerSub').innerText = `${topPlayer.teamname} • ${topPlayer._active_total.toFixed(2)} Avg Pts`;

    const totalGames = filteredPlayers.reduce((acc, p) => acc + p._active_games, 0);
    const avgPts = filteredPlayers.length > 0 ? (filteredPlayers.reduce((acc, p) => acc + p._active_total, 0) / filteredPlayers.length).toFixed(2) : '0.00';
    document.getElementById('avgPtsKpi').innerText = avgPts;

    // Highest single week score
    let maxWeekScore = 0;
    let maxWeekPlayer = '-';
    filteredPlayers.forEach(p => {
      Object.entries(p.weekly_stats).forEach(([wKey, wVal]) => {
        const pts = pointsMode === 'adjusted' ? wVal.adjusted_pts : wVal.fantasy_pts;
        if (pts > maxWeekScore) {
          maxWeekScore = pts;
          maxWeekPlayer = `${p.playername} (${wKey})`;
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
  const selectedSplit = document.getElementById('splitSelect').value;

  // Discover all distinct weeks in filtered dataset
  const weekSet = new Set();
  filteredPlayers.forEach(p => {
    Object.keys(p.weekly_stats).forEach(wKey => {
      if (selectedSplit === 'ALL' || wKey.startsWith(selectedSplit)) {
        weekSet.add(wKey);
      }
    });
  });

  const sortedWeeks = Array.from(weekSet).sort((a, b) => {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
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
            <th onclick="sortTable('avg_pts')">Avg Pts / Game ⇳</th>
  `;

  sortedWeeks.forEach(w => {
    const displayLabel = selectedSplit !== 'ALL' ? w.replace(selectedSplit, '').trim() : w;
    html += `<th onclick="sortTable('${w}')">${displayLabel} ⇳</th>`;
  });

  html += `
          </tr>
        </thead>
        <tbody>
  `;

  filteredPlayers.forEach((p, idx) => {
    const avgPts = (p._active_total || 0).toFixed(2);
    const activeGames = p._active_games || p.total_games;
    const swapBadge = p.is_swapped ? `<span class="roster-swap-badge" title="Roster swap: ${p.teams.join(' ➔ ')}">🔄 Swapped</span>` : '';
    const priceSource = p.pricing_source === 'official_market_api'
      ? '<div style="font-size: 10px; color: #00e676; margin-top: 2px;">OFFICIAL API</div>'
      : '<div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">ESTIMATED</div>';

    html += `
      <tr onclick="openPlayerModal('${escapeHtml(p.playername)}', '${p.year}', '${p.league}')">
        <td class="rank-cell">${idx + 1}</td>
        <td>
          <div class="player-name-cell">
            <span>${escapeHtml(p.playername)}</span>
            ${swapBadge}
          </div>
        </td>
        <td><span class="team-badge">${escapeHtml(p.teamname)}</span></td>
        <td><span class="pos-tag ${p.position}">${p.position}</span></td>
        <td style="color: var(--text-muted);">${activeGames}</td>
        <td style="font-weight: 800; color: var(--accent-cyan);">${avgPts}</td>
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

function renderPriceTable() {
  const container = document.getElementById('priceTableContainer');
  if (!container) return;

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
            <th>Base Price</th>
            <th onclick="sortTable('current_price')">Current Market Price ⇳</th>
            <th>Latest Week Change</th>
            <th onclick="sortTable('total_price_change')">Total Season Change ⇳</th>
          </tr>
        </thead>
        <tbody>
  `;

  filteredPlayers.forEach((p, idx) => {
    const basePrice = (p.start_price || 15.0).toFixed(2);
    const currPrice = (p.current_price || 15.0).toFixed(2);
    const weeklyChg = p.latest_weekly_change || 0.0;
    const totalChg = p.total_price_change || 0.0;

    let wBadgeClass = 'neutral';
    let wPrefix = '';
    if (weeklyChg > 0) { wBadgeClass = 'up'; wPrefix = '+'; }
    else if (weeklyChg < 0) { wBadgeClass = 'down'; }

    let tBadgeClass = 'neutral';
    let tPrefix = '';
    if (totalChg > 0) { tBadgeClass = 'up'; tPrefix = '+'; }
    else if (totalChg < 0) { tBadgeClass = 'down'; }

    const swapBadge = p.is_swapped ? `<span class="roster-swap-badge" title="Roster swap: ${p.teams.join(' ➔ ')}">🔄 Swapped</span>` : '';

    html += `
      <tr onclick="openPriceModal('${escapeHtml(p.playername)}', '${p.year}', '${p.league}')">
        <td class="rank-cell">${idx + 1}</td>
        <td>
          <div class="player-name-cell">
            <span>${escapeHtml(p.playername)}</span>
            ${swapBadge}
          </div>
        </td>
        <td><span class="team-badge">${escapeHtml(p.teamname)}</span></td>
        <td><span class="pos-tag ${p.position}">${p.position}</span></td>
        <td style="color: var(--text-muted);">${basePrice}g</td>
        <td style="font-weight: 800; color: var(--accent-cyan); font-size: 15px;">${currPrice} Gold${priceSource}</td>
        <td><span class="price-badge ${wBadgeClass}">${wPrefix}${weeklyChg.toFixed(2)}g</span></td>
        <td><span class="price-badge ${tBadgeClass}">${tPrefix}${totalChg.toFixed(2)}g</span></td>
      </tr>
    `;
  });

  html += `
        </tbody>
      </table>
    </div>
  `;

  container.innerHTML = html;
}

function openPriceModal(pname, year, league) {
  const player = rawData.players.find(p => p.playername === pname && p.year === year && p.league === league);
  if (!player || !player.price_history) return;

  const selectedSplit = document.getElementById('splitSelect').value;
  const filteredHist = (player.price_history || []).filter(h => {
    return selectedSplit === 'ALL' || h.week.startsWith(selectedSplit) || h.split === selectedSplit;
  });
  const historyToUse = filteredHist.length > 0 ? filteredHist : player.price_history;

  const detailsEl = document.getElementById('priceModalDetails');
  const totalChg = player.total_price_change || 0;
  const isUp = totalChg >= 0;

  let swapNotice = '';
  if (player.is_swapped) {
    swapNotice = `
      <div style="background: rgba(255, 171, 0, 0.1); border: 1px solid rgba(255, 171, 0, 0.3); padding: 10px 14px; border-radius: 10px; margin-bottom: 16px; font-size: 13px; color: #ffab00;">
        🔄 <strong>Roster Swap History:</strong> This player moved between teams during the season: <strong>${player.teams.join(' ➔ ')}</strong>.
      </div>
    `;
  }

  let tableRows = historyToUse.map(h => {
    const chgClass = h.change > 0 ? 'up' : (h.change < 0 ? 'down' : 'neutral');
    const chgPrefix = h.change > 0 ? '+' : '';
    const tm = h.teamname || player.teamname;
    const weekLabel = selectedSplit !== 'ALL' ? h.week.replace(selectedSplit, '').trim() : h.week;
    const pointsLabel = h.pts == null ? '—' : `${h.pts.toFixed(1)} Pts`;
    return `
      <tr>
        <td><strong>${weekLabel}</strong></td>
        <td><span class="team-badge">${escapeHtml(tm)}</span></td>
        <td>${pointsLabel}</td>
        <td><span class="price-badge ${chgClass}">${chgPrefix}${h.change.toFixed(2)}g</span></td>
        <td style="font-weight: 800; color: var(--accent-cyan);">${h.price.toFixed(2)}g</td>
      </tr>
    `;
  }).join('');

  const pricingNotice = player.pricing_source === 'official_market_api'
    ? '<div style="color: #00e676; font-size: 12px; margin-top: 3px;">Official LCS Fantasy market API price</div>'
    : '<div style="color: var(--text-muted); font-size: 12px; margin-top: 3px;">Experimental estimated price; no official snapshot captured</div>';

  detailsEl.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
      <div>
        <h2 style="font-size: 22px; font-weight: 800;">💰 ${escapeHtml(player.playername)} Market Trajectory</h2>
        <div style="color: var(--text-muted); font-size: 13px;">${escapeHtml(player.teamname)} • ${player.position} • ${player.league} ${player.year} ${selectedSplit !== 'ALL' ? `(${selectedSplit})` : ''}</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 24px; font-weight: 900; color: var(--accent-cyan);">${player.current_price.toFixed(2)} Gold</div>
        ${pricingNotice}
        <div style="font-size: 13px; font-weight: 800; color: ${isUp ? '#00e676' : '#ff1744'};">
          ${isUp ? '+' : ''}${totalChg.toFixed(2)}g (${((totalChg / player.start_price)*100).toFixed(1)}%)
        </div>
      </div>
    </div>

    ${swapNotice}

    <div style="background: rgba(10, 14, 23, 0.6); padding: 16px; border-radius: 14px; border: 1px solid var(--border-color); margin-bottom: 20px; height: 260px;">
      <canvas id="priceTrajectoryChart"></canvas>
    </div>

    <h4 style="margin-bottom: 10px; color: var(--text-muted); font-size: 12px; text-transform: uppercase;">Week-by-Week Price Adjustment History</h4>
    <div class="table-wrapper" style="max-height: 200px; overflow-y: auto;">
      <table>
        <thead>
          <tr>
            <th>Week</th>
            <th>Team</th>
            <th>Fantasy Pts</th>
            <th>Price Change</th>
            <th>Ending Market Price</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    </div>
  `;

  document.getElementById('priceModalOverlay').classList.add('active');

  // Render Chart.js line graph for filtered history
  setTimeout(() => {
    const canvas = document.getElementById('priceTrajectoryChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const labels = historyToUse.map(h => selectedSplit !== 'ALL' ? h.week.replace(selectedSplit, '').trim() : h.week);
    const prices = historyToUse.map(h => h.price);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Market Price (Gold)',
          data: prices,
          borderColor: isUp ? '#00e676' : '#ff1744',
          backgroundColor: isUp ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 23, 68, 0.15)',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
          pointHoverRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Price: ${ctx.raw.toFixed(2)} Gold`
            }
          }
        },
        scales: {
          x: { ticks: { color: '#8a99ad' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#8a99ad' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }, 50);
}

function closePriceModal() {
  document.getElementById('priceModalOverlay').classList.remove('active');
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
  const selectedSplit = document.getElementById('splitSelect').value;
  
  if (trendChart) {
    trendChart.destroy();
  }

  const top5 = filteredPlayers.slice(0, 5);
  if (top5.length === 0) return;

  // Discover weeks filtered by selectedSplit
  const weekSet = new Set();
  top5.forEach(p => {
    Object.keys(p.weekly_stats).forEach(wKey => {
      if (selectedSplit === 'ALL' || wKey.startsWith(selectedSplit) || (p.weekly_stats[wKey].split && p.weekly_stats[wKey].split === selectedSplit)) {
        weekSet.add(wKey);
      }
    });
  });

  const weeks = Array.from(weekSet).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

  if (weeks.length === 0) return;

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

  const displayLabels = weeks.map(w => selectedSplit !== 'ALL' ? w.replace(selectedSplit, '').trim() : w);

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: displayLabels,
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

function openPlayerModal(pname, year, league) {
  const player = rawData.players.find(p => p.playername === pname && p.year === year && p.league === league);
  if (!player) return;

  const selectedSplit = document.getElementById('splitSelect').value;
  const content = document.getElementById('modalDetails');
  
  let swapNotice = '';
  if (player.is_swapped) {
    swapNotice = `
      <div style="background: rgba(255, 171, 0, 0.1); border: 1px solid rgba(255, 171, 0, 0.3); padding: 10px 14px; border-radius: 10px; margin-bottom: 16px; font-size: 13px; color: #ffab00;">
        🔄 <strong>Roster Swap History:</strong> Swapped between <strong>${player.teams.join(' ➔ ')}</strong>.
      </div>
    `;
  }

  const filteredEntries = Object.entries(player.weekly_stats).filter(([wKey, w]) => {
    return selectedSplit === 'ALL' || wKey.startsWith(selectedSplit) || w.split === selectedSplit;
  });
  const entriesToDisplay = filteredEntries.length > 0 ? filteredEntries : Object.entries(player.weekly_stats);

  let weeksHtml = entriesToDisplay.map(([wKey, w]) => {
    const displayLabel = selectedSplit !== 'ALL' ? wKey.replace(selectedSplit, '').trim() : wKey;
    return `
      <div style="background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong style="color: var(--accent-cyan);">${displayLabel}</strong> • <span class="team-badge">${escapeHtml(w.teamname || player.teamname)}</span> • ${w.games} Game(s)
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

    ${swapNotice}

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px;">
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; text-align: center;">
        <div style="font-size: 11px; color: var(--text-muted);">CURRENT MARKET PRICE</div>
        <div style="font-size: 20px; font-weight: 800; color: var(--accent-cyan);">${(player.current_price || 15.0).toFixed(2)}g</div>
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

  const headers = ['Player', 'Team', 'Position', 'League', 'Year', 'Split', 'Current Price (Gold)', 'Total Price Change', 'Games', 'Total Fantasy Pts', 'Avg Pts Per Game'];
  const rows = filteredPlayers.map(p => [
    `"${p.playername}"`,
    `"${p.teamname}"`,
    p.position,
    p.league,
    p.year,
    `"${p.split}"`,
    (p.current_price || 15.0).toFixed(2),
    (p.total_price_change || 0).toFixed(2),
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
