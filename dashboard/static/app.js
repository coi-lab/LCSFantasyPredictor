// LCS Fantasy Interactive Weekly Dashboard Logic

let rawData = null;
let championLabData = null;
let weeklyChampionData = null;
let matchupOptimizerData = null;
let historicalLineupData = null;
let evalDevSummary = null;
let evalWeeklyResults = null;
let evalLeaderboard = null;
let evalProvenance = null;
let evalM3Diagnostics = null;
let m3DiagFilters = { week: 'ALL', role: 'ALL', team: 'ALL', search: '' };
let m3DiagSort = { col: 'absolute_error', dir: 'desc' };
let selectedDiagPlayerPeriod = null;
let m3GroupTabActive = 'week';
let m3DiagCurrentPage = 1;
const m3DiagPageSize = 15;
let selectedEvalWeekNum = 1;
let selectedMatchupLineupRank = 1;
let filteredPlayers = [];
let currentPositionFilter = 'ALL';
let currentSortCol = 'total_pts';
let currentSortDir = 'desc';
let pointsMode = 'raw'; // 'raw' or 'adjusted'
let trendChart = null;
let championPoolChart = null;
let championSplitChart = null;
let devProgressionChart = null;

const TEAM_COLORS = {
  '100 Thieves': '#e31b23',
  'Cloud9': '#00aeef',
  'Dignitas': '#ffe600',
  'Disguised': '#a66a3f',
  'FlyQuest': '#2ecc71',
  'Immortals': '#00e5ff',
  'LYON': '#d4af37',
  'Sentinels': '#e31b23',
  'Shopify Rebellion': '#39ff14',
  'Team Liquid': '#3b82f6',
  'TSM': '#f8fafc',

  // Provisional historical LCS colors; easy to update after confirmation.
  'Counter Logic Gaming': '#5dade2',
  'Evil Geniuses': '#00a88f',
  'Golden Guardians': '#ff9e1b',
  'NRG': '#ff5c35'
};

const TEAM_ALIASES = {
  'Cloud9 Kia': 'Cloud9',
  'Team Liquid Alienware': 'Team Liquid'
};

const FALLBACK_TEAM_COLORS = [
  '#9b5de5', '#f15bb5', '#00bbf9', '#00f5d4', '#f97316',
  '#a3e635', '#fb7185', '#818cf8', '#22d3ee', '#c084fc'
];

function getTeamColor(teamName) {
  const rawName = String(teamName || 'Unknown').trim();
  const canonicalName = TEAM_ALIASES[rawName] || rawName;
  if (TEAM_COLORS[canonicalName]) return TEAM_COLORS[canonicalName];

  let hash = 0;
  for (let i = 0; i < canonicalName.length; i += 1) {
    hash = ((hash << 5) - hash + canonicalName.charCodeAt(i)) | 0;
  }
  return FALLBACK_TEAM_COLORS[Math.abs(hash) % FALLBACK_TEAM_COLORS.length];
}

function colorWithAlpha(hex, alphaHex = '26') {
  return /^#[0-9a-f]{6}$/i.test(hex) ? `${hex}${alphaHex}` : hex;
}

const patchBoundaryPlugin = {
  id: 'patchBoundaries',
  afterDraw(chart, _args, options) {
    const markers = options && options.markers;
    if (!markers || !markers.initialPatch || !chart.chartArea || !chart.scales.x) return;

    const { ctx, chartArea, scales } = chart;
    ctx.save();
    ctx.font = '700 11px Inter, sans-serif';
    ctx.fillStyle = '#ffb703';
    ctx.textBaseline = 'top';
    ctx.fillText(`Patch ${markers.initialPatch}`, chartArea.left + 6, chartArea.top + 6);

    markers.boundaries.forEach(boundary => {
      const previousX = scales.x.getPixelForValue(Math.max(0, boundary.index - 1));
      const currentX = scales.x.getPixelForValue(boundary.index);
      const x = (previousX + currentX) / 2;

      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#ffb703';
      ctx.lineWidth = 2;
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      const label = `Patch ${boundary.patch}`;
      const labelWidth = ctx.measureText(label).width + 10;
      ctx.fillStyle = 'rgba(10, 14, 23, 0.9)';
      ctx.fillRect(x + 4, chartArea.top + 4, labelWidth, 18);
      ctx.fillStyle = '#ffb703';
      ctx.fillText(label, x + 9, chartArea.top + 7);
    });
    ctx.restore();
  }
};

function buildPatchMarkers(entries) {
  let activePatch = null;
  const boundaries = [];

  entries.forEach((entry, index) => {
    const patch = String(entry && entry.patch || '').trim();
    if (!patch) return;
    if (activePatch === null) {
      activePatch = patch;
    } else if (patch !== activePatch) {
      boundaries.push({ index, patch });
      activePatch = patch;
    }
  });

  return {
    initialPatch: entries.map(entry => String(entry && entry.patch || '').trim()).find(Boolean) || null,
    boundaries
  };
}

document.addEventListener('DOMContentLoaded', async () => {
  addChampionLabTab();
  // Make the audit route visible immediately; its small payload loads
  // independently of the much larger player-history dashboard JSON.
  if (window.location.hash === '#historical-lineups') {
    document.querySelectorAll('.view-tab-btn').forEach(button => button.classList.remove('active'));
    document.querySelectorAll('.view-section').forEach(section => section.classList.remove('active'));
    document.querySelector('[data-view="view-historical-lineups"]')?.classList.add('active');
    document.getElementById('view-historical-lineups')?.classList.add('active');
  }
  await loadDashboardData();
  setupEventListeners();
  if (window.location.hash === '#weekly-champions') {
    document.querySelector(
      '[data-view="view-weekly-champions"]'
    )?.click();
  } else if (window.location.hash === '#matchup-optimizer') {
    document.querySelector(
      '[data-view="view-matchup-optimizer"]'
    )?.click();
  } else if (window.location.hash === '#historical-lineups') {
    document.querySelector(
      '[data-view="view-historical-lineups"]'
    )?.click();
  }
});

function addChampionLabTab() {
  const tabRow = document.querySelector('.view-tabs');
  if (!tabRow || tabRow.querySelector('[data-view="view-champion-lab"]')) return;
  const button = document.createElement('button');
  button.className = 'view-tab-btn';
  button.dataset.view = 'view-champion-lab';
  button.innerHTML = '<span>Champion Lab</span>';
  tabRow.appendChild(button);
}

async function loadDashboardData() {
  const historicalRequest = fetch('../generated/current/historical_lineups.json')
    .then(async response => {
      historicalLineupData = response.ok ? await response.json() : { phases: [] };
      populateHistoricalLineupControls();
      renderHistoricalLineups();
    })
    .catch(error => {
      console.error('Error loading historical lineup data:', error);
      historicalLineupData = { phases: [] };
      renderHistoricalLineups();
    });
  try {
    const [
      resp,
      championResp,
      weeklyChampionResp,
      matchupOptimizerResp
    ] = await Promise.all([
      fetch('../generated/current/dashboard_data.json'),
      fetch('../generated/current/champion_lab_data.json'),
      fetch('../generated/current/weekly_champion_predictions.json'),
      fetch('../generated/current/matchup_lineups.json')
    ]);
    if (!resp.ok) throw new Error('Could not load dashboard_data.json');
    rawData = await resp.json();
    championLabData = championResp.ok
      ? await championResp.json()
      : { profiles: [], players: [] };
    weeklyChampionData = weeklyChampionResp.ok
      ? await weeklyChampionResp.json()
      : { players: [] };
    matchupOptimizerData = matchupOptimizerResp.ok
      ? await matchupOptimizerResp.json()
      : { weeks: [] };

    populateFilterDropdowns();
    applyFilters();
    renderWeeklyChampionPicks();
    populateMatchupWeekSelect();
    renderMatchupOptimizer();

    // Fetch model evaluation data
    const evalRequest = Promise.all([
      fetch('../generated/current/model-development-summary.json').then(r => r.ok ? r.json() : null),
      fetch('../generated/current/stage7-weekly-results.json').then(r => r.ok ? r.json() : null),
      fetch('../generated/current/stage7-leaderboard-comparison.json').then(r => r.ok ? r.json() : null),
      fetch('../generated/current/stage7-provenance.json').then(r => r.ok ? r.json() : null),
      fetch('../generated/current/m3-player-diagnostics.json').then(r => r.ok ? r.json() : null)
    ]).then(([dev, weekly, lb, prov, diag]) => {
      evalDevSummary = dev;
      evalWeeklyResults = weekly;
      evalLeaderboard = lb;
      evalProvenance = prov;
      evalM3Diagnostics = diag;
      setupEvalTabs();
      renderModelEvaluation();
    }).catch(error => {
      console.error('Error loading model evaluation data:', error);
    });

    await Promise.all([historicalRequest, evalRequest]);
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
  const sortedYears = [...rawData.years].sort().reverse();
  const years = ['ALL', ...sortedYears];
  yearSelect.innerHTML = years.map(y => `<option value="${y}">${y === 'ALL' ? 'All Years' : y}</option>`).join('');
  if (sortedYears.length > 0) {
    yearSelect.value = sortedYears[0];
  }

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
      if (targetViewId === 'view-champion-lab') {
        const yearSelect = document.getElementById('yearSelect');
        if (Number(yearSelect.value) > 2025 || yearSelect.value === 'ALL') {
          yearSelect.value = '2025';
          updateSplitDropdown();
          applyFilters();
        } else {
          renderChampionLab();
        }
      } else if (targetViewId === 'view-weekly-champions') {
        renderWeeklyChampionPicks();
      } else if (targetViewId === 'view-matchup-optimizer') {
        renderMatchupOptimizer();
      } else if (targetViewId === 'view-historical-lineups') {
        renderModelEvaluation();
      }
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
  document.getElementById('championPlayerSelect').addEventListener('change', renderChampionLab);
  document.getElementById('matchupWeekSelect')?.addEventListener('change', () => {
    selectedMatchupLineupRank = 1;
    renderMatchupOptimizer();
  });
  document.getElementById('historicalPhaseSelect')?.addEventListener('change', () => {
    populateHistoricalPolicySelect();
    populateHistoricalWeekSelect();
    renderHistoricalLineups();
  });
  document.getElementById('historicalPolicySelect')?.addEventListener('change', () => {
    populateHistoricalWeekSelect();
    renderHistoricalLineups();
  });
  document.getElementById('historicalWeekSelect')?.addEventListener('change', renderHistoricalLineups);
}

function renderWeeklyChampionPicks() {
  const container = document.getElementById('weeklyChampionMatchups');
  const notice = document.getElementById('weeklyChampionNotice');
  const validationContainer = document.getElementById('weeklyChampionValidation');
  if (!container || !notice) return;
  const players = weeklyChampionData && Array.isArray(weeklyChampionData.players)
    ? weeklyChampionData.players
    : [];
  if (players.length === 0) {
    notice.textContent = 'No current weekly champion predictions are available.';
    container.innerHTML = '';
    return;
  }

  document.getElementById('weeklyChampionTitle').textContent =
    `${weeklyChampionData.round_name || 'Current Round'} Champion Picks`;
  document.getElementById('weeklyChampionMeta').textContent =
    `Patch ${weeklyChampionData.patch || 'unknown'} | Roster lock ${weeklyChampionData.roster_lock || 'unknown'} | ${players.length} projected starters`;

  const tierAvailability = ['1.3x', '1.5x', '1.7x'].map(tier =>
    players.some(player => player.picks && player.picks[tier] && player.picks[tier].available)
  );
  notice.textContent = tierAvailability[0] || tierAvailability[1]
    ? 'The first option is the highest-ranked choice. Percentages are relative ranking strength, not calibrated odds; use the measured backtest and evidence labels below.'
    : 'No eligible champion predictions are available for this round.';
  const validation = weeklyChampionData.validation || {};
  if (validationContainer) {
    validationContainer.innerHTML = validation.sample_size ? `
      <div class="weekly-validation-card">
        <span>Measured Hit@1</span>
        <strong>${(Number(validation.hit_at_1) * 100).toFixed(1)}%</strong>
        <small>${escapeHtml(validation.label || 'Held-out backtest')}</small>
      </div>
      <div class="weekly-validation-card">
        <span>Measured Hit@3</span>
        <strong>${(Number(validation.hit_at_3) * 100).toFixed(1)}%</strong>
        <small>${Number(validation.sample_size)} player-weeks</small>
      </div>
      <div class="weekly-validation-card warning">
        <span>Interpretation</span>
        <strong>Pool &gt; ordering</strong>
        <small>${escapeHtml(
          (validation.by_split_week || []).length === 1
            ? `Only Split Week ${validation.by_split_week[0].split_week} is represented; no improving trend can be inferred.`
            : validation.warning || ''
        )}</small>
      </div>
    ` : '';
  }

  const matchupGroups = new Map();
  players.forEach(player => {
    const teams = [String(player.team), String(player.opponent)].sort();
    const key = teams.join(' vs ');
    if (!matchupGroups.has(key)) matchupGroups.set(key, []);
    matchupGroups.get(key).push(player);
  });

  const renderTier = (player, tier, cssClass) => {
    const entry = player.picks && player.picks[tier];
    if (!entry || !entry.available || !entry.pick) {
      return `<td class="weekly-pick unavailable"><span>Not available</span><small>Current split history</small></td>`;
    }
    const options = Array.isArray(entry.options) && entry.options.length
      ? entry.options
      : [entry.pick];
    return `
      <td class="weekly-pick ${cssClass}">
        <div class="weekly-ordering-note">
          ${escapeHtml(entry.ordering_confidence || 'low')} ordering confidence
          · ${((Number(entry.ordering_margin) || 0) * 100).toFixed(1)} point lead
        </div>
        <div class="weekly-pick-options">
          ${options.map((pick, index) => `
            <div class="weekly-pick-option">
              <span class="weekly-pick-rank">${escapeHtml(pick.option_basis || `#${index + 1}`)}</span>
              <div>
                <strong>${escapeHtml(pick.champion)}</strong>
                <span>${(Number(pick.estimated_pick_chance ?? pick.ranking_share) * 100).toFixed(1)}% relative ranking strength</span>
                <div class="weekly-context-flags">
                  <span class="confidence-${escapeHtml(pick.confidence || 'low')}">${escapeHtml(pick.confidence || 'low')} evidence</span>
                  ${(pick.flags || []).map(flag => `<span>${escapeHtml(flag)}</span>`).join('')}
                </div>
                <small>Available ${(Number(pick.availability) * 100).toFixed(0)}% · Bonus ${Number(pick.expected_multiplier_bonus).toFixed(2)}</small>
                <details class="weekly-pick-explanation">
                  <summary>Why this candidate?</summary>
                  ${(pick.explanations || []).map(line => `<p>${escapeHtml(line)}</p>`).join('')}
                </details>
              </div>
            </div>
          `).join('')}
        </div>
      </td>
    `;
  };

  container.innerHTML = Array.from(matchupGroups.entries()).map(([matchup, group]) => `
    <section class="card weekly-matchup-card">
      <div class="weekly-matchup-title">
        <h3>${escapeHtml(matchup)}</h3>
        <span>${group.length} projected starters</span>
      </div>
      <div class="table-responsive">
        <table class="weekly-picks-table">
          <thead>
            <tr><th>Player</th><th>Team / Role</th><th>x1.3 Opening / Comfort</th><th>x1.5 Adoption</th><th>x1.7 Novelty</th></tr>
          </thead>
          <tbody>
            ${group.sort((a, b) =>
              String(a.team).localeCompare(String(b.team)) ||
              String(a.role).localeCompare(String(b.role))
            ).map(player => `
              <tr>
                <td><strong>${escapeHtml(player.player)}</strong><small>Proj. ${Number(player.projected_fantasy_points).toFixed(2)} pts</small></td>
                <td><span class="team-badge">${escapeHtml(player.team)}</span><small>${escapeHtml(player.role)} vs ${escapeHtml(player.opponent)}</small></td>
                ${renderTier(player, '1.3x', 'tier-13-cell')}
                ${renderTier(player, '1.5x', 'tier-15-cell')}
                ${renderTier(player, '1.7x', 'tier-17-cell')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </section>
  `).join('');
}

function populateMatchupWeekSelect() {
  const select = document.getElementById('matchupWeekSelect');
  if (!select) return;
  const weeks = matchupOptimizerData && Array.isArray(matchupOptimizerData.weeks)
    ? matchupOptimizerData.weeks
    : [];
  select.innerHTML = weeks.slice().reverse().map(week => `
    <option value="${escapeHtml(week.week_id)}">${escapeHtml(week.round_name)}</option>
  `).join('');
}

function selectMatchupLineup(rank) {
  selectedMatchupLineupRank = Number(rank) || 1;
  renderMatchupOptimizer();
}

function matchupChampionOptions(player, preferWeekly = false) {
  let rawOptions = [];
  if (preferWeekly) {
    const weeklyPlayers = weeklyChampionData && Array.isArray(weeklyChampionData.players)
      ? weeklyChampionData.players
      : [];
    const match = weeklyPlayers.find(candidate =>
      String(candidate.player).toLowerCase() === String(player.player).toLowerCase() &&
      String(candidate.team).toLowerCase() === String(player.team).toLowerCase()
    );
    if (match && match.picks) {
      ['1.3x', '1.5x', '1.7x'].forEach(multiplier => {
        const tier = match.picks[multiplier];
        if (!tier || !tier.available) return;
        const tierOptions = Array.isArray(tier.options) ? tier.options : [];
        tierOptions.forEach(pick => rawOptions.push({ ...pick, multiplier }));
      });
    }
  }
  if (!rawOptions.length && Array.isArray(player.champion_options)) {
    rawOptions = player.champion_options;
  }

  if (!rawOptions.length) return [];

  // Deduplicate by champion name (keeping highest pick chance), then sort descending by pick chance and take top 3
  const champMap = new Map();
  rawOptions.forEach(opt => {
    const key = String(opt.champion).toLowerCase();
    const chance = Number(opt.estimated_pick_chance ?? opt.ranking_share ?? 0);
    if (!champMap.has(key) || chance > Number(champMap.get(key).estimated_pick_chance ?? champMap.get(key).ranking_share ?? 0)) {
      champMap.set(key, opt);
    }
  });

  return Array.from(champMap.values())
    .sort((a, b) => Number(b.estimated_pick_chance ?? b.ranking_share ?? 0) - Number(a.estimated_pick_chance ?? a.ranking_share ?? 0))
    .slice(0, 3);
}

function renderMatchupOptimizer() {
  const content = document.getElementById('matchupOptimizerContent');
  const notice = document.getElementById('matchupOptimizerNotice');
  const tabs = document.getElementById('matchupLineupTabs');
  const meta = document.getElementById('matchupOptimizerMeta');
  const weekSelect = document.getElementById('matchupWeekSelect');
  if (!content || !notice || !tabs || !meta || !weekSelect) return;

  const weeks = matchupOptimizerData && Array.isArray(matchupOptimizerData.weeks)
    ? matchupOptimizerData.weeks
    : [];
  if (!weeks.length) {
    notice.textContent = 'No optimized matchup lineups are available.';
    tabs.innerHTML = '';
    content.innerHTML = '';
    return;
  }

  const selectedWeek = weeks.find(week => week.week_id === weekSelect.value)
    || weeks[weeks.length - 1];
  if (!weekSelect.value) weekSelect.value = selectedWeek.week_id;
  const lineups = Array.isArray(selectedWeek.lineups) ? selectedWeek.lineups : [];
  const lineup = lineups.find(item => Number(item.rank) === selectedMatchupLineupRank)
    || lineups[0];
  if (!lineup) {
    notice.textContent = 'No legal lineup was found for this week and budget.';
    tabs.innerHTML = '';
    content.innerHTML = '';
    return;
  }

  selectedMatchupLineupRank = Number(lineup.rank);
  const currentRecommendationWeek = (
    String(selectedWeek.round_name) === String(weeklyChampionData?.round_name)
    && String(selectedWeek.roster_lock) === String(weeklyChampionData?.roster_lock)
  );
  meta.textContent = `${selectedWeek.round_name} | Roster lock ${selectedWeek.roster_lock} | ${Number(selectedWeek.budget).toFixed(1)} gold budget`;
  notice.textContent =
    'Lineups are ranked by projected points after matchup-conflict risk. Current-week champions use the exact Weekly Champion Picks ordering; archived weeks retain their frozen recommendations. Percentages are relative ranking strength, not calibrated odds.';
  tabs.innerHTML = lineups.map(item => `
    <button
      class="matchup-lineup-tab ${Number(item.rank) === selectedMatchupLineupRank ? 'active' : ''}"
      onclick="selectMatchupLineup(${Number(item.rank)})"
    >
      Lineup ${Number(item.rank)}
      <small>${Number(item.risk_adjusted_points ?? item.projected_total_points).toFixed(1)} risk pts</small>
    </button>
  `).join('');

  const renderChampionPicks = (player, opponentPlayer, isOpponentColumn = false) => {
    const options = matchupChampionOptions(player, currentRecommendationWeek);
    if (!options.length) {
      return '<div class="optimizer-no-picks">No champion recommendations available</div>';
    }
    const oppOptions = opponentPlayer
      ? matchupChampionOptions(opponentPlayer, currentRecommendationWeek)
      : [];
    const oppChamps = new Set(oppOptions.map(o => String(o.champion).toLowerCase()));

    return `
      <div class="optimizer-champion-list">
        ${options.map(pick => {
          const champLower = String(pick.champion).toLowerCase();
          const isConflict = oppChamps.has(champLower);
          const oppMatchingPick = oppOptions.find(o => String(o.champion).toLowerCase() === champLower);
          const oppChance = oppMatchingPick ? (Number(oppMatchingPick.estimated_pick_chance ?? oppMatchingPick.ranking_share) * 100).toFixed(0) : 0;

          return `
            <div class="optimizer-champion-pick ${isConflict ? 'has-collision' : 'clean-pick'}">
              <span class="tier-chip tier-${String(pick.multiplier).replace('.', '').replace('x', '')}">${escapeHtml(pick.multiplier)}</span>
              <div>
                <div class="optimizer-champ-name-row">
                  <strong>${escapeHtml(pick.champion)}</strong>
                  ${isConflict ? `<span class="collision-badge" title="High contest risk: Opponent also has ${oppChance}% pick chance">⚠️ Shared (${oppChance}%)</span>` : '<span class="unique-badge">✓ Uncontested</span>'}
                </div>
                <small>${escapeHtml(pick.option_basis || '')} ${(Number(pick.estimated_pick_chance ?? pick.ranking_share) * 100).toFixed(1)}% ranking strength</small>
                <div class="weekly-context-flags">
                  <span class="confidence-${escapeHtml(pick.confidence || 'low')}">${escapeHtml(pick.confidence || 'low')} evidence</span>
                  ${(pick.flags || []).map(flag => `<span>${escapeHtml(flag)}</span>`).join('')}
                </div>
                ${(pick.explanations || []).length ? `
                  <details class="weekly-pick-explanation">
                    <summary>Why?</summary>
                    ${pick.explanations.map(line => `<p>${escapeHtml(line)}</p>`).join('')}
                  </details>
                ` : ''}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  };

  const findOpposingPlayer = (player) => {
    const weeklyPlayers = weeklyChampionData && Array.isArray(weeklyChampionData.players)
      ? weeklyChampionData.players
      : [];
    return weeklyPlayers.find(candidate =>
      String(candidate.role).toLowerCase() === String(player.role).toLowerCase() &&
      String(candidate.team).toLowerCase() === String(player.opponent).toLowerCase()
    );
  };

  const rosterCards = lineup.players.map(player => {
    const opponentPlayer = findOpposingPlayer(player);
    const playerPicksHtml = renderChampionPicks(player, opponentPlayer, false);
    const opponentPicksHtml = opponentPlayer
      ? renderChampionPicks(opponentPlayer, player, true)
      : '<div class="optimizer-no-picks">Opponent data unavailable</div>';

    return `
      <article class="card optimizer-roster-card">
        <div class="optimizer-card-head">
          <span class="optimizer-role">${escapeHtml(String(player.role).toUpperCase())}</span>
          <span class="optimizer-price">${Number(player.price).toFixed(1)}g</span>
        </div>
        <h3>${escapeHtml(player.player)}</h3>
        <p class="optimizer-matchup">
          <span style="--team-color:${getTeamColor(player.team)}">${escapeHtml(player.team)}</span>
          vs ${escapeHtml(player.opponent || 'TBD')}
        </p>
        <div class="optimizer-point-line">
          <span>Player projection</span>
          <strong>${Number(player.projected_points).toFixed(2)}</strong>
        </div>
        <p class="optimizer-coach-note">Estimated team win probability: ${(Number(player.team_win_probability ?? 0.5) * 100).toFixed(1)}% (${escapeHtml(player.win_probability_source || 'not available')}); projection adjustment: ${Number(player.win_probability_adjustment || 0).toFixed(2)} pts.</p>
        ${player.carry_concentration_enabled ? `<p class="optimizer-coach-note">Win/loss carry estimate: ${Number(player.carry_score_if_win).toFixed(2)} in wins / ${Number(player.carry_score_if_loss).toFixed(2)} in losses; current-team win fantasy share: ${(Number(player.carry_win_fantasy_share || 0) * 100).toFixed(1)}%.</p>` : ''}
        <div class="optimizer-point-line">
          <span>Floor / Ceiling range</span>
          <strong>${player.floor_pts != null ? Number(player.floor_pts).toFixed(1) : '-'} – ${player.ceiling_pts != null ? Number(player.ceiling_pts).toFixed(1) : '-'} pts</strong>
        </div>
        <div class="optimizer-point-line">
          <span>Champion upside</span>
          <strong>+${Number(player.champion_expected_bonus || 0).toFixed(2)}</strong>
        </div>

        <div class="optimizer-picks-comparison">
          <div class="optimizer-picks-column">
            <div class="optimizer-pick-title">${escapeHtml(player.player)}'s Picks</div>
            ${playerPicksHtml}
          </div>
          <div class="optimizer-picks-column opponent-picks-column">
            <div class="optimizer-pick-title">${opponentPlayer ? escapeHtml(opponentPlayer.player) : 'Opponent'}'s Picks (${escapeHtml(player.opponent || '')})</div>
            ${opponentPicksHtml}
          </div>
        </div>
      </article>
    `;
  }).join('');

  const coach = lineup.coach;
  const matchupConflicts = Array.isArray(lineup.matchup_conflicts)
    ? lineup.matchup_conflicts
    : [];
  const conflictDetails = matchupConflicts.length
    ? `
      <div class="optimizer-conflict-list">
        ${matchupConflicts.map(conflict => `
          <span>
            ${escapeHtml(conflict.first.name)} (${escapeHtml(String(conflict.first.role).toUpperCase())})
            vs ${escapeHtml(conflict.second.name)} (${escapeHtml(String(conflict.second.role).toUpperCase())})
            <strong>-${Number(conflict.penalty).toFixed(1)}</strong>
          </span>
        `).join('')}
      </div>
    `
    : '<div class="optimizer-no-picks">No selected slots oppose one another.</div>';
  content.innerHTML = `
    <div class="optimizer-summary-grid">
      <div class="optimizer-summary-card"><span>Projected total</span><strong>${Number(lineup.projected_total_points).toFixed(2)}</strong></div>
      <div class="optimizer-summary-card"><span>Risk-adjusted rank score</span><strong>${Number(lineup.risk_adjusted_points ?? lineup.projected_total_points).toFixed(2)}</strong><small>After matchup conflicts</small></div>
      <div class="optimizer-summary-card"><span>Roster cost</span><strong>${Number(lineup.total_cost).toFixed(1)}g</strong><small>${Number(lineup.remaining_gold).toFixed(1)}g left</small></div>
      <div class="optimizer-summary-card"><span>Variety buff</span><strong>+${(Number(lineup.variety_bonus) * 100).toFixed(0)}%</strong><small>${Number(lineup.unique_teams)} teams</small></div>
      <div class="optimizer-summary-card"><span>Matchup risk</span><strong>-${Number(lineup.matchup_conflict_penalty || 0).toFixed(2)}</strong><small>${matchupConflicts.length} opposing slot pair${matchupConflicts.length === 1 ? '' : 's'}</small></div>
    </div>
    <div class="card optimizer-conflict-card">
      <div class="optimizer-pick-title">Head-to-head exposure</div>
      ${conflictDetails}
    </div>
    <div class="optimizer-roster-grid">
      ${rosterCards}
      <article class="card optimizer-roster-card optimizer-coach-card">
        <div class="optimizer-card-head">
          <span class="optimizer-role">COACH</span>
          <span class="optimizer-price">${Number(coach.price).toFixed(1)}g</span>
        </div>
        <h3>${escapeHtml(coach.coach)}</h3>
        <p class="optimizer-matchup">
          <span style="--team-color:${getTeamColor(coach.team)}">${escapeHtml(coach.team)}</span>
          vs ${escapeHtml(coach.opponent || 'TBD')}
        </p>
        <div class="optimizer-point-line">
          <span>Team-average projection</span>
          <strong>${Number(coach.projected_points).toFixed(2)}</strong>
        </div>
        <p class="optimizer-coach-note">Estimated team win probability: ${(Number(coach.team_win_probability ?? 0.5) * 100).toFixed(1)}%. Conditional estimate: ${Number(coach.projected_score_if_win ?? coach.projected_points).toFixed(2)} in wins / ${Number(coach.projected_score_if_loss ?? coach.projected_points).toFixed(2)} in losses. The coach's organization counts toward variety.</p>
      </article>
    </div>
  `;
}

function historicalPhases() {
  return historicalLineupData && Array.isArray(historicalLineupData.phases)
    ? historicalLineupData.phases
    : [];
}

function selectedHistoricalPhase() {
  const phases = historicalPhases();
  const selectedId = document.getElementById('historicalPhaseSelect')?.value;
  return phases.find(phase => phase.phase_id === selectedId) || phases[0] || null;
}

function selectedHistoricalPolicy() {
  const phase = selectedHistoricalPhase();
  const policies = phase && Array.isArray(phase.policies) ? phase.policies : [];
  const selectedId = document.getElementById('historicalPolicySelect')?.value;
  return policies.find(policy => policy.policy_id === selectedId) || policies[0] || null;
}

function populateHistoricalLineupControls() {
  const select = document.getElementById('historicalPhaseSelect');
  if (!select) return;
  const phases = historicalPhases();
  const prior = select.value;
  select.innerHTML = phases.map(phase =>
    `<option value="${escapeHtml(phase.phase_id)}">${escapeHtml(phase.label)}</option>`
  ).join('');
  if (phases.some(phase => phase.phase_id === prior)) select.value = prior;
  populateHistoricalPolicySelect();
  populateHistoricalWeekSelect();
}

function populateHistoricalPolicySelect() {
  const select = document.getElementById('historicalPolicySelect');
  if (!select) return;
  const phase = selectedHistoricalPhase();
  const policies = phase && Array.isArray(phase.policies) ? phase.policies : [];
  const prior = select.value;
  select.innerHTML = policies.map(policy =>
    `<option value="${escapeHtml(policy.policy_id)}">${escapeHtml(policy.label)}</option>`
  ).join('');
  if (policies.some(policy => policy.policy_id === prior)) select.value = prior;
}

function populateHistoricalWeekSelect() {
  const select = document.getElementById('historicalWeekSelect');
  if (!select) return;
  const policy = selectedHistoricalPolicy();
  const weeks = policy && Array.isArray(policy.weeks) ? policy.weeks : [];
  const prior = select.value;
  select.innerHTML = weeks.map((week, index) => `
    <option value="${escapeHtml(week.week_id)}">
      ${escapeHtml(week.round_name || `Week ${index + 1}`)}
    </option>
  `).join('');
  if (weeks.some(week => week.week_id === prior)) select.value = prior;
}

function formatHistoricalNumber(value, digits = 2, unavailable = 'Unavailable') {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? unavailable
    : Number(value).toFixed(digits);
}

function renderHistoricalLineups() {
  const notice = document.getElementById('historicalLineupNotice');
  const summary = document.getElementById('historicalLineupSummary');
  const content = document.getElementById('historicalLineupContent');
  const weekSelect = document.getElementById('historicalWeekSelect');
  if (!notice || !summary || !content || !weekSelect) return;

  const phase = selectedHistoricalPhase();
  const policy = selectedHistoricalPolicy();
  const weeks = policy && Array.isArray(policy.weeks) ? policy.weeks : [];
  const week = weeks.find(item => item.week_id === weekSelect.value) || weeks[0];
  if (!phase || !policy || !week) {
    notice.textContent = 'No historical lineup audit is available. Run the historical lineup dashboard exporter.';
    summary.innerHTML = '';
    content.innerHTML = '<div class="card historical-empty">No weekly lineup records found.</div>';
    return;
  }
  if (!weekSelect.value) weekSelect.value = week.week_id;

  notice.innerHTML = `
    <strong>${escapeHtml(phase.price_status)}</strong><br>
    ${escapeHtml(phase.champion_status)}<br>
    ${escapeHtml(historicalLineupData.budget_notice || '')}
  `;
  const relativeValue = week.winner_relative != null
    ? `${(Number(week.winner_relative) * 100).toFixed(2)}%`
    : (week.opportunity_capture != null
      ? `${(Number(week.opportunity_capture) * 100).toFixed(2)}%`
      : 'Unavailable');
  const relativeLabel = week.winner_relative != null ? 'Relative to first' : 'Opportunity capture';
  const budget = week.budget || {};
  summary.innerHTML = `
    <div class="historical-summary-card"><span>Phase</span><strong>${escapeHtml(phase.category.replace('_', ' '))}</strong><small>${escapeHtml(policy.label)}</small></div>
    <div class="historical-summary-card"><span>Weekly score</span><strong>${formatHistoricalNumber(week.score)}</strong><small>Champion and variety included when available</small></div>
    <div class="historical-summary-card"><span>${escapeHtml(relativeLabel)}</span><strong>${escapeHtml(relativeValue)}</strong><small>${week.oracle_score != null ? `Oracle ${formatHistoricalNumber(week.oracle_score)} pts` : 'Official leaderboard comparison'}</small></div>
    <div class="historical-summary-card"><span>Starting budget</span><strong>${formatHistoricalNumber(budget.starting_gold, 1)}g</strong><small>${budget.official ? 'Official' : 'Synthetic scenario'}</small></div>
    <div class="historical-summary-card"><span>Variety bonus</span><strong>+${formatHistoricalNumber(Number(week.variety_bonus) * 100, 0)}%</strong><small>${new Set((week.lineup || []).map(entry => entry.player)).size} roster entries</small></div>
    <div class="historical-summary-card"><span>${week.regret != null ? 'Oracle regret' : 'Champion hits'}</span><strong>${week.regret != null ? formatHistoricalNumber(week.regret) : formatHistoricalNumber(week.champion_top1_hits, 0)}</strong><small>${week.regret != null ? 'Points below best legal lineup' : `+${formatHistoricalNumber(week.realized_champion_bonus)} realized bonus`}</small></div>
  `;

  const budgetCards = [
    ['Starting', budget.starting_gold],
    ['Roster spent', budget.spent_gold],
    ['Unspent', budget.unspent_gold],
    ['Held asset change', budget.held_asset_change],
    ['Ending', budget.ending_gold]
  ].filter(([, value]) => value != null).map(([label, value]) => `
    <div><span>${escapeHtml(label)}</span><strong>${label === 'Held asset change' && Number(value) >= 0 ? '+' : ''}${formatHistoricalNumber(value, 1)}g</strong></div>
  `).join('');

  const rosterCards = (week.lineup || []).map(entry => {
    const slotPrice = entry.gold_price != null
      ? `${formatHistoricalNumber(entry.gold_price, 1)}g`
      : 'Price unavailable';
    const priceChange = entry.gold_change != null
      ? `<p class="historical-price-change">Next: ${formatHistoricalNumber(entry.next_gold_price, 1)}g (${Number(entry.gold_change) >= 0 ? '+' : ''}${formatHistoricalNumber(entry.gold_change, 1)}g)</p>`
      : '';
    let championHtml;
    if (entry.is_coach) {
      championHtml = '<p class="historical-champion unavailable">Coach slot has no champion lock.</p>';
    } else if (!entry.champion_pick) {
      championHtml = '<p class="historical-champion unavailable">Champion lock not preserved for this phase.</p>';
    } else {
      const outcomeClass = entry.champion_hit === true ? 'hit' : (entry.champion_hit === false ? 'miss' : 'unknown');
      const outcome = entry.champion_hit === true ? 'Hit' : (entry.champion_hit === false ? 'Miss' : 'Unscored');
      championHtml = `
        <div class="historical-champion ${outcomeClass}">
          <span>Top-1 champion</span>
          <strong>${escapeHtml(entry.champion_pick)} <small>x${formatHistoricalNumber(entry.multiplier, 1, '-')}</small></strong>
          <em>${escapeHtml(outcome)}</em>
        </div>
        <p class="historical-actual-champions">Played: ${escapeHtml((entry.actual_champions || []).join(', ') || 'Unavailable')}</p>
      `;
    }
    return `
      <article class="card historical-roster-card">
        <div class="optimizer-card-head">
          <span class="optimizer-role">${escapeHtml(String(entry.role).toUpperCase())}</span>
          <span class="historical-slot-price">${slotPrice}</span>
        </div>
        <h3>${escapeHtml(entry.player)}</h3>
        ${priceChange}
        ${championHtml}
      </article>
    `;
  }).join('');

  const comparisons = week.oracle_score != null ? `
    <div><span>Current baseline</span><strong>${formatHistoricalNumber(week.baseline_score)}</strong></div>
    <div><span>Exact legal oracle</span><strong>${formatHistoricalNumber(week.oracle_score)}</strong></div>
    <div><span>Candidate regret</span><strong>${formatHistoricalNumber(week.regret)}</strong></div>
  ` : `
    <div><span>Base score</span><strong>${formatHistoricalNumber(week.base_score)}</strong></div>
    <div><span>Cumulative score</span><strong>${formatHistoricalNumber(week.cumulative_score)}</strong></div>
    <div><span>First place cumulative</span><strong>${formatHistoricalNumber(week.winner_cumulative_points)}</strong></div>
  `;
  content.innerHTML = `
    <div class="card historical-week-heading">
      <div>
        <span class="weekly-eyebrow">${escapeHtml(phase.label)}</span>
        <h3>${escapeHtml(week.round_name)}</h3>
        <p>${escapeHtml(policy.status.replaceAll('_', ' '))}</p>
      </div>
      <div class="historical-comparison-strip">${comparisons}</div>
    </div>
    <div class="card historical-budget-panel">
      <div>
        <span class="weekly-eyebrow">Weekly budget ledger</span>
        <h3>${budget.official ? 'Official account state' : 'Historical account state'}</h3>
        <p>${escapeHtml(budget.source || 'Budget source unavailable')}</p>
      </div>
      <div class="historical-budget-grid">${budgetCards}</div>
    </div>
    <div class="historical-roster-grid">${rosterCards}</div>
  `;
}

function getActivePriceHistory(player, selectedSplit) {
  const history = Array.isArray(player.price_history) ? player.price_history : [];
  return history
    .filter(entry => selectedSplit === 'ALL' || entry.split === selectedSplit)
    .slice()
    .sort((a, b) => {
      const dateCompare = String(a.week_start || a.captured_at_utc || '')
        .localeCompare(String(b.week_start || b.captured_at_utc || ''));
      return dateCompare || ((a.week_num || 0) - (b.week_num || 0));
    });
}

function getActivePriceMetrics(player, selectedSplit) {
  const history = getActivePriceHistory(player, selectedSplit);
  if (history.length === 0) {
    return {
      history,
      start: Number(player.start_price || 15.0),
      current: Number(player.current_price || 15.0),
      latestChange: Number(player.latest_weekly_change || 0.0),
      totalChange: Number(player.total_price_change || 0.0),
      source: player.pricing_source || 'estimated_score_price_mean_reversion'
    };
  }

  const first = history[0];
  const latest = history[history.length - 1];
  const firstPrice = Number(first.price || 0);
  const firstChange = Number(first.change || 0);
  const start = first.previous_price != null
    ? Number(first.previous_price)
    : firstPrice - firstChange;
  const current = Number(latest.price || 0);

  return {
    history,
    start,
    current,
    latestChange: Number(latest.change || 0),
    totalChange: current - start,
    source: latest.source || 'estimated_score_price_mean_reversion'
  };
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
    p._active_price = getActivePriceMetrics(p, split);
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
      valA = a._active_price.current;
      valB = b._active_price.current;
    } else if (currentSortCol === 'total_price_change') {
      valA = a._active_price.totalChange;
      valB = b._active_price.totalChange;
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
  renderChampionLab();
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
    const metrics = p._active_price || getActivePriceMetrics(p, document.getElementById('splitSelect').value);
    const basePrice = metrics.start.toFixed(2);
    const currPrice = metrics.current.toFixed(2);
    const weeklyChg = metrics.latestChange;
    const totalChg = metrics.totalChange;
    const priceSource = metrics.source === 'official_market_api'
      ? '<div style="font-size: 10px; color: #00e676; margin-top: 2px;">OFFICIAL API</div>'
      : '<div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">ESTIMATED</div>';

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
  const metrics = getActivePriceMetrics(player, selectedSplit);
  const historyToUse = metrics.history;

  const detailsEl = document.getElementById('priceModalDetails');
  const totalChg = metrics.totalChange;
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

  const pricingNotice = metrics.source === 'official_market_api'
    ? '<div style="color: #00e676; font-size: 12px; margin-top: 3px;">Official LCS Fantasy market API price</div>'
    : '<div style="color: var(--text-muted); font-size: 12px; margin-top: 3px;">Experimental score + previous-price estimate; official snapshots override it</div>';

  detailsEl.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
      <div>
        <h2 style="font-size: 22px; font-weight: 800;">💰 ${escapeHtml(player.playername)} Market Trajectory</h2>
        <div style="color: var(--text-muted); font-size: 13px;">${escapeHtml(player.teamname)} • ${player.position} • ${player.league} ${player.year} ${selectedSplit !== 'ALL' ? `(${selectedSplit})` : ''}</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 24px; font-weight: 900; color: var(--accent-cyan);">${metrics.current.toFixed(2)} Gold</div>
        ${pricingNotice}
        <div style="font-size: 13px; font-weight: 800; color: ${isUp ? '#00e676' : '#ff1744'};">
          ${isUp ? '+' : ''}${totalChg.toFixed(2)}g (${metrics.start ? ((totalChg / metrics.start)*100).toFixed(1) : '0.0'}%)
        </div>
      </div>
    </div>

    ${swapNotice}

    <div style="background: rgba(10, 14, 23, 0.6); padding: 16px; border-radius: 14px; border: 1px solid var(--border-color); margin-bottom: 20px; height: 300px;">
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
    const points = historyToUse.map(h => h.pts == null ? null : Number(h.pts));
    const hasPoints = points.some(value => Number.isFinite(value));
    const teamColors = historyToUse.map(h => getTeamColor(h.teamname || player.teamname));
    const primaryTeamColor = teamColors[teamColors.length - 1] || getTeamColor(player.teamname);
    const patchMarkers = buildPatchMarkers(historyToUse);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Market Price (Gold)',
            data: prices,
            yAxisID: 'gold',
            borderColor: primaryTeamColor,
            backgroundColor: colorWithAlpha(primaryTeamColor),
            pointBackgroundColor: teamColors,
            pointBorderColor: teamColors,
            segment: {
              borderColor: context => teamColors[context.p1DataIndex] || primaryTeamColor
            },
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8
          },
          {
            label: 'Fantasy Points',
            data: points,
            yAxisID: 'points',
            borderColor: '#f7b955',
            backgroundColor: 'rgba(247, 185, 85, 0.12)',
            pointBackgroundColor: '#f7b955',
            pointBorderColor: '#f7b955',
            borderDash: [6, 4],
            spanGaps: true,
            fill: false,
            tension: 0.25,
            pointRadius: 4,
            pointHoverRadius: 7,
            hidden: !hasPoints
          }
        ]
      },
      plugins: [patchBoundaryPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: hasPoints,
            labels: { color: '#c7d0dc', usePointStyle: true }
          },
          patchBoundaries: { markers: patchMarkers },
          tooltip: {
            callbacks: {
              afterTitle: items => {
                const entry = historyToUse[items[0].dataIndex];
                return entry && entry.patch ? `Patch ${entry.patch}` : '';
              },
              label: (ctx) => ctx.dataset.yAxisID === 'points'
                ? `Fantasy Points: ${Number(ctx.raw).toFixed(2)}`
                : `Price: ${Number(ctx.raw).toFixed(2)} Gold`
            }
          }
        },
        scales: {
          x: { ticks: { color: '#8a99ad' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          gold: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Gold', color: '#8a99ad' },
            ticks: { color: '#8a99ad' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          points: {
            type: 'linear',
            position: 'right',
            display: hasPoints,
            title: { display: true, text: 'Fantasy Points', color: '#f7b955' },
            ticks: { color: '#f7b955' },
            grid: { drawOnChartArea: false }
          }
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
  const selectedYear = document.getElementById('yearSelect').value;

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

  const weekStart = week => {
    for (const player of top5) {
      const stats = player.weekly_stats[week];
      if (stats && stats.week_start) return stats.week_start;
    }
    return '';
  };
  const weeks = Array.from(weekSet).sort((a, b) => {
    const dateCompare = weekStart(a).localeCompare(weekStart(b));
    return dateCompare || a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
  });

  if (weeks.length === 0) return;

  const datasets = top5.map(p => {
    const dataPoints = weeks.map(w => {
      const s = p.weekly_stats[w];
      return s ? (pointsMode === 'adjusted' ? s.adjusted_pts : s.fantasy_pts) : null;
    });
    const teamColors = weeks.map(w => {
      const s = p.weekly_stats[w];
      return getTeamColor(s && s.teamname ? s.teamname : p.teamname);
    });
    const primaryTeamColor = getTeamColor(p.teamname);

    return {
      label: `${p.playername} (${p.teamname})`,
      data: dataPoints,
      borderColor: primaryTeamColor,
      backgroundColor: colorWithAlpha(primaryTeamColor, '20'),
      pointBackgroundColor: teamColors,
      pointBorderColor: teamColors,
      segment: {
        borderColor: context => teamColors[context.p1DataIndex] || primaryTeamColor
      },
      tension: 0.3,
      fill: false,
      pointRadius: 5,
      pointHoverRadius: 8
    };
  });

  const displayLabels = weeks.map(w => selectedSplit !== 'ALL' ? w.replace(selectedSplit, '').trim() : w);
  const patchTimeline = weeks.map(w => {
    for (const player of top5) {
      if (player.weekly_stats[w] && player.weekly_stats[w].patch) return player.weekly_stats[w];
    }
    return {};
  });
  const patchMarkers = selectedYear === 'ALL'
    ? { initialPatch: null, boundaries: [] }
    : buildPatchMarkers(patchTimeline);

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: displayLabels,
      datasets: datasets
    },
    plugins: [patchBoundaryPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#f0f4fc', font: { family: 'Inter', size: 12 } }
        },
        patchBoundaries: { markers: patchMarkers },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            afterTitle: items => {
              const entry = patchTimeline[items[0].dataIndex];
              return entry && entry.patch ? `Patch ${entry.patch}` : '';
            }
          }
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

function formatLabNumber(value, digits = 1) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? '-'
    : Number(value).toFixed(digits);
}

function formatLabPercent(value, digits = 1) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? '-'
    : `${(Number(value) * 100).toFixed(digits)}%`;
}

function renderChampionLab() {
  const selector = document.getElementById('championPlayerSelect');
  const empty = document.getElementById('championLabEmpty');
  const content = document.getElementById('championLabContent');
  if (!selector || !empty || !content) return;

  const profiles = championLabData && Array.isArray(championLabData.profiles)
    ? championLabData.profiles
    : [];
  const search = document.getElementById('searchInput').value.toLowerCase().trim();
  const league = document.getElementById('leagueSelect').value;
  const year = document.getElementById('yearSelect').value;
  const split = document.getElementById('splitSelect').value;

  const matchingProfiles = profiles.filter(profile => {
    const playerMatches = String(profile.player || '').toLowerCase().includes(search);
    const teamMatches = (profile.teams || []).some(team => String(team).toLowerCase().includes(search));
    if (search && !playerMatches && !teamMatches) return false;
    if (league !== 'ALL' && profile.league !== league) return false;
    if (year !== 'ALL' && profile.year !== year) return false;
    if (split !== 'ALL' && profile.split !== split) return false;
    if (currentPositionFilter !== 'ALL' && profile.position !== currentPositionFilter) return false;
    return true;
  });

  const players = Array.from(new Set(matchingProfiles.map(profile => profile.player)))
    .sort((a, b) => a.localeCompare(b));
  const priorSelection = selector.value;
  selector.innerHTML = players.map(player =>
    `<option value="${escapeHtml(player)}">${escapeHtml(player)}</option>`
  ).join('');
  selector.value = players.includes(priorSelection) ? priorSelection : (players[0] || '');

  if (!selector.value) {
    empty.hidden = false;
    content.hidden = true;
    if (championPoolChart) championPoolChart.destroy();
    if (championSplitChart) championSplitChart.destroy();
    championPoolChart = null;
    championSplitChart = null;
    return;
  }

  empty.hidden = true;
  content.hidden = false;
  const selectedProfiles = matchingProfiles
    .filter(profile => profile.player === selector.value)
    .sort((a, b) => String(b.end_date || '').localeCompare(String(a.end_date || '')));
  const profile = selectedProfiles[0];
  const history = profiles
    .filter(item =>
      item.player === selector.value &&
      (league === 'ALL' || item.league === league)
    )
    .sort((a, b) => String(a.start_date || '').localeCompare(String(b.start_date || '')));

  renderChampionSummary(profile);
  renderChampionPoolChart(profile);
  renderChampionSplitChart(history);
  renderChampionSegmentTable(profile);
  renderChampionPickTable(profile);
  renderChampionBanTable(profile);
}

function renderChampionSummary(profile) {
  const summary = profile.summary;
  const teamText = (profile.teams || []).join(', ') || 'Unknown team';
  document.getElementById('championProfileLabel').textContent =
    `${profile.player} | ${teamText} | ${profile.league} ${profile.year} ${profile.split} | ${summary.games} games`;

  const cards = [
    ['Pool shape', summary.pool_shape, `${summary.unique_champions} unique champions`],
    ['Top-3 concentration', formatLabPercent(summary.top_three_concentration), 'Share of games on three most-picked champions'],
    ['Win rate', formatLabPercent(summary.win_rate), `${summary.wins} wins in ${summary.games} games`],
    ['Fantasy points', formatLabNumber(summary.avg_fantasy_points, 2), 'Average per game'],
    ['Damage / minute', formatLabNumber(summary.avg_dpm, 0), 'Observed average'],
    ['Gold diff @15', formatLabNumber(summary.avg_gold_diff_15, 0), 'Observed lane-state average']
  ];
  document.getElementById('championSummaryGrid').innerHTML = cards.map(([label, value, note]) => `
    <div class="champion-summary-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>
  `).join('');
}

function renderChampionPoolChart(profile) {
  const canvas = document.getElementById('championPoolChart');
  if (championPoolChart) championPoolChart.destroy();
  const picks = (profile.champion_picks || []).slice(0, 10);
  const teamColor = getTeamColor((profile.teams || [])[0]);
  championPoolChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: picks.map(item => item.champion),
      datasets: [{
        label: 'Pick share',
        data: picks.map(item => item.pick_share * 100),
        backgroundColor: colorWithAlpha(teamColor, '99'),
        borderColor: teamColor,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#a8b2c7' }, grid: { display: false } },
        y: {
          beginAtZero: true,
          ticks: { color: '#a8b2c7', callback: value => `${value}%` },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      }
    }
  });
}

function renderChampionSplitChart(history) {
  const canvas = document.getElementById('championSplitChart');
  if (championSplitChart) championSplitChart.destroy();
  const labels = history.map(profile => `${profile.year} ${profile.split}`);
  championSplitChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Unique champions',
          data: history.map(profile => profile.summary.unique_champions),
          borderColor: '#00f2fe',
          backgroundColor: 'rgba(0,242,254,0.15)',
          yAxisID: 'y',
          tension: 0.25
        },
        {
          label: 'Top-3 concentration',
          data: history.map(profile => profile.summary.top_three_concentration * 100),
          borderColor: '#ffb703',
          backgroundColor: 'rgba(255,183,3,0.12)',
          yAxisID: 'y1',
          tension: 0.25
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#dce3f1' } } },
      scales: {
        x: { ticks: { color: '#a8b2c7' }, grid: { display: false } },
        y: {
          beginAtZero: true,
          position: 'left',
          ticks: { color: '#00f2fe', precision: 0 },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        y1: {
          beginAtZero: true,
          max: 100,
          position: 'right',
          ticks: { color: '#ffb703', callback: value => `${value}%` },
          grid: { drawOnChartArea: false }
        }
      }
    }
  });
}

function renderChampionPickTable(profile) {
  const rows = (profile.champion_picks || []).map(item => `
    <tr>
      <td><strong>${escapeHtml(item.champion)}</strong></td>
      <td>${item.games}</td>
      <td>${formatLabPercent(item.pick_share)}</td>
      <td>${formatLabPercent(item.win_rate)}</td>
      <td>${formatLabNumber(item.avg_fantasy_points, 2)}</td>
      <td>${formatLabNumber(item.avg_kills)} / ${formatLabNumber(item.avg_deaths)} / ${formatLabNumber(item.avg_assists)}</td>
      <td>${formatLabNumber(item.avg_dpm, 0)}</td>
      <td>${formatLabPercent(item.avg_damage_share)}</td>
      <td>${formatLabNumber(item.avg_gold_diff_15, 0)}</td>
      <td>${escapeHtml((item.patches || []).join(', '))}</td>
    </tr>
  `).join('');
  document.getElementById('championPickTableContainer').innerHTML = `
    <div class="table-wrapper">
      <table>
        <thead><tr>
          <th>Champion</th><th>Games</th><th>Pick share</th><th>Win rate</th>
          <th>Fantasy pts</th><th>K / D / A</th><th>DPM</th><th>Damage share</th>
          <th>GD @15</th><th>Patches</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderChampionSegmentTable(profile) {
  const segments = profile.split_segments || [];
  const rows = segments.map(segment => `
    <tr>
      <td><strong>${escapeHtml(segment.label)}</strong></td>
      <td>${segment.games}</td>
      <td>${segment.unique_champions}</td>
      <td>${formatLabPercent(segment.top_three_concentration)}</td>
      <td>${segment.top_picks.map(pick =>
        `${escapeHtml(pick.champion)} (${pick.games}, ${formatLabPercent(pick.pick_share)})`
      ).join(', ')}</td>
    </tr>
  `).join('');
  document.getElementById('championSegmentTableContainer').innerHTML = `
    <div class="table-wrapper">
      <table>
        <thead><tr>
          <th>Period</th><th>Games</th><th>Unique champions</th>
          <th>Top-3 concentration</th><th>Most-picked champions</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderChampionBanTable(profile) {
  const bans = (profile.opponent_bans || []).slice(0, 25);
  const rows = bans.map(item => {
    const liftClass = item.targeted_ban_lift > 0 ? 'positive-lift' : 'negative-lift';
    const prefix = item.targeted_ban_lift > 0 ? '+' : '';
    return `
      <tr>
        <td><strong>${escapeHtml(item.champion)}</strong></td>
        <td>${item.ban_games}</td>
        <td>${formatLabPercent(item.faced_ban_rate)}</td>
        <td>${formatLabPercent(item.global_side_ban_rate)}</td>
        <td class="${liftClass}">${prefix}${formatLabPercent(item.targeted_ban_lift)}</td>
      </tr>
    `;
  }).join('');
  document.getElementById('championBanTableContainer').innerHTML = `
    <div class="table-wrapper">
      <table>
        <thead><tr>
          <th>Champion</th><th>Games banned</th><th>Faced-ban rate</th>
          <th>Normal split rate</th><th>Ban lift</th>
        </tr></thead>
        <tbody>${rows || '<tr><td colspan="5">No recorded opponent bans.</td></tr>'}</tbody>
      </table>
    </div>
  `;
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

function setupEvalTabs() {
  const btns = document.querySelectorAll('.eval-subtab-btn');
  if (btns.length === 0) return;
  btns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.eval-subtab-btn').forEach(b => {
        b.classList.remove('active');
        b.style.background = 'none';
        b.style.color = 'var(--text-muted)';
        b.style.border = '1px solid transparent';
      });
      document.querySelectorAll('.eval-subview-section').forEach(s => s.style.display = 'none');

      const targetBtn = e.currentTarget;
      const targetSubviewId = targetBtn.dataset.subview;
      targetBtn.classList.add('active');
      targetBtn.style.background = 'rgba(76, 201, 240, 0.15)';
      targetBtn.style.color = 'var(--primary-color)';
      targetBtn.style.border = '1px solid rgba(76, 201, 240, 0.3)';

      const subviewEl = document.getElementById(targetSubviewId);
      if (subviewEl) subviewEl.style.display = 'block';

      if (targetSubviewId === 'eval-development') {
        renderEvalDevelopment();
      } else if (targetSubviewId === 'eval-reconstructed') {
        renderEvalReconstructed();
      } else if (targetSubviewId === 'eval-diagnostics') {
        renderM3Diagnostics();
      } else if (targetSubviewId === 'eval-archived') {
        renderHistoricalLineups();
      }
    });
  });

  // Apply default styles to the active button
  const activeBtn = document.querySelector('.eval-subtab-btn.active');
  if (activeBtn) {
    activeBtn.style.background = 'rgba(76, 201, 240, 0.15)';
    activeBtn.style.color = 'var(--primary-color)';
    activeBtn.style.border = '1px solid rgba(76, 201, 240, 0.3)';
  }
}

function renderModelEvaluation() {
  const activeBtn = document.querySelector('.eval-subtab-btn.active');
  const activeSubviewId = activeBtn ? activeBtn.dataset.subview : 'eval-development';

  if (activeSubviewId === 'eval-development') {
    renderEvalDevelopment();
  } else if (activeSubviewId === 'eval-reconstructed') {
    renderEvalReconstructed();
  } else if (activeSubviewId === 'eval-diagnostics') {
    renderM3Diagnostics();
  } else if (activeSubviewId === 'eval-archived') {
    renderHistoricalLineups();
  }
}

function renderEvalDevelopment() {
  if (!evalDevSummary) return;
  const dev = evalDevSummary;
  const modelInfo = dev.final_model;

  // 1. Populate Selected Model Identity
  document.getElementById('developmentFinalModel').innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 10px; color: #f0f4fc;">
      <div><strong>Model ID:</strong> <span class="tier-chip tier-13" style="font-size: 11px;">${escapeHtml(modelInfo.candidate_id)}</span></div>
      <div><strong>Architecture:</strong> ${escapeHtml(modelInfo.architecture)}</div>
      <div><strong>Description:</strong> ${escapeHtml(modelInfo.description)}</div>
      <div><strong>Regularization (Alpha):</strong> <code>${modelInfo.alpha.toFixed(1)}</code></div>
      <div><strong>Included Feature Blocks:</strong> ${escapeHtml(modelInfo.included_blocks.join(', '))}</div>
      <div><strong>Excluded Feature Blocks:</strong> ${escapeHtml(modelInfo.excluded_blocks.join(', '))}</div>
      <div><strong>Registered Interactions:</strong> ${modelInfo.included_registered_interactions.length > 0 ? escapeHtml(modelInfo.included_registered_interactions.join(', ')) : '<em>none retained</em>'}</div>
      <div><strong>Estimator:</strong> ${escapeHtml(modelInfo.estimator)}</div>
      <div><strong>Solver:</strong> <code>${escapeHtml(modelInfo.solver)}</code></div>
      <div style="margin-top: 5px; padding-top: 10px; border-top: 1px solid var(--border-color); font-weight: 700; color: var(--primary-color); font-size: 15px;">
        Development MAE: ${formatHistoricalNumber(modelInfo.development_mae, 6)}
      </div>
      <div style="font-size: 11px; color: var(--text-muted); font-family: monospace; word-break: break-all; margin-top: 5px;">
        Policy SHA-256: ${escapeHtml(modelInfo.policy_hash)}
      </div>
    </div>
  `;

  // 2. Populate Feature Family Conclusions
  const conclusionsHtml = dev.feature_family_conclusions.map(c => `
    <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); background: rgba(255, 255, 255, 0.01);">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <strong style="color: #fff; font-size: 13px;">Block ${escapeHtml(c.block)}: ${escapeHtml(c.name)}</strong>
        <span class="tier-chip ${c.conclusion === 'retained' ? 'tier-13' : 'tier-15'}" style="font-size: 10px; padding: 2px 6px;">${escapeHtml(c.conclusion)}</span>
      </div>
      <p style="margin: 0; font-size: 12px; color: var(--text-muted);">${escapeHtml(c.evidence)}</p>
      <small style="display: block; margin-top: 4px; font-weight: 700; color: ${c.conclusion === 'retained' ? 'var(--primary-color)' : '#ff1744'};">
        Result: ${escapeHtml(c.language)}
      </small>
    </div>
  `).join('');
  document.getElementById('developmentFeatureConclusions').innerHTML = conclusionsHtml;

  // 3. Populate Model Progression Table
  const progTableHtml = `
    <table class="leaderboard-table" style="width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 10px;">
      <thead>
        <tr style="border-bottom: 1px solid var(--border-color);">
          <th style="text-align: left; padding: 6px;">Model</th>
          <th style="text-align: left; padding: 6px;">Alpha</th>
          <th style="text-align: right; padding: 6px;">MAE</th>
          <th style="text-align: right; padding: 6px;">RMSE</th>
        </tr>
      </thead>
      <tbody>
        ${dev.model_progression.map(m => `
          <tr class="${m.model_id === 'OBC' ? 'swapped-row' : ''}" style="border-bottom: 1px solid rgba(255,255,255,0.03);">
            <td style="padding: 6px;"><strong>${escapeHtml(m.model_id)}</strong></td>
            <td style="padding: 6px;">${m.alpha !== null ? m.alpha.toFixed(1) : 'N/A'}</td>
            <td style="text-align: right; padding: 6px; font-weight: 700; color: ${m.model_id === 'OBC' ? 'var(--primary-color)' : '#fff'};">${formatHistoricalNumber(m.mae, 5)}</td>
            <td style="text-align: right; padding: 6px; color: var(--text-muted);">${formatHistoricalNumber(m.rmse, 5)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
  document.getElementById('developmentProgressionTable').innerHTML = progTableHtml;

  // 4. Render Progression Chart
  if (devProgressionChart) devProgressionChart.destroy();
  const canvas = document.getElementById('developmentProgressionChart');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    const chartLabels = dev.model_progression.map(m => m.model_id);
    const chartData = dev.model_progression.map(m => m.mae);

    devProgressionChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: chartLabels,
        datasets: [{
          label: 'MAE',
          data: chartData,
          backgroundColor: chartLabels.map(label => label === 'OBC' ? 'rgba(76, 201, 240, 0.6)' : 'rgba(76, 201, 240, 0.15)'),
          borderColor: chartLabels.map(label => label === 'OBC' ? '#4cc9f0' : 'rgba(76, 201, 240, 0.4)'),
          borderWidth: 1,
          barThickness: 24
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { ticks: { color: '#8a99ad' }, grid: { display: false } },
          y: {
            min: 5.0,
            max: 5.2,
            ticks: { color: '#8a99ad', stepSize: 0.05 },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });
  }

  // 5. Populate Interaction Candidates Table
  const interactionTableHtml = `
    <table class="leaderboard-table" style="width: 100%; font-size: 12px; border-collapse: collapse;">
      <thead>
        <tr style="border-bottom: 1px solid var(--border-color);">
          <th style="text-align: left; padding: 6px;">Candidate</th>
          <th style="text-align: left; padding: 6px;">Description</th>
          <th style="text-align: right; padding: 6px;">MAE</th>
          <th style="text-align: right; padding: 6px;">Delta vs G0</th>
        </tr>
      </thead>
      <tbody>
        ${dev.stage6g_interaction_results.map(r => `
          <tr class="${r.candidate_id === 'G0' ? 'swapped-row' : ''}" style="border-bottom: 1px solid rgba(255,255,255,0.03);">
            <td style="padding: 6px;"><strong>${escapeHtml(r.candidate_id)}</strong></td>
            <td style="padding: 6px; color: var(--text-muted);">${escapeHtml(r.description)}</td>
            <td style="text-align: right; padding: 6px; font-weight: 700; color: ${r.candidate_id === 'G0' ? 'var(--primary-color)' : '#fff'};">${formatHistoricalNumber(r.mae, 5)}</td>
            <td style="text-align: right; padding: 6px; color: ${r.delta_vs_g0 > 0 ? '#ff1744' : '#4a5b6c'};">${r.delta_vs_g0 > 0 ? '+' : ''}${formatHistoricalNumber(r.delta_vs_g0, 6)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
    <div style="margin-top: 12px; font-size: 11.5px; color: var(--text-muted); font-style: italic; line-height: 1.4;">
      * Delta vs G0 represents the difference in MAE. No interaction term strictly improved upon the baseline OBC model (G0).
    </div>
  `;
  document.getElementById('developmentInteractionTable').innerHTML = interactionTableHtml;
}

function renderEvalReconstructed() {
  if (!evalWeeklyResults || !evalLeaderboard || !evalProvenance) return;
  const weekly = evalWeeklyResults;
  const lb = evalLeaderboard;

  // 1. Populate Determinism Badge
  const badgeEl = document.getElementById('determinismBadge');
  if (badgeEl) {
    if (weekly.determinism_passed) {
      badgeEl.className = 'price-badge neutral';
      badgeEl.style.background = 'rgba(0, 230, 118, 0.15)';
      badgeEl.style.color = '#00e676';
      badgeEl.style.border = '1px solid rgba(0, 230, 118, 0.3)';
      badgeEl.innerHTML = `✓ Determinism check passed (${weekly.determinism_runs} runs)`;
    } else {
      badgeEl.className = 'price-badge down';
      badgeEl.innerHTML = `⚠️ Determinism check failed`;
    }
  }

  // 2. Populate Headline Summary Grid
  document.getElementById('reconstructedHeadlineGrid').innerHTML = `
    <div class="historical-summary-card">
      <span>Final score</span>
      <strong>${formatHistoricalNumber(weekly.cumulative_score, 2)}</strong>
      <small>Realized score over 11 periods</small>
    </div>
    <div class="historical-summary-card">
      <span>Leaderboard winner</span>
      <strong>${formatHistoricalNumber(lb.winner_score, 2)}</strong>
      <small>Cumulative points</small>
    </div>
    <div class="historical-summary-card" style="border: 1px solid ${lb.gap_to_winner <= 0 ? 'rgba(255,23,68,0.2)' : 'rgba(0,230,118,0.2)'};">
      <span>Gap to winner</span>
      <strong style="color: ${lb.gap_to_winner <= 0 ? '#ff1744' : '#00e676'};">${lb.gap_to_winner >= 0 ? '+' : ''}${formatHistoricalNumber(lb.gap_to_winner, 2)}</strong>
      <small>Points behind winner</small>
    </div>
    <div class="historical-summary-card">
      <span>Rayz actual score</span>
      <strong>${formatHistoricalNumber(lb.rayz_score, 2)}</strong>
      <small>User's historical entry</small>
    </div>
    <div class="historical-summary-card" style="border: 1px solid rgba(0,230,118,0.2);">
      <span>Model vs Rayz</span>
      <strong style="color: #00e676;">+${formatHistoricalNumber(lb.gap_to_rayz, 2)}</strong>
      <small>Improvement over user</small>
    </div>
    <div class="historical-summary-card">
      <span>Rounds simulated</span>
      <strong>${weekly.period_count}</strong>
      <small>11 consecutive periods</small>
    </div>
  `;

  // 3. Populate Weekly Selection List
  const listHtml = weekly.weeks.map(w => {
    const isActive = w.week === selectedEvalWeekNum;
    const activeStyle = isActive ? 'background: rgba(76, 201, 240, 0.15); border: 1px solid rgba(76, 201, 240, 0.4); color: #fff;' : 'background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); color: var(--text-muted);';
    return `
      <div class="card reconstructed-week-row" data-week="${w.week}" style="padding: 10px; cursor: pointer; border-radius: 8px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; ${activeStyle}">
        <div>
          <strong>Week ${w.week}: ${escapeHtml(w.stage_round)}</strong>
        </div>
        <div style="font-weight: 700; color: ${isActive ? 'var(--primary-color)' : '#fff'};">
          ${formatHistoricalNumber(w.realized_score, 2)} pts
        </div>
      </div>
    `;
  }).join('');
  document.getElementById('reconstructedWeeksList').innerHTML = listHtml;

  // Add click listeners to rows
  document.querySelectorAll('.reconstructed-week-row').forEach(row => {
    row.addEventListener('click', (e) => {
      selectedEvalWeekNum = parseInt(e.currentTarget.dataset.week);
      renderEvalReconstructed();
    });
  });

  // 4. Render Selected Week Detail
  renderReconstructedWeekDetail(selectedEvalWeekNum);

  // 5. Populate Leaderboard Comparison Card
  document.getElementById('reconstructedLeaderboardCompare').innerHTML = `
    <div style="font-size: 13.5px; line-height: 1.6; color: #f0f4fc;">
      <div style="margin-bottom: 10px;"><strong>Competition:</strong> ${escapeHtml(lb.competition_label)}</div>
      <div style="margin-bottom: 10px;"><strong>Leaderboard Source:</strong> ${escapeHtml(lb.leaderboard_source)}</div>
      <div style="margin-bottom: 10px;"><strong>Screenshots Hash:</strong> <code style="font-size: 11.5px;">${escapeHtml(lb.leaderboard_screenshots_sha256)}</code></div>
      <div style="margin-bottom: 10px;"><strong>Status:</strong> <span class="tier-chip tier-15" style="font-size: 11px;">${escapeHtml(lb.leaderboard_status)}</span></div>
      <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); background: rgba(76, 201, 240, 0.03);">
        <strong style="display: block; color: var(--primary-color); margin-bottom: 4px;">Rank & Percentile Claims:</strong>
        <p style="margin: 0; font-size: 12.5px; line-height: 1.5;">${escapeHtml(lb.rank_claim_verbose)}</p>
      </div>
      <div>
        <strong>Surviving entries:</strong> 2 (winner & user Rayz)<br>
        <span style="font-size: 11px; color: var(--text-muted);">* Exact rank bound: ${escapeHtml(lb.rank_bound)}</span>
      </div>
    </div>
  `;
}

function renderReconstructedWeekDetail(weekNum) {
  if (!evalWeeklyResults || !evalProvenance) return;
  const week = evalWeeklyResults.weeks.find(w => w.week === weekNum);
  if (!week) return;

  const prov = evalProvenance;

  // Roster rendering
  const rosterHtml = week.roster.map(r => {
    const isCoach = r.is_coach;
    const priceText = r.price !== null ? `${r.price.toFixed(1)}g` : 'N/A';

    let championText = '';
    if (isCoach) {
      championText = '<span style="color: var(--text-muted);">N/A (Coach)</span>';
    } else if (r.predicted_champion) {
      const oc = r.champion_outcome;
      if (oc) {
        const outcomeClass = oc.hit ? 'hit' : 'miss';
        const multLabel = oc.multiplier === 1.3 ? 'Comfort' : (oc.multiplier === 1.5 ? 'Adoption' : 'Novelty');
        championText = `
          <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.02); padding-top: 6px;">
            <div class="historical-champion ${outcomeClass}" style="margin: 0;">
              <span>Predicted Champion</span>
              <strong>${escapeHtml(r.predicted_champion)} <small>x${oc.multiplier.toFixed(1)} (${multLabel})</small></strong>
              <em>${oc.hit ? 'Hit' : 'Miss'}</em>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
              Played: ${escapeHtml(oc.actual_champions.join(', ') || 'none')}
            </div>
            <div style="font-size: 11.5px; color: ${oc.hit ? '#00e676' : 'var(--text-muted)'}; font-weight: 700; margin-top: 2px;">
              Champion bonus score: +${formatHistoricalNumber(oc.realized_bonus, 2)}
            </div>
          </div>
        `;
      } else {
        championText = `
          <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.02); padding-top: 6px;">
            <strong style="color: var(--text-muted); font-size: 12px;">Predicted champion: ${escapeHtml(r.predicted_champion)}</strong>
            <div style="font-size: 11px; color: var(--text-muted);">Outcomes not available</div>
          </div>
        `;
      }
    } else {
      championText = '<div style="font-size: 11.5px; color: var(--text-muted); margin-top: 4px; font-style: italic;">No champion recommendation preserved</div>';
    }

    return `
      <article class="card historical-roster-card" style="padding: 12px; margin-bottom: 0;">
        <div class="optimizer-card-head" style="margin-bottom: 5px;">
          <span class="optimizer-role">${escapeHtml(isCoach ? 'COACH' : 'ROLE PLAYER')}</span>
          <span class="historical-slot-price" style="font-size: 12px; font-weight: 700; color: var(--primary-color);">${priceText}</span>
        </div>
        <h4 style="margin: 0 0 5px; font-size: 15px; color: #fff;">${escapeHtml(isCoach ? r.player.replace('coach::', '') : r.player)}</h4>
        ${championText}
      </article>
    `;
  }).join('');

  // Weekly budget ledger
  const budgetCards = [
    ['Starting budget', week.starting_budget, 'g'],
    ['Roster cost', week.roster_cost, 'g'],
    ['Unspent gold', week.unused_gold, 'g'],
    ['Held asset change', week.held_asset_change, 'g', true],
    ['Next budget', week.next_budget, 'g']
  ].filter(([, val]) => val !== null).map(([label, val, unit, showSign]) => {
    const sign = showSign && val >= 0 ? '+' : '';
    return `
      <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border-color); text-align: center;">
        <span style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; display: block; margin-bottom: 4px;">${escapeHtml(label)}</span>
        <strong style="font-size: 14px; color: #fff;">${sign}${val.toFixed(1)}${unit}</strong>
      </div>
    `;
  }).join('');

  // Score comparison strip
  const scoreStrip = `
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;">
      <div><span style="font-size: 11px; color: var(--text-muted);">Roster raw points:</span> <strong style="color: #fff;">${formatHistoricalNumber(week.roster_raw_points, 2)}</strong></div>
      <div><span style="font-size: 11px; color: var(--text-muted);">Champion bonus:</span> <strong style="color: var(--primary-color);">+${formatHistoricalNumber(week.realized_champion_bonus, 2)}</strong></div>
      <div><span style="font-size: 11px; color: var(--text-muted);">Variety bonus:</span> <strong style="color: var(--primary-color);">+${(week.variety_bonus * 100).toFixed(0)}%</strong></div>
      <div style="padding-left: 15px; border-left: 1px solid var(--border-color);"><span style="font-size: 11px; color: var(--text-muted); font-weight: 700;">Realized score:</span> <strong style="color: #00e676; font-size: 14px;">${formatHistoricalNumber(week.realized_score, 2)}</strong></div>
      <div><span style="font-size: 11px; color: var(--text-muted);">Cumulative score:</span> <strong style="color: #fff; font-size: 14px;">${formatHistoricalNumber(week.cumulative_score, 2)}</strong></div>
    </div>
  `;

  // Weekly integrity/provenance block
  const provenanceHtml = `
    <div class="card" style="margin-top: 15px; padding: 12px; border: 1px solid rgba(76,201,240,0.15); background: rgba(76,201,240,0.01);">
      <h4 style="margin: 0 0 8px; font-size: 12px; color: var(--primary-color); text-transform: uppercase; letter-spacing: 0.5px;">✓ Chronological point-in-time safety verified</h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; font-size: 11.5px; color: var(--text-muted);">
        <div><strong>Pre-lock Cutoff:</strong> <code>${escapeHtml(week.prelock_cutoff)}</code></div>
        <div><strong>Sealed Lineup SHA-256:</strong> <code style="word-break: break-all;">${escapeHtml(week.sealed_lineup_sha256)}</code></div>
        <div><strong>Player Model:</strong> <code>${escapeHtml(prov.player_model.candidate_id)}</code> (Alpha=${prov.player_model.alpha})</div>
        <div><strong>Champion Predictor:</strong> <code>${escapeHtml(prov.champion_predictor.id)}</code></div>
        <div><strong>Pricing Policy Hash:</strong> <code style="word-break: break-all;">${escapeHtml(prov.pricing_policy)}</code></div>
        <div><strong>Budget Policy Hash:</strong> <code style="word-break: break-all;">${escapeHtml(prov.budget_policy)}</code></div>
        <div><strong>Scoring Configuration:</strong> <code style="word-break: break-all;">${escapeHtml(prov.scoring_config)}</code></div>
      </div>
    </div>
  `;

  document.getElementById('reconstructedWeekDetail').innerHTML = `
    <div class="card historical-week-heading" style="padding: 15px; margin-bottom: 12px;">
      <div>
        <span class="weekly-eyebrow">Period detail view</span>
        <h3 style="margin: 4px 0 0;">Week ${week.week}: ${escapeHtml(week.stage_round)}</h3>
        <p style="margin: 4px 0 0; font-size: 11.5px; color: var(--text-muted);">Target patch: <code>${escapeHtml(week.patch)}</code> | Period ID: <code>${escapeHtml(week.prediction_period_id)}</code></p>
      </div>
      <div style="text-align: right;">
        ${scoreStrip}
      </div>
    </div>

    <!-- Budget ledgers -->
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 12px;">
      ${budgetCards}
    </div>

    <!-- Roster Grid -->
    <div class="historical-roster-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
      ${rosterHtml}
    </div>

    <!-- Provenance binding block -->
    ${provenanceHtml}
  `;
}

function renderM3Diagnostics() {
  if (!evalM3Diagnostics || evalM3Diagnostics.length === 0) {
    document.getElementById('m3DiagnosticsTableBody').innerHTML = `
      <tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No diagnostic data loaded.</td></tr>
    `;
    return;
  }

  // 1. Populate Dropdowns once if empty
  const weekSelect = document.getElementById('m3FilterWeek');
  if (weekSelect && weekSelect.children.length === 0) {
    const weeks = [...new Set(evalM3Diagnostics.map(r => r.week_id))].sort((a, b) => {
      const numA = parseInt(a.replace(/\D/g, '')) || 0;
      const numB = parseInt(b.replace(/\D/g, '')) || 0;
      return numA - numB;
    });
    weekSelect.innerHTML = '<option value="ALL">All Weeks</option>' +
      weeks.map(w => `<option value="${escapeHtml(w)}">${escapeHtml(w)}</option>`).join('');
    weekSelect.value = m3DiagFilters.week;
    weekSelect.addEventListener('change', (e) => {
      m3DiagFilters.week = e.target.value;
      m3DiagCurrentPage = 1;
      renderM3Diagnostics();
    });
  }

  const teamSelect = document.getElementById('m3FilterTeam');
  if (teamSelect && teamSelect.children.length === 0) {
    const teams = [...new Set(evalM3Diagnostics.map(r => r.player_team_at_period))].sort();
    teamSelect.innerHTML = '<option value="ALL">All Teams</option>' +
      teams.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    teamSelect.value = m3DiagFilters.team;
    teamSelect.addEventListener('change', (e) => {
      m3DiagFilters.team = e.target.value;
      m3DiagCurrentPage = 1;
      renderM3Diagnostics();
    });
  }

  const searchInput = document.getElementById('m3FilterSearch');
  if (searchInput && !searchInput.dataset.wired) {
    searchInput.dataset.wired = 'true';
    searchInput.value = m3DiagFilters.search;
    searchInput.addEventListener('input', (e) => {
      m3DiagFilters.search = e.target.value.trim();
      m3DiagCurrentPage = 1;
      renderM3Diagnostics();
    });
  }

  const roleSelect = document.getElementById('m3FilterRole');
  if (roleSelect && !roleSelect.dataset.wired) {
    roleSelect.dataset.wired = 'true';
    roleSelect.value = m3DiagFilters.role;
    roleSelect.addEventListener('change', (e) => {
      m3DiagFilters.role = e.target.value;
      m3DiagCurrentPage = 1;
      renderM3Diagnostics();
    });
  }

  const resetBtn = document.getElementById('m3ResetFiltersBtn');
  if (resetBtn && !resetBtn.dataset.wired) {
    resetBtn.dataset.wired = 'true';
    resetBtn.addEventListener('click', () => {
      m3DiagFilters = { week: 'ALL', role: 'ALL', team: 'ALL', search: '' };
      if (searchInput) searchInput.value = '';
      if (weekSelect) weekSelect.value = 'ALL';
      if (roleSelect) roleSelect.value = 'ALL';
      if (teamSelect) teamSelect.value = 'ALL';
      m3DiagCurrentPage = 1;
      renderM3Diagnostics();
    });
  }

  // 2. Wire Group Tabs
  const tabWeek = document.getElementById('m3GroupTabWeek');
  const tabTeam = document.getElementById('m3GroupTabTeam');
  const tabFallback = document.getElementById('m3GroupTabFallback');

  const setupGroupTab = (btn, mode) => {
    if (btn && !btn.dataset.wired) {
      btn.dataset.wired = 'true';
      btn.addEventListener('click', () => {
        [tabWeek, tabTeam, tabFallback].forEach(b => {
          if (b) {
            b.classList.remove('active');
            b.style.background = 'none';
            b.style.color = 'var(--text-muted)';
            b.style.border = '1px solid transparent';
          }
        });
        btn.classList.add('active');
        btn.style.background = 'rgba(76, 201, 240, 0.15)';
        btn.style.color = 'var(--primary-color)';
        btn.style.border = '1px solid rgba(76, 201, 240, 0.3)';
        m3GroupTabActive = mode;
        updateM3GroupDiagnosticsTable();
      });
    }
  };
  setupGroupTab(tabWeek, 'week');
  setupGroupTab(tabTeam, 'team');
  setupGroupTab(tabFallback, 'fallback');

  // 3. Filter and Sort
  let filtered = evalM3Diagnostics.filter(r => {
    if (m3DiagFilters.week !== 'ALL' && r.week_id !== m3DiagFilters.week) return false;
    if (m3DiagFilters.role !== 'ALL' && r.role.toUpperCase() !== m3DiagFilters.role) return false;
    if (m3DiagFilters.team !== 'ALL' && r.player_team_at_period !== m3DiagFilters.team) return false;
    if (m3DiagFilters.search) {
      const s = m3DiagFilters.search.toLowerCase();
      if (!r.player_name.toLowerCase().includes(s)) return false;
    }
    return true;
  });

  const sortCol = m3DiagSort.col;
  const sortDir = m3DiagSort.dir === 'asc' ? 1 : -1;
  filtered.sort((a, b) => {
    let valA = a[sortCol];
    let valB = b[sortCol];
    if (typeof valA === 'string') {
      return valA.localeCompare(valB) * sortDir;
    }
    if (valA === null || valA === undefined) return 1;
    if (valB === null || valB === undefined) return -1;
    return (valA - valB) * sortDir;
  });

  // 4. Global performance summary
  const sumAbsError = filtered.reduce((acc, r) => acc + r.absolute_error, 0);
  const sumSignedError = filtered.reduce((acc, r) => acc + r.signed_error, 0);
  const globalMae = filtered.length > 0 ? (sumAbsError / filtered.length) : 0;
  const globalMse = filtered.length > 0 ? (sumSignedError / filtered.length) : 0;

  let exactCount = 0;
  let overCount = 0;
  let underCount = 0;
  filtered.forEach(r => {
    if (r.absolute_error < 0.5) exactCount++;
    else if (r.signed_error < 0) overCount++;
    else underCount++;
  });

  document.getElementById('m3DiagnosticsGlobalStats').innerHTML = `
    <div class="historical-summary-card" style="padding: 10px;">
      <span style="font-size: 11px;">Exposed MAE</span>
      <strong style="font-size: 18px; color: #fff;">${globalMae.toFixed(3)}</strong>
      <small style="font-size: 10px;">on ${filtered.length} rows</small>
    </div>
    <div class="historical-summary-card" style="padding: 10px;">
      <span style="font-size: 11px;">Bias (Mean Err)</span>
      <strong style="font-size: 18px; color: ${globalMse >= 0 ? '#00e676' : '#ff1744'};">${globalMse >= 0 ? '+' : ''}${globalMse.toFixed(3)}</strong>
      <small style="font-size: 10px;">${globalMse >= 0 ? 'underpredicted' : 'overpredicted'}</small>
    </div>
    <div class="historical-summary-card" style="padding: 10px;">
      <span style="font-size: 11px;">Accuracy Split</span>
      <div style="font-size: 11px; margin-top: 5px; color: var(--text-muted); font-weight: 700;">
        <span style="color:#00e676;">${underCount} U</span> |
        <span style="color:#fff;">${exactCount} E</span> |
        <span style="color:#ff1744;">${overCount} O</span>
      </div>
      <small style="font-size: 9px; display:block; margin-top:3px;">U=Under, E=Exact, O=Over</small>
    </div>
  `;

  // Role Breakdown Table
  const rolesGrouped = {};
  filtered.forEach(r => {
    rolesGrouped[r.role] = rolesGrouped[r.role] || { abs: 0, sign: 0, count: 0 };
    rolesGrouped[r.role].abs += r.absolute_error;
    rolesGrouped[r.role].sign += r.signed_error;
    rolesGrouped[r.role].count++;
  });

  const roleStatsHtml = `
    <table style="width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 5px;">
      <thead>
        <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-muted);">
          <th style="text-align: left; padding: 4px;">Role</th>
          <th style="text-align: right; padding: 4px;">n</th>
          <th style="text-align: right; padding: 4px;">MAE</th>
          <th style="text-align: right; padding: 4px;">Bias (Mean Signed)</th>
        </tr>
      </thead>
      <tbody>
        ${Object.keys(rolesGrouped).sort().map(rl => {
          const g = rolesGrouped[rl];
          const mae = g.abs / g.count;
          const bias = g.sign / g.count;
          return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
              <td style="padding: 4px; text-transform: uppercase;"><strong>${escapeHtml(rl)}</strong></td>
              <td style="text-align: right; padding: 4px; color: var(--text-muted);">${g.count}</td>
              <td style="text-align: right; padding: 4px; font-weight: 700; color: #fff;">${mae.toFixed(2)}</td>
              <td style="text-align: right; padding: 4px; color: ${bias >= 0 ? '#00e676' : '#ff1744'};">${bias >= 0 ? '+' : ''}${bias.toFixed(2)}</td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;
  document.getElementById('m3DiagnosticsRoleStats').innerHTML = roleStatsHtml;

  // Update Group Diagnostics table on right
  updateM3GroupDiagnosticsTable();

  // 5. Paginate and Render Table
  const totalPages = Math.ceil(filtered.length / m3DiagPageSize) || 1;
  if (m3DiagCurrentPage > totalPages) m3DiagCurrentPage = totalPages;
  const startIdx = (m3DiagCurrentPage - 1) * m3DiagPageSize;
  const pageRows = filtered.slice(startIdx, startIdx + m3DiagPageSize);

  if (pageRows.length === 0) {
    document.getElementById('m3DiagnosticsTableBody').innerHTML = `
      <tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No player records match selected filters.</td></tr>
    `;
    document.getElementById('m3DiagnosticsPagination').innerHTML = '';
  } else {
    // Populate Default detail view on first load or filter change
    if (!selectedDiagPlayerPeriod && pageRows.length > 0) {
      selectedDiagPlayerPeriod = pageRows[0];
    }

    const rowsHtml = pageRows.map(r => {
      const isSelected = selectedDiagPlayerPeriod &&
        selectedDiagPlayerPeriod.player_id === r.player_id &&
        selectedDiagPlayerPeriod.prediction_period_id === r.prediction_period_id;
      const highlightClass = isSelected ? 'swapped-row' : '';
      const errColor = r.signed_error >= 0 ? '#00e676' : '#ff1744';
      const errSign = r.signed_error >= 0 ? '+' : '';

      return `
        <tr class="${highlightClass}" style="border-bottom: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: all 0.2s;" onclick="selectM3DiagnosticsRow('${escapeHtml(r.player_id)}', '${escapeHtml(r.prediction_period_id)}')">
          <td style="padding: 8px;">${escapeHtml(r.week_id)}</td>
          <td style="padding: 8px; font-weight: 700; color: #fff;">${escapeHtml(r.player_name)}</td>
          <td style="padding: 8px; text-transform: uppercase;">${escapeHtml(r.role)}</td>
          <td style="padding: 8px;">${escapeHtml(r.player_team_at_period)}</td>
          <td style="padding: 8px; color: var(--text-muted);">${escapeHtml(r.opponent_team_at_period)}</td>
          <td style="text-align: right; padding: 8px;">${r.projection_m3.toFixed(2)}</td>
          <td style="text-align: right; padding: 8px; font-weight: 700; color: #fff;">${r.actual_player_only_points.toFixed(2)}</td>
          <td style="text-align: right; padding: 8px; font-weight: 700; color: ${errColor};">${errSign}${r.signed_error.toFixed(2)}</td>
          <td style="text-align: right; padding: 8px; color: var(--text-muted);">${r.absolute_error.toFixed(2)}</td>
        </tr>
      `;
    }).join('');
    document.getElementById('m3DiagnosticsTableBody').innerHTML = rowsHtml;

    // Render Pagination Info
    document.getElementById('m3DiagnosticsPagination').innerHTML = `
      <div>Showing ${startIdx + 1} - ${Math.min(startIdx + m3DiagPageSize, filtered.length)} of ${filtered.length} entries</div>
      <div style="display: flex; gap: 8px;">
        <button class="eval-subtab-btn" onclick="changeM3DiagPage(-1)" ${m3DiagCurrentPage === 1 ? 'disabled style="opacity: 0.5; cursor: default;"' : ''}>Previous</button>
        <div style="padding: 6px 12px; font-weight: 700; color: #fff;">Page ${m3DiagCurrentPage} of ${totalPages}</div>
        <button class="eval-subtab-btn" onclick="changeM3DiagPage(1)" ${m3DiagCurrentPage === totalPages ? 'disabled style="opacity: 0.5; cursor: default;"' : ''}>Next</button>
      </div>
    `;

    // Wire Sort headers
    document.querySelectorAll('#view-historical-lineups th[data-sort]').forEach(th => {
      const col = th.dataset.sort;
      const isSorted = m3DiagSort.col === col;
      const arrow = isSorted ? (m3DiagSort.dir === 'asc' ? ' ▲' : ' ▼') : ' ↕';
      th.innerHTML = th.innerHTML.replace(/[▲▼↕]/, arrow);
      th.onclick = () => {
        if (m3DiagSort.col === col) {
          m3DiagSort.dir = m3DiagSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          m3DiagSort.col = col;
          m3DiagSort.dir = 'desc';
        }
        renderM3Diagnostics();
      };
    });
  }

  // 6. Populate Player Detail Panel
  renderM3DiagnosticsDetail();
}

window.selectM3DiagnosticsRow = function(playerId, periodId) {
  if (!evalM3Diagnostics) return;
  const found = evalM3Diagnostics.find(r => r.player_id === playerId && r.prediction_period_id === periodId);
  if (found) {
    selectedDiagPlayerPeriod = found;
    renderM3Diagnostics();
  }
};

window.changeM3DiagPage = function(delta) {
  m3DiagCurrentPage += delta;
  renderM3Diagnostics();
};

function updateM3GroupDiagnosticsTable() {
  if (!evalM3Diagnostics) return;

  const keyField = m3GroupTabActive === 'week'
    ? 'week_id'
    : (m3GroupTabActive === 'team' ? 'player_team_at_period' : 'fallback_level');

  const grouped = {};
  evalM3Diagnostics.forEach(r => {
    const k = r[keyField];
    grouped[k] = grouped[k] || { abs: 0, sign: 0, count: 0 };
    grouped[k].abs += r.absolute_error;
    grouped[k].sign += r.signed_error;
    grouped[k].count++;
  });

  const headers = {
    week: 'Week',
    team: 'Team',
    fallback: 'Fallback Level'
  };

  const groupRowsHtml = Object.keys(grouped).sort().map(k => {
    const g = grouped[k];
    const mae = g.abs / g.count;
    const bias = g.sign / g.count;
    return `
      <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
        <td style="padding: 6px;"><strong>${escapeHtml(k)}</strong></td>
        <td style="text-align: right; padding: 6px; color: var(--text-muted);">${g.count}</td>
        <td style="text-align: right; padding: 6px; font-weight: 700; color: #fff;">${mae.toFixed(2)}</td>
        <td style="text-align: right; padding: 6px; color: ${bias >= 0 ? '#00e676' : '#ff1744'};">${bias >= 0 ? '+' : ''}${bias.toFixed(2)}</td>
      </tr>
    `;
  }).join('');

  document.getElementById('m3DiagnosticsGroupTable').innerHTML = `
    <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
      <thead>
        <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-muted); position: sticky; top: 0; background: #0c1017; z-index: 10;">
          <th style="text-align: left; padding: 6px;">${headers[m3GroupTabActive]}</th>
          <th style="text-align: right; padding: 6px;">n</th>
          <th style="text-align: right; padding: 6px;">MAE</th>
          <th style="text-align: right; padding: 6px;">Bias</th>
        </tr>
      </thead>
      <tbody>
        ${groupRowsHtml}
      </tbody>
    </table>
  `;
}

function renderM3DiagnosticsDetail() {
  const panel = document.getElementById('m3DiagnosticsDetailPanel');
  if (!panel) return;
  if (!selectedDiagPlayerPeriod) {
    panel.innerHTML = `
      <div style="height: 100%; display: flex; align-items: center; justify-content: center; text-align: center; color: var(--text-muted); padding: 40px;">
        Select a player row to inspect M3 point-in-time features, target labels, and residual error.
      </div>
    `;
    return;
  }

  const r = selectedDiagPlayerPeriod;
  const errColor = r.signed_error >= 0 ? '#00e676' : '#ff1744';
  const errSign = r.signed_error >= 0 ? '+' : '';

  // Categorize error bucket
  let bucket = '';
  if (r.absolute_error < 0.5) bucket = '<span class="price-badge up" style="background:rgba(0,230,118,0.15); border:1px solid rgba(0,230,118,0.3); color:#00e676;">NEAR_EXACT (< 0.5)</span>';
  else if (r.signed_error < 0) bucket = '<span class="price-badge down" style="background:rgba(255,23,68,0.15); border:1px solid rgba(255,23,68,0.3); color:#ff1744;">OVERPREDICTED (>= 0.5)</span>';
  else bucket = '<span class="price-badge up" style="background:rgba(76,201,240,0.15); border:1px solid rgba(76,201,240,0.3); color:var(--primary-color);">UNDERPREDICTED (>= 0.5)</span>';

  panel.innerHTML = `
    <div class="card-header" style="border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 15px;">
      <h3 class="card-title">🔍 Diagnostic Panel: ${escapeHtml(r.player_name)}</h3>
    </div>

    <div style="font-size: 12.5px; line-height: 1.6; color: var(--text-muted);">
      <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div><strong>Role:</strong> <span style="text-transform: uppercase; color: #fff;">${escapeHtml(r.role)}</span></div>
        <div>${bucket}</div>
      </div>

      <!-- Key Identifiers -->
      <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 10px;">
        <div><strong>Player ID:</strong> <code style="word-break: break-all; font-size: 11px;">${escapeHtml(r.player_id)}</code></div>
        <div><strong>Period ID:</strong> <code style="word-break: break-all; font-size: 11px;">${escapeHtml(r.prediction_period_id)}</code></div>
        <div><strong>Week:</strong> <span style="color: #fff;">${escapeHtml(r.week_id)}</span></div>
        <div><strong>Target Cutoff:</strong> <code style="font-size: 11px;">${escapeHtml(r.target_cutoff)}</code></div>
      </div>

      <!-- Projections & Error -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; background: rgba(255,255,255,0.02); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.03);">
        <div>
          <span style="font-size: 11px; display: block;">M3 Projected Points</span>
          <strong style="font-size: 18px; color: #fff;">${r.projection_m3.toFixed(2)}</strong>
        </div>
        <div>
          <span style="font-size: 11px; display: block;">Actual Player Points</span>
          <strong style="font-size: 18px; color: #fff;">${r.actual_player_only_points.toFixed(2)}</strong>
        </div>
        <div style="margin-top: 5px;">
          <span style="font-size: 11px; display: block;">Signed Error</span>
          <strong style="font-size: 16px; color: ${errColor};">${errSign}${r.signed_error.toFixed(2)}</strong>
        </div>
        <div style="margin-top: 5px;">
          <span style="font-size: 11px; display: block;">Absolute Error (MAE)</span>
          <strong style="font-size: 16px; color: #fff;">${r.absolute_error.toFixed(2)}</strong>
        </div>
      </div>

      <!-- Match Context -->
      <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <h4 style="margin: 0 0 6px; font-size: 12px; color: var(--primary-color);">Matchup Context</h4>
        <div><strong>Player Team at Period:</strong> <span style="color: #fff;">${escapeHtml(r.player_team_at_period)}</span></div>
        <div><strong>Opponent(s) at Period:</strong> <span style="color: #fff;">${escapeHtml(r.opponent_team_at_period)}</span></div>
        <div style="font-size: 10px; color: var(--text-muted); font-style: italic; margin-top: 4px;">
          ⚠️ Opponent context is retrospective diagnostic metadata only.
        </div>
      </div>

      <!-- Point in time features -->
      <div>
        <h4 style="margin: 0 0 6px; font-size: 12px; color: var(--primary-color);">Point-in-Time Features</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11.5px;">
          <div><strong>Fallback Level:</strong> <span style="color: #fff; text-transform: uppercase;">${escapeHtml(r.fallback_level)}</span></div>
          <div><strong>History Count:</strong> <span style="color: #fff;">${r.history_count} games</span></div>
          <div><strong>Uncertainty:</strong> <span style="color: #fff;">${r.uncertainty !== null ? r.uncertainty.toFixed(4) : 'N/A'}</span></div>
          <div><strong>Core V2 Status:</strong> <span style="color: #fff;">${r.core_status !== null ? r.core_status.toFixed(4) : 'N/A'}</span></div>
          <div style="grid-column: span 2;"><strong>Team Strength Context:</strong> <span style="color: #fff;">${r.team_context_coverage !== null ? r.team_context_coverage.toFixed(4) : 'N/A'}</span></div>
        </div>
      </div>

      <!-- Provenance -->
      <div style="margin-top: 15px; font-size: 10.5px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; color: var(--text-muted);">
        <div><strong>Model Spec Hash:</strong> <code style="word-break: break-all;">${escapeHtml(r.model_artifact_sha256)}</code></div>
        <div><strong>Data Quality Status:</strong> <code style="color: #00e676;">${escapeHtml(r.data_quality_status)}</code></div>
      </div>
    </div>
  `;
}
