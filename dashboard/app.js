// LCS Fantasy Interactive Weekly Dashboard Logic

let rawData = null;
let championLabData = null;
let weeklyChampionData = null;
let matchupOptimizerData = null;
let selectedMatchupLineupRank = 1;
let filteredPlayers = [];
let currentPositionFilter = 'ALL';
let currentSortCol = 'total_pts';
let currentSortDir = 'desc';
let pointsMode = 'raw'; // 'raw' or 'adjusted'
let trendChart = null;
let championPoolChart = null;
let championSplitChart = null;

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
  try {
    const [
      resp,
      championResp,
      weeklyChampionResp,
      matchupOptimizerResp
    ] = await Promise.all([
      fetch('./dashboard_data.json'),
      fetch('./champion_lab_data.json'),
      fetch('./weekly_champion_predictions.json'),
      fetch('./matchup_lineups.json')
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
}

function renderWeeklyChampionPicks() {
  const container = document.getElementById('weeklyChampionMatchups');
  const notice = document.getElementById('weeklyChampionNotice');
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
    ? 'Each column shows up to three candidates eligible for that official multiplier. Pick chance is a normalized model estimate, not a calibrated probability.'
    : 'No eligible champion predictions are available for this round.';

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
        <div class="weekly-pick-options">
          ${options.map((pick, index) => `
            <div class="weekly-pick-option">
              <span class="weekly-pick-rank">${escapeHtml(pick.option_basis || `#${index + 1}`)}</span>
              <div>
                <strong>${escapeHtml(pick.champion)}</strong>
                <span>${(Number(pick.estimated_pick_chance ?? pick.ranking_share) * 100).toFixed(1)}% estimated pick chance</span>
                <small>Available ${(Number(pick.availability) * 100).toFixed(0)}% · Bonus ${Number(pick.expected_multiplier_bonus).toFixed(2)}</small>
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

function matchupChampionOptions(player) {
  let rawOptions = [];
  if (Array.isArray(player.champion_options) && player.champion_options.length) {
    rawOptions = player.champion_options;
  } else {
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
  meta.textContent = `${selectedWeek.round_name} | Roster lock ${selectedWeek.roster_lock} | ${Number(selectedWeek.budget).toFixed(1)} gold budget`;
  notice.textContent =
    'Lineups are ranked by projected points after matchup-conflict risk. Opposing TOP exposure receives half the normal penalty because TOP scores have been more stable. Champion chances are normalized model estimates.';
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
    const options = matchupChampionOptions(player);
    if (!options.length) {
      return '<div class="optimizer-no-picks">No champion recommendations available</div>';
    }
    const oppOptions = opponentPlayer ? matchupChampionOptions(opponentPlayer) : [];
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
                <small>${escapeHtml(pick.option_basis || '')} ${(Number(pick.estimated_pick_chance ?? pick.ranking_share) * 100).toFixed(1)}% pick chance</small>
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
        <p class="optimizer-coach-note">Coach score is projected from the average of the team's five starters. The coach's organization counts toward variety.</p>
      </article>
    </div>
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
      source: player.pricing_source || 'estimated_split_reset_diminishing'
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
    source: latest.source || 'estimated_split_reset_diminishing'
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
    : '<div style="color: var(--text-muted); font-size: 12px; margin-top: 3px;">Experimental estimated price; no official snapshot captured</div>';

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
    const teamColors = historyToUse.map(h => getTeamColor(h.teamname || player.teamname));
    const primaryTeamColor = teamColors[teamColors.length - 1] || getTeamColor(player.teamname);
    const patchMarkers = buildPatchMarkers(historyToUse);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Market Price (Gold)',
          data: prices,
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
        }]
      },
      plugins: [patchBoundaryPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          patchBoundaries: { markers: patchMarkers },
          tooltip: {
            callbacks: {
              afterTitle: items => {
                const entry = historyToUse[items[0].dataIndex];
                return entry && entry.patch ? `Patch ${entry.patch}` : '';
              },
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
