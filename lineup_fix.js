// Replace the displayLineups function in app.py with this corrected version
function displayLineups(lineups, contestType) {
    const section = document.getElementById('lineups-section');
    const content = document.getElementById('lineups-content');
    
    if (!lineups || lineups.length === 0) {
        content.innerHTML = '<div class="error">No lineups generated</div>';
        section.style.display = 'block';
        return;
    }
    
    content.innerHTML = lineups.map((lineup, index) => {
        // Parse players and organize by position
        const playersByPos = {QB: [], RB: [], WR: [], TE: [], D: []};
        
        lineup.players.forEach(playerStr => {
            const parts = playerStr.split(' - ');
            if (parts.length >= 2) {
                const posTeam = parts[1].split('-');
                const pos = posTeam[0];
                if (playersByPos[pos]) {
                    playersByPos[pos].push(playerStr);
                }
            }
        });
        
        // Build FanDuel order: QB, RB, RB, WR, WR, WR, TE, FLEX, D
        let orderedPlayers = [];
        
        // QB (1)
        if (playersByPos.QB[0]) orderedPlayers.push({label: 'QB', player: playersByPos.QB[0]});
        
        // RB (2) 
        if (playersByPos.RB[0]) orderedPlayers.push({label: 'RB', player: playersByPos.RB[0]});
        if (playersByPos.RB[1]) orderedPlayers.push({label: 'RB', player: playersByPos.RB[1]});
        
        // WR (3)
        if (playersByPos.WR[0]) orderedPlayers.push({label: 'WR', player: playersByPos.WR[0]});
        if (playersByPos.WR[1]) orderedPlayers.push({label: 'WR', player: playersByPos.WR[1]});  
        if (playersByPos.WR[2]) orderedPlayers.push({label: 'WR', player: playersByPos.WR[2]});
        
        // TE (1)
        if (playersByPos.TE[0]) orderedPlayers.push({label: 'TE', player: playersByPos.TE[0]});
        
        // FLEX (remaining RB/WR/TE)
        const usedPlayers = orderedPlayers.map(p => p.player);
        const flexCandidates = [...playersByPos.RB, ...playersByPos.WR, ...playersByPos.TE]
            .filter(p => !usedPlayers.includes(p));
        if (flexCandidates[0]) orderedPlayers.push({label: 'FLEX', player: flexCandidates[0]});
        
        // D (1)
        if (playersByPos.D[0]) orderedPlayers.push({label: 'D', player: playersByPos.D[0]});
        
        const playersHtml = orderedPlayers.map(({label, player}) => 
            `<li><strong>${label}:</strong> ${player.split(' - ')[0]}<br><span class="player-detail">${player.split(' - ').slice(1).join(' - ')}</span></li>`
        ).join('');
        
        return `
            <div class="lineup-card ${contestType}">
                <h3>Lineup ${index + 1}</h3>
                <div><strong>Salary:</strong> $${lineup.total_salary.toLocaleString()}</div>
                <div><strong>Projected:</strong> ${lineup.projected_points.toFixed(1)} pts</div>
                <div><strong>Ceiling:</strong> ${lineup.ceiling_score.toFixed(1)} pts</div>
                <div><strong>Floor:</strong> ${lineup.floor_score.toFixed(1)} pts</div>
                <ul style="font-size: 13px;">${playersHtml}</ul>
            </div>
        `;
    }).join('');
    
    section.style.display = 'block';
}
