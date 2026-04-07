const weatherEmoji = {
    sunny: "☀️🌾",
    cloudy: "☁️🌾",
    rainy: "🌧️🌾",
    stormy: "⛈️🌾"
};

const growthEmoji = [
    "🌱", "🌿", "🌾", "🌾🌾", "🌾🌾🌾", "🏆🌾"
];

let isAutoTraining = false;

async function resetEnv() {
    const res = await fetch("/reset", { method: "POST" });
    const data = await res.json();
    if (data.success) {
        updateState(data.state);
        addLog("🔄 Environment Reset!");
    }
}

async function takeAction(action) {
    const actions = [
        "💧 Watered crops",
        "🌿 Applied fertilizer",
        "🔫 Applied pesticide",
        "⏸ Did nothing",
        "🌾 Harvested!"
    ];
    const res = await fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
    });
    const data = await res.json();
    if (data.success) {
        updateState(data.state);
        addLog(`${actions[action]} | Reward: ${data.reward} | ${data.info}`);
        if (data.done) {
            addLog(`✅ Episode Done! Total Reward: ${data.state.yield_score}`);
        }
    }
}

async function trainAgent() {
    const btn = document.getElementById("train-btn");
    btn.disabled = true;
    btn.innerText = "⏳ Training...";

    const res = await fetch("/agent/run", { method: "POST" });
    const data = await res.json();

    if (data.success) {
        document.getElementById("episode").innerText = data.episode;
        document.getElementById("best-reward").innerText = data.best_reward;
        document.getElementById("epsilon").innerText = data.epsilon;
        document.getElementById("last-reward").innerText = data.total_reward;

        updateChartBars(data.reward_history);
        updateState(data.steps[data.steps.length - 1]);

        addLog(`--- Episode ${data.episode} ---`);
        data.steps.forEach(step => {
            addLog(`Day ${step.day}: ${step.action} | ${step.info} | Reward: ${step.reward}`);
        });
        addLog(`✅ Total Reward: ${data.total_reward} | Epsilon: ${data.epsilon}`);
    }

    btn.disabled = false;
    btn.innerText = "▶ Run Agent Episode";
}

async function autoTrain() {
    if (isAutoTraining) {
        isAutoTraining = false;
        document.getElementById("auto-btn").innerText = "⚡ Auto Train (10 Episodes)";
        return;
    }

    isAutoTraining = true;
    document.getElementById("auto-btn").innerText = "⏹ Stop Auto Train";

    for (let i = 0; i < 10; i++) {
        if (!isAutoTraining) break;
        await trainAgent();
        await sleep(500);
    }

    isAutoTraining = false;
    document.getElementById("auto-btn").innerText = "⚡ Auto Train (10 Episodes)";
}

function updateState(state) {
    if (!state) return;

    // Update farm visual
    const growth = Math.floor(state.growth_stage || 0);
    const visual = document.getElementById("farm-visual");
    if (visual) {
        visual.innerText = weatherEmoji[state.weather] || "🌾";
    }

    // Update values
    setText("day", state.day || 0);
    setText("weather", state.weather || "sunny");
    setText("water", (state.water_level || 0).toFixed(1) + "%");
    setText("soil", (state.soil_health || 0).toFixed(1) + "%");
    setText("pest", (state.pest_level || 0).toFixed(1) + "%");
    setText("growth", (state.growth_stage || 0).toFixed(2));

    // Update progress bars
    setBar("water-bar", state.water_level || 0);
    setBar("soil-bar", state.soil_health || 0);
    setBar("pest-bar", state.pest_level || 0);
    setBar("growth-bar", (state.growth_stage || 0) * 20);
}

function updateChartBars(history) {
    const container = document.getElementById("chart-bars");
    if (!container || !history.length) return;

    const max = Math.max(...history.map(Math.abs)) || 1;
    container.innerHTML = history.map(val => {
        const height = Math.abs(val) / max * 100;
        const isNeg = val < 0;
        return `<div class="chart-bar ${isNeg ? 'negative' : ''}"
            style="height: ${height}%"
            title="Reward: ${val}"></div>`;
    }).join("");
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

function setBar(id, value) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min(100, value) + "%";
}

function addLog(message) {
    const log = document.getElementById("log-box");
    if (!log) return;
    log.innerHTML += `<div>> ${message}</div>`;
    log.scrollTop = log.scrollHeight;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Initialize
window.onload = () => {
    resetEnv();
    addLog("🌾 FarmEnv AI Ready!");
    addLog("📋 Use manual actions or run AI agent!");
};
