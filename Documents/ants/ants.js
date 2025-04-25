const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let colony = [];
let pheromones = [];
let bestPath = [];
let bestDistance = Infinity;

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  colony.push({ x, y });
  drawcolony();
});

function drawcolony() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < colony.length; i++) {
    const city = colony[i];

    if (i === 0) {
      ctx.beginPath();
      ctx.arc(city.x, city.y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = "white";
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(city.x, city.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ffeb3b";
      ctx.fill();
    } else {

      ctx.beginPath();
      ctx.arc(city.x, city.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
    }
  }

  if (bestPath.length > 0) {
    ctx.strokeStyle = "#ffeb3b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(colony[bestPath[0]].x, colony[bestPath[0]].y);
    for (let i = 1; i < bestPath.length; i++) {
      ctx.lineTo(colony[bestPath[i]].x, colony[bestPath[i]].y);
    }
    ctx.lineTo(colony[bestPath[0]].x, colony[bestPath[0]].y);
    ctx.stroke();
  }
}



function distance(a, b) {
  const dx = colony[a].x - colony[b].x;
  const dy = colony[a].y - colony[b].y;
  return Math.sqrt(dx * dx + dy * dy);
}

function initializePheromones() {
  const n = colony.length;
  pheromones = Array.from({ length: n }, () => Array(n).fill(1));
}

function chooseNextColony(visited, currentColony, alpha = 1, beta = 5) {
  const n = colony.length;
  const probabilities = [];
  let sum = 0;

  for (let j = 0; j < n; j++) {
    if (!visited[j]) {
      const tau = pheromones[currentColony][j] ** alpha;
      const eta = (1 / distance(currentColony, j)) ** beta;
      const prob = tau * eta;
      probabilities[j] = prob;
      sum += prob;
    } else {
      probabilities[j] = 0;
    }
  }

  let r = Math.random() * sum;
  for (let j = 0; j < n; j++) {
    if (probabilities[j] > 0) {
      r -= probabilities[j];
      if (r <= 0) return j;
    }
  }

  return probabilities.findIndex((p) => p > 0);
}

function runAntColony(iterations = 100, ants = 20, evaporation = 0.5, Q = 100) {
  const n = colony.length;
  if (n < 2) return;

  initializePheromones();

  for (let iter = 0; iter < iterations; iter++) {
    const paths = [];
    const distances = [];

    for (let k = 0; k < ants; k++) {
      const visited = Array(n).fill(false);
      const path = [];
      let current = Math.floor(Math.random() * n);
      visited[current] = true;
      path.push(current);

      for (let step = 1; step < n; step++) {
        const next = chooseNextColony(visited, current);
        visited[next] = true;
        path.push(next);
        current = next;
      }

      const totalDistance = path.reduce((sum, city, i) => {
        const nextColony = path[(i + 1) % n];
        return sum + distance(city, nextColony);
      }, 0);

      paths.push(path);
      distances.push(totalDistance);

      if (totalDistance < bestDistance) {
        bestDistance = totalDistance;
        bestPath = path.slice();
        drawcolony();
      }
    }

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        pheromones[i][j] *= (1 - evaporation);
      }
    }

    for (let k = 0; k < ants; k++) {
      const path = paths[k];
      const dist = distances[k];
      for (let i = 0; i < path.length; i++) {
        const from = path[i];
        const to = path[(i + 1) % n];
        pheromones[from][to] += Q / dist;
        pheromones[to][from] += Q / dist;
      }
    }
  }
}

function startAntColony() {
  bestDistance = Infinity;
  bestPath = [];
  runAntColony();
  drawcolony();
}

function reset() {
  colony = [];
  pheromones = [];
  bestPath = [];
  bestDistance = Infinity;
  drawcolony();
}
