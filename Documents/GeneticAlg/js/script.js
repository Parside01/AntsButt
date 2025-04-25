let cities = [];
let scale = 5;


const canvas = document.getElementById('myCanvas');
const canvasWidth = canvas.width;
const canvasHeight = canvas.height;

function setCanvasSize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    updateScaleValue(scale);
}

window.addEventListener('load', setCanvasSize);
window.addEventListener('resize', setCanvasSize);

const ctx = canvas.getContext('2d');

function updateScaleValue(value) {
    scale = parseFloat(value);
    scaleValue.textContent = value;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < cities.length; i++) {
        drawPoint(cities[i].x, cities[i].y, ctx, transformCoordinates);
    }
}

function transformCoordinates(x, y) {
    return {
        x: (canvas.width / 2) + (x * canvas.width / (2 * scale)),
        y: (canvas.height / 2) - (y * canvas.height / (2 * scale))
    };
}

let isFirstPoint = true;

function drawPoint(x, y, ctx, transformCoordinates, isFirst = false) {
    const transformed = transformCoordinates(x, y);

    ctx.beginPath();
    ctx.arc(transformed.x, transformed.y, 6, 0, 2 * Math.PI);

    if (isFirst) {
        ctx.fillStyle = 'rgb(255, 201, 14)';
    } else {
        ctx.fillStyle = 'white';
    }
    ctx.fill();
}

const coordinatesDiv = document.getElementById('coordinates');

canvas.addEventListener('mousemove', function (event) {
    const x = event.offsetX;
    const y = event.offsetY;

    const worldCoordinates = {
        x: (x - (canvas.width / 2)) * (2 * scale) / canvas.width,
        y: ((canvas.height / 2) - y) * (2 * scale) / canvas.height
    };

    coordinatesDiv.textContent = `X: ${worldCoordinates.x.toFixed(2)}, Y: ${worldCoordinates.y.toFixed(2)}`;
});

canvas.addEventListener('click', function (event) {
    const x = event.offsetX;
    const y = event.offsetY;

    const worldCoordinates = {
        x: (x - (canvas.width / 2)) * (2 * scale) / canvas.width,
        y: ((canvas.height / 2) - y) * (2 * scale) / canvas.height
    };

    cities.push({ x: worldCoordinates.x, y: worldCoordinates.y });

    if (cities.length === 1) {
        localStorage.setItem('firstPointIndex', 0);
    }

    drawPoint(worldCoordinates.x, worldCoordinates.y, ctx, transformCoordinates, cities.length === 1);
});

const coordinateInput = document.getElementById('coordinateInput');
const addPointButton = document.getElementById('addPointButton');

addPointButton.addEventListener('click', function () {
    const coordinates = coordinateInput.value.split(/[, ]+/);
    const x = parseFloat(coordinates[0]);
    const y = parseFloat(coordinates[1]);

    if (!isNaN(x) && !isNaN(y)) {
        const worldCoordinates = { x: x, y: y };
        cities.push({ x: worldCoordinates.x, y: worldCoordinates.y });

        if (cities.length === 1) {
            localStorage.setItem('firstPointIndex', 0);
        }

        drawPoint(worldCoordinates.x, worldCoordinates.y, ctx, transformCoordinates, cities.length === 1);
    }
    else {
        alert("Введите корректные координаты X и Y через запятую или пробел.");
    }
});

function clearRoute(canvas, ctx, cities, drawPoint, transformCoordinates) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const firstPointIndex = localStorage.getItem('firstPointIndex');

    for (let i = 0; i < cities.length; i++) {
        drawPoint(cities[i].x, cities[i].y, ctx, transformCoordinates, i === parseInt(firstPointIndex));
    }
}

const runButton = document.getElementById('runButton');
const algorithmStatus = document.getElementById('algorithmStatus');

runButton.addEventListener('click', function () {
    if (cities.length < 3) {
        alert("Добавьте хотя бы 3 города!");
        return;
    }

    if (!algorithmRunning) {
        algorithmRunning = true;
        algorithmStatus.textContent = "Алгоритм запущен";
        animationInterval = setInterval(updateAlgorithmStatus, 500);

        voidGenethicAlgorithm(cities);
    }
});

const clearButton = document.getElementById('clearButton');
const generationInfo = document.getElementById('generationInfo');

clearButton.addEventListener('click', function () {
    cities = [];
    isFirstPoint = true;
    localStorage.removeItem('firstPointIndex');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (algorithmRunning) {
        algorithmRunning = false;
        clearInterval(animationInterval);
        algorithmStatus.textContent = "Запустите алгоритм";
        generationInfo.textContent = "Поколение: 0 | Приспособленность: 0";
    }
});

function voidGenethicAlgorithm(cities) {
    let populationCount = 300;
    let cityCount = cities.length;

    let distance = new Array(cityCount);
    for (let i = 0; i < cityCount; i++) {
        distance[i] = new Array(cityCount);
        for (let j = 0; j < cityCount; j++) {
            distance[i][j] = (i === j) ? 0 : calculateDistance(cities[i], cities[j]);
        }
    }

    let currentGeneration = 0;
    let population = createInitialPopulation(populationCount, cityCount, distance);
    let generationCount;

    if (cityCount < 5) {
        generationCount = 10;
    }
    else if (cityCount < 10) {
        generationCount = 30;
    }
    else if (cityCount < 20) {
        generationCount = 100;
    }
    else if (cityCount < 30) {
        generationCount = 200;
    }
    else if (cityCount < 40) {
        generationCount = 500;
    }
    else if (cityCount < 50) {
        generationCount = 1000;
    }
    else if (cityCount < 100) {
        generationCount = 2000;
    }
    else {
        generationCount = 5000;
    }

    function iterate() {
        if (!algorithmRunning) {
            return;
        }

        if (currentGeneration >= generationCount) {
            algorithmRunning = false;
            clearInterval(animationInterval);
            algorithmStatus.textContent = "Алгоритм завершен";
            return;
        }

        let newGeneration = [];

        population.sort(compareIndividuals);

        let eliteSize = Math.round((20.0 * populationCount) / 100.0);
        for (let i = 0; i < eliteSize; i++) {
            newGeneration.push(population[i]);
        }

        while (newGeneration.length < populationCount) {
            let r = randomNum(0, population.length - 1);
            let parent1 = population[r];
            r = randomNum(0, population.length - 1);
            let parent2 = population[r];
            let child = mate(parent1, parent2, cityCount, distance);
            newGeneration.push(child);
        }

        population = newGeneration;

        clearRoute(canvas, ctx, cities, drawPoint, transformCoordinates);
        drawRoute(population[0].chromosome, cities, ctx, transformCoordinates);

        currentGeneration++;

        generationInfo.textContent = `Поколение: ${currentGeneration} | Приспособленность: ${Math.floor(population[0].fitness)}`;

        requestAnimationFrame(() => setTimeout(iterate, 200));
    }

    iterate();
}


let algorithmRunning = false;
let animationInterval;
const animationStates = [
    "Алгоритм запущен",
    "Алгоритм запущен .",
    "Алгоритм запущен . .",
    "Алгоритм запущен . . ."
];
let currentAnimationState = 0;

function updateAlgorithmStatus() {
    algorithmStatus.textContent = animationStates[currentAnimationState];
    currentAnimationState = (currentAnimationState + 1) % animationStates.length;
}

const scaleSlider = document.getElementById('scaleSlider');
const scaleValue = document.getElementById('scaleValue');

window.addEventListener('load', function () {
    updateScaleValue(scaleSlider.value);
});

scaleSlider.addEventListener('input', function () {
    updateScaleValue(scaleSlider.value);
});