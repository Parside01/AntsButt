function csvToArray(csvText) {
    const lines = csvText.split('\n');
    const data = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line === '') {
            continue;
        }

        const values = line.split(/\s+/);
        const numbers = values.map(value => parseFloat(value.trim()));

        if (numbers.every(number => !isNaN(number))) {
            data.push(numbers);
        }
    }

    return data;
}

const trainingDataTextarea = document.getElementById('trainingData');
let trainingData = [];

trainingDataTextarea.addEventListener('input', function () {
    let csvText = trainingDataTextarea.value;
    trainingData = csvToArray(csvText);
    updateMaxDepth(trainingData);
});

const treeDepthSlider = document.getElementById('treeDepth');
const depthValueSpan = document.getElementById('depthValue');

function updateMaxDepth(data) {
    const maxDepth = data.length > 0 ? data.length - 1 : 3;
    treeDepthSlider.max = maxDepth;
    treeDepthSlider.value = maxDepth;
    depthValueSpan.textContent = maxDepth;
}

treeDepthSlider.addEventListener('input', function () {
    depthValueSpan.textContent = treeDepthSlider.value;
});

function csvToFeaturesArray(csvText) {
    const lines = csvText.split('\n');
    const data = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line === '') {
            continue;
        }

        const values = line.split(',');
        const numbers = values.map(value => parseFloat(value.trim()));

        if (numbers.every(number => !isNaN(number))) {
            data.push(numbers);
        }
    }

    return data;
}

const testDataTextarea = document.getElementById('testData');
let testFeatures = [];

testDataTextarea.addEventListener('input', function () {
    let csvText = testDataTextarea.value;
    testFeatures = csvToFeaturesArray(csvText);
});

const taskTypeSelect = document.getElementById('taskType');
const trainTreeButton = document.getElementById('trainTree');

let decisionTree = null;

trainTreeButton.addEventListener('click', function () {
    const taskType = taskTypeSelect.value;
    const depth = parseInt(depthValueSpan.textContent, 10);
    nodeCounter = 0;

    if (trainingData.length === 0) {
        alert("Пожалуйста, введите обучающие данные.");
        return;
    }

    if (decisionTree) {
        if (taskType === 'classification') deleteTreeClassification(decisionTree);
        else deleteTreeRegression(decisionTree);
    }

    const allIndexes = Array.from({ length: trainingData.length }, (_, i) => i);
    if (taskType === 'classification') {
        decisionTree = buildTreeClassification(trainingData, allIndexes, depth);
        const treeJson = treeToJsonClassification(decisionTree);
        drawTree(treeJson);
    } else {
        decisionTree = buildTreeRegression(trainingData, allIndexes, depth);
        const treeJson = treeToJsonRegression(decisionTree);
        drawTree(treeJson);
    }
});

const testButton = document.getElementById('testTree');
const predictionResultInput = document.getElementById('predictionResult');

const optimizeButton = document.getElementById('optimizeTree');

optimizeButton.addEventListener('click', function () {
    nodeCounter = 0;
    if (trainingData.length === 0) {
        alert("Пожалуйста, введите обучающие данные.");
        return;
    }

    const taskType = taskTypeSelect.value;

    if (decisionTree) {
        if (taskType === 'classification') deleteTreeClassification(decisionTree);
        else deleteTreeRegression(decisionTree);
    }
    currentPath = [];
    currentNodeIndex = 0;
    isManualSelection = false;
    lastTestData = null;
    d3.selectAll(".tree-node").attr("fill", "white");
    document.getElementById("nodeInfoContainer").innerHTML = "";

    const allIndexes = Array.from({ length: trainingData.length }, (_, i) => i);

    if (taskType === 'classification') {
        let optimalDepthClassification = chooseOptimalDepthClassification(trainingData);
        decisionTree = buildTreeClassification(trainingData, allIndexes, optimalDepthClassification);
        const treeJson = treeToJsonClassification(decisionTree);
        drawTree(treeJson);
    } else {
        let optimalDepthRegression = chooseOptimalDepthRegression(trainingData);
        decisionTree = buildTreeRegression(trainingData, allIndexes, optimalDepthRegression);
        const treeJson = treeToJsonRegression(decisionTree);
        drawTree(treeJson);
    }

})

let currentPath = [];
let currentNodeIndex = 0;
let isManualSelection = false;
let lastTestData = null;

function highlightCurrentNode(nodeId) {
    d3.selectAll(".tree-node")
        .attr("fill", "white");

    d3.select(`.tree-node[data-id="${nodeId}"]`)
        .attr("fill", isManualSelection ? "rgb(255, 240, 0)" : "rgb(255, 201, 14)");
}



function drawTree(treeData) {
    const svg = d3.select("#tree-svg");
    svg.selectAll("*").remove();

    const width = 1200;
    const height = 600;
    svg.attr("width", width).attr("height", height);

    const g = svg.append("g")
        .attr("transform", "translate(100, 50)");

    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().size([width - 200, height - 150]);
    treeLayout(root);

    const allNodes = root.descendants();

    g.append("g")
        .selectAll("path")
        .data(root.links())
        .enter()
        .append("path")
        .attr("d", d3.linkVertical()
            .x(d => d.x)
            .y(d => d.y))
        .attr("fill", "none")
        .attr("stroke", "rgb(255, 240, 0)");

    const nodes = g.append("g")
        .selectAll("g")
        .data(allNodes)
        .enter()
        .append("g")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .on("click", (event, d) => {
            isManualSelection = true;
            highlightCurrentNode(d.data.id);
            showNodeInfo(d.data);

            const pathIndex = currentPath.findIndex(node => node.id === d.data.id);
            if (pathIndex !== -1) {
                currentNodeIndex = pathIndex;
            }
        });

    nodes.append("circle")
        .attr("r", 10)
        .attr("fill", "white")
        .attr("class", "tree-node")
        .attr("data-id", (d) => d.data.id);

    const zoom = d3.zoom()
        .scaleExtent([0.5, 3])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoom);
}


function showNodeInfo(nodeData) {
    const infoContainer = document.getElementById("nodeInfoContainer");
    let infoText = "";

    if (nodeData.isLeaf) {
        infoText = taskTypeSelect.value === 'classification'
            ? `ID: ${nodeData.id} | Class: ${nodeData.predictedClass || nodeData.value}`
            : `ID: ${nodeData.id} | Predict: ${nodeData.predictValue ? nodeData.predictValue.toFixed(2) : nodeData.value.toFixed(2)}`;
    } else {
        infoText = `ID: ${nodeData.id} | Feature ${nodeData.featureIndex} ≤ ${nodeData.threshold.toFixed(2)}`;
    }

    infoContainer.innerHTML = `<pre>${infoText}</pre>`;
}


function findPath(tree, example) {
    const path = [];
    let node = tree;

    while (node) {
        path.push({
            id: node.id,
            data: {
                id: node.id,
                featureIndex: node.featureIndex,
                threshold: node.threshold,
                isLeaf: node.isLeaf,
                predictedClass: node.predictedClass,
                predictValue: node.predictValue,
                value: node.predictedClass ?? node.predictValue
            }
        });

        if (node.isLeaf) break;
        node = (example[node.featureIndex] <= node.threshold) ? node.left : node.right;
    }

    return path;
}

document.getElementById("nextNode").addEventListener("click", () => {
    if (currentPath.length === 0 || currentNodeIndex === -1) {
        return;
    }

    if (currentNodeIndex < currentPath.length - 1) {
        currentNodeIndex++;
        isManualSelection = false;
        highlightCurrentNode(currentPath[currentNodeIndex].id);
        showNodeInfo(currentPath[currentNodeIndex].data);
    }
});

document.getElementById("prevNode").addEventListener("click", () => {
    if (currentPath.length === 0 || currentNodeIndex === -1) {
        return;
    }

    if (currentNodeIndex > 0) {
        currentNodeIndex--;
        isManualSelection = false;
        highlightCurrentNode(currentPath[currentNodeIndex].id);
        showNodeInfo(currentPath[currentNodeIndex].data);
    }
});

testButton.addEventListener('click', function () {
    if (!decisionTree) {
        alert("Пожалуйста, сначала обучите дерево.");
        return;
    }

    if (testFeatures.length !== 1) {
        alert("Пожалуйста, введите ровно одну строку тестовых данных.");
        return;
    }

    const testData = testFeatures[0];
    const taskType = taskTypeSelect.value;
    let prediction;

    if (taskType === 'classification') {
        prediction = predictClassification(decisionTree, testData);
    }
    else if (taskType === 'regression') {
        prediction = predictRegression(decisionTree, testData);
    }
    else {
        alert("Пожалуйста, выберите тип задачи.");
        return;
    }

    predictionResultInput.textContent = prediction;

    d3.selectAll(".tree-node")
        .attr("fill", "white");

    currentPath = findPath(decisionTree, testData);
    currentNodeIndex = 0;
    isTestMode = true;
    selectedNodeId = null;

    if (currentPath.length > 0) {
        highlightCurrentNode(currentPath[0].id);
        showNodeInfo(currentPath[0].data);
        currentNodeIndex = 0;
    }
});