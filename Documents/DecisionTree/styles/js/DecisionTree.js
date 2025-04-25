function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

let nodeCounter = 0;

class NodeRegression {
  constructor() {
    this.featureIndex = -1;
    this.threshold = 0.0;
    this.predictValue = 0.0;
    this.left = null;
    this.right = null;
    this.isLeaf = false;
    this.id = -1;
  }
}

function calculatePredictRegression(data, indixes) {
  if (indixes.length === 0) {
    return 0.0;
  }

  let sum = 0.0;
  for (const index of indixes) {
    sum += data[index][data[0].length - 1];
  }

  return sum / indixes.length;
}

function calculateSME(data, indixes, predict) {
  if (indixes.length === 0) {
    return 0.0;
  }

  let mse = 0.0;
  for (const index of indixes) {
    let error = data[index][data[0].length - 1] - predict;
    mse += Math.pow(error, 2);
  }

  return mse / indixes.length;
}

function calculateIGRegression(
  mseParent,
  nLeft,
  mseLeft,
  nRight,
  mseRight,
  nParent
) {
  const informationGain =
    mseParent - ((nLeft / nParent) * mseLeft + (nRight / nParent) * mseRight);

  return informationGain;
}

function chooseTheBestSplitRegression(data, indixes) {
  let bestNode = new NodeRegression();
  let bestIG = -Infinity;

  const numberFeatures = data[0].length - 1;
  const numberExamples = indixes.length;
  const predictParent = calculatePredictRegression(data, indixes);
  const mseParent = calculateSME(data, indixes, predictParent);

  for (let featureIndex = 0; featureIndex < numberFeatures; featureIndex++) {
    let uniqueValues = [];
    for (let i = 0; i < indixes.length; i++) {
      const index = indixes[i];
      let isUnique = true;
      for (let k = 0; k < uniqueValues.length; k++) {
        if (data[index][featureIndex] === uniqueValues[k]) {
          isUnique = false;
          break;
        }
      }

      if (isUnique) {
        uniqueValues.push(data[index][featureIndex]);
      }
    }

    for (const threshold of uniqueValues) {
      let leftIndixes = [];
      let rightIndixes = [];

      for (let i = 0; i < indixes.length; ++i) {
        const index = indixes[i];
        if (data[index][featureIndex] <= threshold) {
          leftIndixes.push(index);
        } else {
          rightIndixes.push(index);
        }
      }

      let predictLeft = calculatePredictRegression(data, leftIndixes);
      let mseLeft = calculateSME(data, leftIndixes, predictLeft);
      let predictRight = calculatePredictRegression(data, rightIndixes);
      let mseRight = calculateSME(data, rightIndixes, predictRight);

      let informationGain = calculateIGRegression(
        mseParent,
        leftIndixes.length,
        mseLeft,
        rightIndixes.length,
        mseRight,
        numberExamples
      );

      if (informationGain > bestIG) {
        bestIG = informationGain;
        bestNode.featureIndex = featureIndex;
        bestNode.threshold = threshold;
      }
    }
  }

  return bestNode;
}

function buildTreeRegression(
  data,
  indixes,
  maxDepth,
  currentDepth = 0,
  minSamplesSplit = 2
) {
  if (indixes.length === 0) {
    return null;
  }

  if (indixes.length < minSamplesSplit) {
    let leaf = new NodeRegression();
    leaf.isLeaf = true;
    leaf.predictValue = calculatePredictRegression(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  if (currentDepth >= maxDepth) {
    let leaf = new NodeRegression();
    leaf.isLeaf = true;
    leaf.predictValue = calculatePredictRegression(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  let temp = chooseTheBestSplitRegression(data, indixes);

  if (temp.featureIndex === -1) {
    let leaf = new NodeRegression();
    leaf.isLeaf = true;
    leaf.predictValue = calculatePredictRegression(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  let node = new NodeRegression();
  node.featureIndex = temp.featureIndex;
  node.threshold = temp.threshold;
  node.id = nodeCounter++;

  let leftIndixes = [];
  let rightIndixes = [];

  for (const index of indixes) {
    if (data[index][node.featureIndex] <= node.threshold) {
      leftIndixes.push(index);
    } else {
      rightIndixes.push(index);
    }
  }

  node.left = buildTreeRegression(
    data,
    leftIndixes,
    maxDepth,
    currentDepth + 1
  );
  node.right = buildTreeRegression(
    data,
    rightIndixes,
    maxDepth,
    currentDepth + 1
  );

  return node;
}

function predictRegression(tree, example) {
  let current = tree;

  while (!current.isLeaf && current !== null) {
    if (example[current.featureIndex] <= current.threshold) {
      current = current.left;
    } else {
      current = current.right;
    }
  }

  return current ? current.predictValue : 0.0;
}

function deleteTreeRegression(node) {
  if (node === null) {
    return;
  }

  deleteTreeRegression(node.left);
  deleteTreeRegression(node.right);

  nodeCounter = 0;
}

function chooseOptimalDepthRegression(data) {
  const minDepth = 2;
  const maxDepth = data.length - 1;
  let bestDepth = minDepth;
  let bestMAE = Infinity;

  for (let depth = minDepth; depth <= maxDepth; depth++) {
    let testSize = Math.round(data.length * 0.6);
    let indixesTest = [];
    let indixesTraining = [];
    let indixes = Array.from({ length: data.length }, (_, i) => i);

    shuffleArray(indixes);

    for (let i = 0; i < testSize; ++i) {
      indixesTest.push(indixes[i]);
    }
    for (let i = testSize; i < data.length; ++i) {
      indixesTraining.push(indixes[i]);
    }

    let root = buildTreeRegression(data, indixesTraining, depth);
    let mae = 0.0;

    for (let i = 0; i < indixesTest.length; i++) {
      const index = indixesTest[i];
      let features = data[index].slice(0, data[index].length - 1);
      let prediction = predictRegression(root, features);
      mae += Math.abs(prediction - data[index][data[index].length - 1]);
    }

    mae /= indixesTest.length;

    if (mae < bestMAE) {
      bestMAE = mae;
      bestDepth = depth;
    }

    deleteTreeRegression(root);
  }

  return bestDepth;
}

class NodeClassification {
  constructor() {
    this.featureIndex = -1;
    this.threshold = 0.0;
    this.left = null;
    this.right = null;
    this.isLeaf = false;
    this.predictedClass = 0.0;
    this.id = -1;
  }
}

function calculateEntropy(data, indixes) {
  if (indixes.length === 0) {
    return 0.0;
  }

  let classCounts = {};

  for (const index of indixes) {
    let classValue = data[index][data[0].length - 1];
    classCounts[classValue] = (classCounts[classValue] || 0) + 1;
  }

  let entropy = 0.0;
  const total = indixes.length;

  for (const classValue in classCounts) {
    const count = classCounts[classValue];
    let probability = count / total;

    if (probability > 0.0) {
      entropy -= probability * Math.log2(probability);
    }
  }

  return entropy;
}

function calculateIGClassification(
  entropyParent,
  nLeft,
  entropyLeft,
  nRight,
  entropyRight,
  nParent
) {
  const informationGain =
    entropyParent -
    (nLeft / nParent) * entropyLeft -
    (nRight / nParent) * entropyRight;

  return informationGain;
}

function chooseTheBestSplitClassification(data, indixes) {
  let bestNode = new NodeClassification();
  bestNode.featureIndex = -1;
  let bestIG = -Infinity;
  const entropyParent = calculateEntropy(data, indixes);

  if (entropyParent === 0) {
    return bestNode;
  }

  const numberFeatures = data[0].length - 1;

  for (let featureIndex = 0; featureIndex < numberFeatures; featureIndex++) {
    let uniqueValues = new Set();
    for (const index of indixes) {
      uniqueValues.add(data[index][featureIndex]);
    }

    for (const threshold of uniqueValues) {
      let leftIndixes = [];
      let rightIndixes = [];

      for (const index of indixes) {
        if (data[index][featureIndex] <= threshold) {
          leftIndixes.push(index);
        } else {
          rightIndixes.push(index);
        }
      }

      const entropyLeft = calculateEntropy(data, leftIndixes);
      const entropyRight = calculateEntropy(data, rightIndixes);
      const informationGain = calculateIGClassification(
        entropyParent,
        leftIndixes.length,
        entropyLeft,
        rightIndixes.length,
        entropyRight,
        indixes.length
      );

      if (informationGain > bestIG) {
        bestIG = informationGain;
        bestNode.featureIndex = featureIndex;
        bestNode.threshold = threshold;
      }
    }
  }

  return bestNode;
}

function calculatePredictClassification(data, indixes) {
  if (indixes.length === 0) {
    return 0.0;
  }

  let classCounts = {};
  for (const index of indixes) {
    let classValue = data[index][data[0].length - 1];
    classCounts[classValue] = (classCounts[classValue] || 0) + 1;
  }

  let predictedClass = 0.0;
  let maxCount = 0;

  for (const classValue in classCounts) {
    const count = classCounts[classValue];
    if (count > maxCount) {
      maxCount = count;
      predictedClass = parseFloat(classValue);
    }
  }

  return predictedClass;
}

function predictClassification(tree, example) {
  let current = tree;

  while (current && !current.isLeaf) {
    if (example[current.featureIndex] <= current.threshold) {
      current = current.left;
    } else {
      current = current.right;
    }
  }

  return current ? current.predictedClass : 0.0;
}

function buildTreeClassification(
  data,
  indixes,
  maxDepth,
  currentDepth = 0,
  minSamplesSplit = 2
) {
  if (indixes.length === 0) {
    return null;
  }

  if (indixes.length < minSamplesSplit) {
    let leaf = new NodeClassification();
    leaf.isLeaf = true;
    leaf.predictedClass = calculatePredictClassification(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  if (currentDepth >= maxDepth) {
    let leaf = new NodeClassification();
    leaf.isLeaf = true;
    leaf.predictedClass = calculatePredictClassification(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  let bestNode = chooseTheBestSplitClassification(data, indixes);

  if (bestNode.featureIndex === -1) {
    let leaf = new NodeClassification();
    leaf.isLeaf = true;
    leaf.predictedClass = calculatePredictClassification(data, indixes);
    leaf.id = nodeCounter++;

    return leaf;
  }

  let node = new NodeClassification();
  node.featureIndex = bestNode.featureIndex;
  node.threshold = bestNode.threshold;
  node.id = nodeCounter++;

  let leftIndixes = [];
  let rightIndixes = [];

  for (const index of indixes) {
    if (data[index][node.featureIndex] <= node.threshold) {
      leftIndixes.push(index);
    }
    else {
      rightIndixes.push(index);
    }
  }

  node.left = buildTreeClassification(
    data,
    leftIndixes,
    maxDepth,
    currentDepth + 1,
    minSamplesSplit
  );
  node.right = buildTreeClassification(
    data,
    rightIndixes,
    maxDepth,
    currentDepth + 1,
    minSamplesSplit
  );

  return node;
}

function deleteTreeClassification(node) {
  if (node === null) {
    return;
  }

  deleteTreeClassification(node.left);
  deleteTreeClassification(node.right);

  nodeCounter = 0;
}

function chooseOptimalDepthClassification(data) {
  const minDepth = 2;
  const maxDepth = data.length - 1;
  let averageDepth = new Set();
  let bestDepth = minDepth;
  let accuracy = 0;

  for (let depth = minDepth; depth <= maxDepth; depth++) {
    const testSize = Math.round(data.length * 0.6);
    const indixesTest = [];
    const indixesTraining = [];
    const indixes = Array.from({ length: data.length }, (_, i) => i);

    shuffleArray(indixes);

    for (let i = 0; i < testSize; ++i) {
      indixesTest.push(indixes[i]);
    }
    for (let i = testSize; i < data.length; ++i) {
      indixesTraining.push(indixes[i]);
    }

    const root = buildTreeClassification(data, indixesTraining, depth);
    let rightAnswers = 0;

    if (indixesTest.length > 0) {
      for (let i = 0; i < indixesTest.length; ++i) {
        const index = indixesTest[i];
        let check = data[index];
        let features = check.slice(0, check.length - 1);

        const prediction = predictClassification(root, features);
        if (prediction === data[index][data[index].length - 1]) {
          rightAnswers++;
        }
      }

      const tempAccuracy = rightAnswers / indixesTest.length;

      if (accuracy <= tempAccuracy) {
        accuracy = tempAccuracy;
        bestDepth = depth;
      }
    }

    averageDepth.add(bestDepth);

    deleteTreeClassification(root);
  }

  let averageDepthArray = Array.from(averageDepth).sort((a, b) => a - b);
  const middleIndex = Math.floor(averageDepthArray.length / 2);
  const optimalDepth = averageDepthArray[middleIndex];

  return optimalDepth;
}

function treeToJsonRegression(node) {
  if (!node) return null;

  const jsonNode = {
    id: node.id,
    name: node.isLeaf
      ? `Predict: ${node.predictValue.toFixed(2)}`
      : `Feature ${node.featureIndex} ≤ ${node.threshold.toFixed(2)}`,
    featureIndex: node.featureIndex,
    threshold: node.threshold,
    isLeaf: node.isLeaf,
    predictValue: node.predictValue,
    value: node.predictValue
  };

  if (!node.isLeaf) {
    jsonNode.children = [
      treeToJsonRegression(node.left),
      treeToJsonRegression(node.right)
    ];
  }

  return jsonNode;
}

function treeToJsonClassification(node) {
  if (!node) return null;

  const jsonNode = {
    id: node.id,
    name: node.isLeaf
      ? `Class: ${node.predictedClass}`
      : `Feature ${node.featureIndex} ≤ ${node.threshold.toFixed(2)}`,
    featureIndex: node.featureIndex,
    threshold: node.threshold,
    isLeaf: node.isLeaf,
    predictedClass: node.predictedClass,
    value: node.predictedClass
  };

  if (!node.isLeaf) {
    jsonNode.children = [
      treeToJsonClassification(node.left),
      treeToJsonClassification(node.right)
    ];
  }

  return jsonNode;
}