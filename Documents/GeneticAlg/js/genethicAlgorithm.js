function addCity(x, y) {
  let newCity = {
    number: cities.length + 1,
    x: x,
    y: y
  };

  cities.push(newCity);
}

function calculateDistance(city1, city2) {
  return Math.sqrt(((city2.x - city1.x) * (city2.x - city1.x)) + ((city2.y - city1.y) * (city2.y - city1.y)));
}

function randomNum(start, end) {
  const range = (end - start) + 1;
  const randomInt = start + Math.floor(Math.random() * range);

  return randomInt;
}

function calculateFitness(distance, chromosome) {
  let summa = 0;
  for (let i = 0; i < chromosome.length - 1; i++) {
    summa += distance[chromosome[i]][chromosome[i + 1]];
  }

  return summa;
}

class Individual {
  constructor(chromosome, distance) {
    this.chromosome = chromosome;
    this.fitness = calculateFitness(distance, chromosome);
  }
}

function createRandomIndividual(cityCount, distance) {
  let chromosome = new Array(cityCount + 1);

  let cityIndixes = new Array(cityCount - 1);
  for (let i = 0; i < cityCount - 1; i++) {
    cityIndixes[i] = i + 1;
  }

  for (let i = cityIndixes.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cityIndixes[i], cityIndixes[j]] = [cityIndixes[j], cityIndixes[i]];
  }

  chromosome[0] = 0;

  for (let i = 0; i < cityCount - 1; ++i) {
    chromosome[i + 1] = cityIndixes[i];
  }

  chromosome[cityCount] = 0;

  return new Individual(chromosome, distance);
}

function createInitialPopulation(populationSize, cityCount, distance) {
  let population = [];

  for (let i = 0; i < populationSize; i++) {
    population.push(createRandomIndividual(cityCount, distance))
  }

  return population;
}

function mate(parent1, parent2, cityCount, distance) {
  let childChromosome = new Array(cityCount + 1);

  let crossoverPoint = randomNum(1, cityCount - 1);

  for (let i = 0; i <= crossoverPoint; i++) {
    childChromosome[i] = parent1.chromosome[i];
  }

  let currentIndex = crossoverPoint + 1;

  for (let i = 1; i < cityCount; i++) {
    let cityAlreadyPresent = false;
    for (let j = 0; j <= crossoverPoint; j++) {
      if (parent2.chromosome[i] === childChromosome[j]) {
        cityAlreadyPresent = true;
        break;
      }
    }

    if (!cityAlreadyPresent) {
      childChromosome[currentIndex] = parent2.chromosome[i];
      currentIndex++;
    }
  }

  for (let i = 1; i < cityCount; i++) {
    let cityAlreadyPresent = false;
    for (let j = 0; j < cityCount; j++) {
      if (parent1.chromosome[i] === childChromosome[j]) {
        cityAlreadyPresent = true;
        break;
      }
    }

    if (!cityAlreadyPresent && currentIndex <= cityCount - 1) {
      childChromosome[currentIndex] = parent1.chromosome[i];
      currentIndex++;
    }
  }

  childChromosome[cityCount] = 0;

  let mutationProbability = 0.2;
  let p = randomNum(0, 100) / 100.0;

  if (p < mutationProbability) {
    let index1 = randomNum(1, cityCount - 1);
    let index2 = randomNum(1, cityCount - 1);
    [childChromosome[index1], childChromosome[index2]] = [childChromosome[index2], childChromosome[index1]];
  }

  return new Individual(childChromosome, distance);
}

function compareIndividuals(individual1, individual2) {
  return individual1.fitness - individual2.fitness;
}

function drawRoute(chromosome, cities, ctx, transformCoordinates) {
  ctx.beginPath();
  ctx.strokeStyle = 'rgb(255, 240, 0)';
  ctx.lineWidth = 2;

  let transformed = transformCoordinates(cities[chromosome[0]].x, cities[chromosome[0]].y);
  ctx.moveTo(transformed.x, transformed.y);

  for (let i = 1; i < chromosome.length; i++) {
    if (chromosome[i] >= cities.length) {
      console.error("Неверный индекс города в хромосоме:", chromosome[i], "Длина массива городов:", cities.length);
      return;
    }

    transformed = transformCoordinates(cities[chromosome[i]].x, cities[chromosome[i]].y);
    ctx.lineTo(transformed.x, transformed.y);
  }

  ctx.stroke();
}