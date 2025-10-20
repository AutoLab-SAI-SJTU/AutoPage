// Global variables
let rawData;
let chart = null;
let lineGraph = null;
let taskScores_save = null;
let currentTask = 'Avg';
const sortby_options = {
    BY_REWARD_SCORE: "sort-by-reward-score",
    BY_SUCCESS_RATE: "sort-by-success-rate",
    BY_GROUNDING_ACC: "sort-by-grounding-acc",
};
let cur_sortby_option = sortby_options.BY_REWARD_SCORE;

const taskSubtaskMapping = {
    'Avg': ['AlfWorld', 'ScienceWorld', 'BabyAI', 'PDDL', 'Jericho', 'WebShop', 'WebArena', 'Tool-Query', 'Tool-Operation'],
    'Embodied': ['AlfWorld', 'ScienceWorld', 'BabyAI'],
    'Game': ['PDDL', 'Jericho'],
    'Web': ['WebShop', 'WebArena'],
    'Tools': ['Tool-Query', 'Tool-Operation'],
};

const SubtaskNameMapping = {
    'AlfWorld': 'ALF',
    'ScienceWorld': 'SW',
    'BabyAI': 'BA',
    'PDDL': 'PL',
    'Jericho': 'JC',
    'WebShop': 'WS',
    'WebArena': 'WA',
    'Tool-Query': 'T-Q',
    'Tool-Operation': 'T-O',
};


const modelColors = {};
const borderStyles = {};


const colors = [
    'rgba(255, 99, 132, 1)',
    'rgba(54, 162, 235, 1)',
    'rgba(255, 206, 86, 1)',
    'rgba(75, 192, 192, 1)',
    'rgba(153, 102, 255, 1)',
    'rgba(255, 159, 64, 1)',
    'rgba(199, 199, 199, 1)',
    'rgba(83, 102, 255, 1)',
    'rgba(40, 159, 64, 1)',
    'rgba(143, 162, 235, 1)',
    'rgba(255, 99, 75, 1)',
    'rgba(71 ,150 ,87, 1)',
    'rgba(210 ,102 ,95, 1)',
    'rgba(51 ,47 ,180, 1)',
];


const borders = [
    {
        borderWidth: 2,
        borderDash: [],
    },
    {
        borderWidth: 2,
        borderDash: [5, 5],
    },
    {
        borderWidth: 2,
        borderDash: [10, 5],
    },
    {
        borderWidth: 2,
        borderDash: [2, 2],
    },
    {
        borderWidth: 2,
        borderDash: [8, 4],
    },
    {
        borderWidth: 2,
        borderDash: [5, 10],
    },
    {
        borderWidth: 2,
        borderDash: [15, 5],
    },
    {
        borderWidth: 2,
        borderDash: [5, 15],
    },
    {
        borderWidth: 2,
        borderDash: [10, 10],
    },
    {
        borderWidth: 2,
        borderDash: [11, 4],
    },
    {
        borderWidth: 2,
        borderDash: [10, 2],
    },
    {
        borderWidth: 2,
        borderDash: [4, 8],
    }, {
        borderWidth: 2,
        borderDash: [3, 5],
    }, {
        borderWidth: 2,
        borderDash: [5, 3],
    },
];


function generateModelColorsAndStyles(models) {
    for (let i = 0; i < models.length; i++) {
        const model = models[i];
        modelColors[model] = colors[i % colors.length];
        borderStyles[model] = borders[i % borders.length];
    }
}

function getScoresForTask(rawData, task) {
    console.log("Current task:", task);

    return rawData.map(model => {
        if (model.tasks[task]) {
            return {
                model: model.model,
                score: parseFloat(model.tasks[task].score),
                accuracy: parseFloat(model.tasks[task].accuracy),
                grounding: parseFloat(model.tasks[task].grounding)
            };
        } else {
            console.error("Task not found:", task);
            return null;
        }
    }).filter(item => item !== null);
}


let selectedModelIndexInBar = null;
let selectedModelIndexInLine = null;


function createMainResultChart() {
    const taskScores = getScoresForTask(rawData, currentTask);

    if (cur_sortby_option === sortby_options.BY_REWARD_SCORE) {
        taskScores.sort((a, b) => b.score - a.score);
    } else if (cur_sortby_option === sortby_options.BY_SUCCESS_RATE) {
        taskScores.sort((a, b) => b.accuracy - a.accuracy);
    } else if (cur_sortby_option === sortby_options.BY_GROUNDING_ACC) {
        taskScores.sort((a, b) => b.grounding - a.grounding);
    }
    taskScores_save = taskScores

    const labels = taskScores.map(item => item.model);
    const scores = taskScores.map(item => item.score);
    const accuracies = taskScores.map(item => item.accuracy);
    const groundings = taskScores.map(item => item.grounding);

    if (chart) {
        chart.destroy();
    }

    const ctx = document.getElementById('chart-success-reward-rate');
    chart = new Chart(ctx, {
        plugins: [ChartDataLabels],
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Progress Rate',
                    data: scores,
                    backgroundColor: '#f398ae',
                },
                {
                    label: 'Success Rate',
                    data: accuracies,
                    backgroundColor: '#78b5f1',
                },
                {
                    label: 'Grounding Accuracy',
                    data: groundings,
                    backgroundColor: '#f3e276',
                }

            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
            },
            indexAxis: 'y',
            scales: {
                x: {
                    ticks: {
                        beginAtZero: true,
                        min: 0,
                        max: 100,
                        font: {
                            size: 12,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    },
                    grace: 10,
                    title: {
                        display: true,
                        text: 'Value (%)',
                        font: {
                            size: 14,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    ticks: {
                        font: {
                            size: 12,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Model',
                        font: {
                            size: 14,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        usePointStyle: true,
                        font: {
                            size: 10,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    },
                    align: 'center',
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.formattedValue;
                            return label;
                        }
                    }
                },
                datalabels: {
                    align: 'end',
                    anchor: 'end',
                    color: 'black',
                    padding: 0,
                    formatter: function (value, context) {
                        return value.toFixed(2);
                    },
                    font: function (context) {
                        var width = context.chart.width;
                        var size = Math.round(width / 48);
                        size = Math.min(size, 10);
                        return {
                            size: size,
                            family: "'Noto Sans', sans-serif",
                        };
                    }
                }
            }
        }
    });

    const subTaskLabels = taskSubtaskMapping[currentTask] || [];

    if (lineGraph) {
        lineGraph.destroy();
    }
    let datasets = [];
    let yAxisTitle = '';
    const lineGraphCtx = document.getElementById('line-graph').getContext('2d');
    if (cur_sortby_option === sortby_options.BY_SUCCESS_RATE) {
        datasets = rawData.map(modelData => {
            return {
                label: modelData.model,
                data: subTaskLabels.map(subtask => modelData.tasks[subtask] ?
                    parseFloat(modelData.tasks[subtask].accuracy) : null),
                borderColor: modelColors[modelData.model] || '#4CAF50',
                fill: false,
                ...borderStyles[modelData.model]
            };
        });
        yAxisTitle = 'Success Rate (%)';
    } else if (cur_sortby_option === sortby_options.BY_REWARD_SCORE) {
        datasets = rawData.map(modelData => {
            return {
                label: modelData.model,
                data: subTaskLabels.map(subtask => modelData.tasks[subtask] ?
                    parseFloat(modelData.tasks[subtask].score) : null),
                borderColor: modelColors[modelData.model] || '#4CAF50',
                fill: false,
                ...borderStyles[modelData.model]
            };
        });
        yAxisTitle = 'Progress Rate (%)';
    } else if (cur_sortby_option === sortby_options.BY_GROUNDING_ACC) {
        datasets = rawData.map(modelData => {
            return {
                label: modelData.model,
                data: subTaskLabels.map(subtask => modelData.tasks[subtask] ?
                    parseFloat(modelData.tasks[subtask].grounding) : null),
                borderColor: modelColors[modelData.model] || '#4CAF50',
                fill: false,
                ...borderStyles[modelData.model]
            };
        });
        yAxisTitle = 'Grounding accuracy (%)';
    }
    lineGraph = new Chart(lineGraphCtx, {
        type: 'line',
        data: {
            labels: subTaskLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: {
                        font: {
                            size: 12,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Sub-Task',
                        font: {
                            size: 14,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        },
                    },
                },
                y: {
                    ticks: {
                        font: {
                            size: 12,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        }
                    },
                    title: {
                        display: true,
                        text: yAxisTitle,
                        font: {
                            size: 14,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        },
                    },
                    min: 0,
                    max: 100,
                },
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        usePointStyle: true,
                        font: {
                            size: 10,
                            family: "'Noto Sans', sans-serif",
                            weight: 'bold'
                        },
                    },
                    align: 'center',
                    position: 'bottom',
                },
                annotation: {
                    annotations: []
                }
            }
        }
    });

    if (ctx) {
        ctx.addEventListener('mousemove', function (event) {
            const activePoints = chart.getElementsAtEventForMode(event, 'nearest', {intersect: true}, true);

            if (activePoints.length > 0) {
                const selectedIndex = activePoints[0].index;
                if (selectedIndex !== selectedModelIndexInBar) {
                    selectedModelIndexInBar = selectedIndex;
                    selectedModelIndexInLine = highlightModel(taskScores_save.map(item => item.model), selectedModelIndexInBar);
                    // updateLineGraphScale(taskScores_save, selectedModelIndexInLine);
                }
            } else {
                selectedModelIndexInBar = null;
                removeHighlight();
                // resetLineGraphScale();
            }
        });
    }
}

document.querySelectorAll('.btn-group.task-filter-selector .btn').forEach(btn => {
    btn.addEventListener('click', () => {
        currentTask = btn.id.replace('filter-by-', '');

        document.querySelectorAll('.btn-group.task-filter-selector .btn.active').forEach(active => {
            active.classList.remove('active');
        });
        btn.classList.add('active');

        createMainResultChart();
    });
});

function highlightModel(labels, index) {
    let highlightedDatasetIndex = -1;
    const modelName = labels[index];

    lineGraph.data.datasets.forEach((dataset, datasetIndex) => {
        if (dataset.label === modelName) {
            dataset.borderColor = '#000000';
            dataset.borderWidth = 4;
            highlightedDatasetIndex = datasetIndex
        } else {
            dataset.borderColor = modelColors[dataset.label];
            dataset.borderWidth = 2;
        }
    });
    lineGraph.options.plugins.annotation.annotations = createAnnotations(modelName);
    lineGraph.update();

    return highlightedDatasetIndex
}

function removeHighlight() {
    lineGraph.data.datasets.forEach((dataset) => {
        dataset.borderColor = modelColors[dataset.label];
        dataset.borderWidth = 2;
    });
    lineGraph.options.plugins.annotation.annotations = [];
    lineGraph.update();
}


function createAnnotations(modelName) {
    return taskSubtaskMapping[currentTask].map(subtask => {
        const modelData = rawData.find(data => data.model === modelName);
        if (!modelData || !modelData.tasks || !modelData.tasks[subtask]) {
            return null;
        }

        let content = `${modelData.model} (success rate):\n`;
        if (cur_sortby_option === sortby_options.BY_REWARD_SCORE) {
            content = `${modelData.model} (progress rate):\n`
        } else if (cur_sortby_option === sortby_options.BY_GROUNDING_ACC) {
            content = `${modelData.model} (grounding acc):\n`
        }
        let focus_score = null
        let avg_content = `${modelData.model}(%):\n`
        let minYValue = 100;
        let maxYValue = 0;
        taskSubtaskMapping[currentTask].forEach(subtask => {
            if (modelData.tasks && modelData.tasks[subtask] && cur_sortby_option === sortby_options.BY_SUCCESS_RATE) {
                focus_score = parseFloat(modelData.tasks[subtask].accuracy);
            } else if (modelData.tasks && modelData.tasks[subtask] && cur_sortby_option === sortby_options.BY_REWARD_SCORE) {
                focus_score = parseFloat(modelData.tasks[subtask].score);
            } else if (modelData.tasks && modelData.tasks[subtask] && cur_sortby_option === sortby_options.BY_GROUNDING_ACC) {
                focus_score = parseFloat(modelData.tasks[subtask].grounding);
            }
            minYValue = Math.min(minYValue, focus_score);
            maxYValue = Math.max(maxYValue, focus_score);
            if (maxYValue === 0) {
                maxYValue = 0.6
            }
            if (currentTask === 'Avg') {
                subtask = SubtaskNameMapping[subtask]
                content = avg_content + `${subtask}: ${focus_score.toFixed(1)}\n`;
                avg_content += `${subtask}: ${focus_score.toFixed(1)}\n`
            } else {
                content += `${subtask}: ${focus_score.toFixed(1)}%\n`;
            }

        });
        return {
            type: 'label',
            content: content,
            xValue: (taskSubtaskMapping[currentTask].length - 1) / 2,
            yValue: (maxYValue + minYValue) / 2,
            backgroundColor: 'rgba(255,255,255,0.8)',
            font: {
                size: 9,
                weight: 'bold',
                color: 'black',
                family: "'Noto Sans', sans-serif"
            },
            xPadding: 6,
            yPadding: 6,
            borderColor: 'black',
            borderWidth: 1,
            borderRadius: 6,
            position: 'center',
            adjustScaleRange: true
        };
    }).filter(annotation => annotation !== null);
}

function updateLineGraphScale(labels, selectedIndex) {
    // const modelName = [index];
    const dataset = lineGraph.data.datasets[selectedIndex];
    const maxValue = Math.max(...dataset.data);
    const minValue = Math.min(...dataset.data.filter(v => v !== null));
    const range = maxValue - minValue;
    let buffer = null;
    if (range > 2) {
        buffer = 0.5
    } else {
        buffer = 1.5;
    }

    lineGraph.options.scales.y.min = Math.max(0, minValue - buffer);
    lineGraph.options.scales.y.max = maxValue + buffer;
    lineGraph.update();
}

function resetLineGraphScale() {
    lineGraph.options.scales.y.min = 0;
    lineGraph.options.scales.y.max = 100;
    lineGraph.update();
}

Object.values(sortby_options).forEach(sortby_option => {
    const btn = document.getElementById(sortby_option);

    btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-group.metric-filter-selector .btn.active').forEach(active => {
            active.classList.remove('active');
        });

        btn.classList.add('active');

        cur_sortby_option = sortby_option;

        createMainResultChart();
    });
});

document.addEventListener('DOMContentLoaded', function () {
    fetch('/agentboard/data/To_Release/main_data_new.json').then(response => response.json()).then((loadedData) => {
        rawData = loadedData;
        generateModelColorsAndStyles(rawData.map(data => data.model));
        createMainResultChart();
    });
});
