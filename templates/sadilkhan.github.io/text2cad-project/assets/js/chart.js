document.addEventListener("DOMContentLoaded", function () {
    let chartInstance = null;
    let animationDuration = 1500;
    let fontSize = 18;

    // Function to handle main tab selection
    function openTab(evt, tabName) {
        const tabcontent = document.getElementsByClassName("tabcontent");
        const tablinks = document.getElementsByClassName("tablinks");

        for (let i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        for (let i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";

        if (chartInstance) chartInstance.destroy();
    }

    // Function to handle sub-tab selection
    function openSubTab(evt, subTabName) {
        const subTabcontent = document.getElementsByClassName("sub-tabcontent");
        const subTablinks = document.getElementsByClassName("sub-tablinks");

        for (let i = 0; i < subTabcontent.length; i++) {
            subTabcontent[i].style.display = "none";
        }
        for (let i = 0; i < subTablinks.length; i++) {
            subTablinks[i].className = subTablinks[i].className.replace(" active", "");
        }
        document.getElementById(subTabName).style.display = "block";
        evt.currentTarget.className += " active";

        if (chartInstance) chartInstance.destroy();

        if (subTabName === 'F1Scores') createF1Chart();
        if (subTabName === 'CDandIR') createCDIRChart();
        if (subTabName === 'GPT4') createGPT4CombinedChart();
        if (subTabName === 'User') createUserCombinedChart();
    }

    // Trigger the first tab and sub-tab to be selected by default when the page loads
    function selectDefaultTabs() {
        const firstMainTab = document.querySelector(".tablinks"); // Select the first main tab
        const firstSubTab = document.querySelector(".sub-tablinks"); // Select the first sub-tab
        
        if (firstMainTab && firstSubTab) {
            // Simulate clicking the first main tab and first sub-tab
            openTab({ currentTarget: firstMainTab }, 'SequenceEvaluation');
            openSubTab({ currentTarget: firstSubTab }, 'F1Scores');
        }
    }

    function createF1Chart() {
        const ctx = document.getElementById('f1Chart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Line', 'Arc', 'Circle', 'Extrusion'],
                datasets: [
                    {
                        label: 'DeepCAD',
                        data: [76.78, 20.04, 65.14, 88.72],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Text2CAD',
                        data: [81.13, 36.03, 74.25, 93.31],
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: fontSize // Increase Y-axis font size here
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: fontSize // Increase X-axis font size here
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: 16 // Increase legend font size
                            }
                        }
                    }
                },
                animation: {
                    duration: animationDuration
                }
            }
            
        });
    }

    function createCDIRChart() {
        const ctx = document.getElementById('cdIrChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Median CD', 'Mean CD', 'IR'],
                datasets: [
                    {
                        label: 'DeepCAD',
                        data: [32.82, 97.93, 10.00],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Text2CAD',
                        data: [0.37, 26.41, 0.93],
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: fontSize // Increase Y-axis font size here
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: fontSize // Increase X-axis font size here
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: fontSize // Increase legend font size
                            }
                        }
                    }
                },
                animation: {
                    duration: animationDuration
                }
            }
            
        });
    }

    function createGPT4CombinedChart() {
        const ctx = document.getElementById('gpt4CombinedChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Abstract', 'Beginner', 'Intermediate', 'Expert'],
                datasets: [
                    {
                        label: 'DeepCAD',
                        data: [47.40, 51.15, 40.20, 36.06],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Undecided',
                        data: [0.80, 0.80, 0.80, 0.70],
                        backgroundColor: 'rgba(255, 206, 86, 0.8)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Text2CAD',
                        data: [51.80, 48.35, 58.80, 63.24],
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: fontSize // Increase Y-axis font size here
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: fontSize // Increase X-axis font size here
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: fontSize // Increase legend font size
                            }
                        }
                    }
                },
                animation: {
                    duration: animationDuration
                }
            }
            
        });
    }

    function createUserCombinedChart() {
        const ctx = document.getElementById('userCombinedChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Abstract', 'Beginner', 'Intermediate', 'Expert'],
                datasets: [
                    {
                        label: 'DeepCAD',
                        data: [50.95, 48.73, 44.94, 41.14],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Undecided',
                        data: [0.80, 0.80, 0.80, 0.70],
                        backgroundColor: 'rgba(255, 206, 86, 0.8)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Text2CAD',
                        data: [49.05, 51.27, 55.06, 58.86],
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: fontSize // Increase Y-axis font size here
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: fontSize // Increase X-axis font size here
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: fontSize // Increase legend font size
                            }
                        }
                    }
                },
                animation: {
                    duration: animationDuration
                }
            }
            
        });
    }
    // Run the selectDefaultTabs function when the DOM is fully loaded
    selectDefaultTabs();
    window.openTab = openTab;
    window.openSubTab = openSubTab;
});
