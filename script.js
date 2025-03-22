document.addEventListener("DOMContentLoaded", function () {
    showSection('simple'); // Default to Simple Analysis
    showTab('search'); // Default to Search tab
    
    // Add event listeners for section buttons
    document.querySelectorAll('nav button').forEach(button => {
        button.addEventListener('click', function() {
            const section = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            showSection(section);
        });
    });
    
    // Add event listeners for tab buttons
    document.querySelectorAll('.tab-buttons button').forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            showTab(tab);
        });
    });
    
    // Add event listeners for advanced tab buttons
    document.querySelectorAll('.advanced-tab-buttons button').forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            showAdvancedTab(tab);
        });
    });
    
    // Add pulse animation to buttons
    animateButtons();
});

function showSection(section) {
    // First fade out all sections
    document.querySelectorAll('.content-section.active').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            el.classList.remove('active');
            
            // Then fade in the selected section
            let targetSection = document.getElementById(section);
            targetSection.classList.add('active');
            
            // Small delay before animation starts
            setTimeout(() => {
                targetSection.style.opacity = '1';
                targetSection.style.transform = 'translateY(0)';
            }, 50);
            
        }, 300); // Match this to your CSS transition duration
    });
    
    // If no sections are currently active, immediately show the selected one
    if (document.querySelectorAll('.content-section.active').length === 0) {
        let targetSection = document.getElementById(section);
        targetSection.classList.add('active');
        
        setTimeout(() => {
            targetSection.style.opacity = '1';
            targetSection.style.transform = 'translateY(0)';
        }, 50);
    }
}

function showTab(tab) {
    // First hide all tabs
    document.querySelectorAll('.tab-content.active').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            el.classList.remove('active');
            
            // Then show the selected tab
            let targetTab = document.getElementById(tab);
            targetTab.classList.add('active');
            
            // Small delay before animation starts
            setTimeout(() => {
                targetTab.style.opacity = '1';
                targetTab.style.transform = 'translateY(0)';
            }, 50);
            
        }, 300); // Match this to your CSS transition duration
    });
    
    // If no tabs are currently active, immediately show the selected one
    if (document.querySelectorAll('.tab-content.active').length === 0) {
        let targetTab = document.getElementById(tab);
        targetTab.classList.add('active');
        
        setTimeout(() => {
            targetTab.style.opacity = '1';
            targetTab.style.transform = 'translateY(0)';
        }, 50);
    }
    
    // Highlight the active tab button
    document.querySelectorAll('.tab-buttons button').forEach(button => {
        if (button.getAttribute('onclick').includes(tab)) {
            button.style.background = 'var(--success-dark)';
        } else {
            button.style.background = 'var(--success)';
        }
    });
}

function showAdvancedTab(tabId) {
    // Hide all advanced tabs with animation
    document.querySelectorAll('.advanced-tab-content').forEach(el => {
        if (el.id !== tabId && el.style.display !== 'none') {
            el.style.opacity = '0';
            el.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                el.style.display = 'none';
                
                // Show the selected tab with animation
                let targetTab = document.getElementById(tabId);
                targetTab.style.display = 'block';
                targetTab.style.opacity = '0';
                targetTab.style.transform = 'scale(0.95)';
                
                // Small delay before fade in
                setTimeout(() => {
                    targetTab.style.opacity = '1';
                    targetTab.style.transform = 'scale(1)';
                }, 50);
                
            }, 300); // Match this to your CSS transition duration
        }
    });
    
    // If the tab isn't already displayed, show it
    if (document.getElementById(tabId).style.display === 'none') {
        document.getElementById(tabId).style.display = 'block';
        
        setTimeout(() => {
            document.getElementById(tabId).style.opacity = '1';
            document.getElementById(tabId).style.transform = 'scale(1)';
        }, 50);
    }
    
    // Highlight the active advanced tab button
    document.querySelectorAll('.advanced-tab-buttons button').forEach(button => {
        if (button.getAttribute('onclick').includes(tabId)) {
            button.style.background = 'var(--primary)';
            button.style.color = 'white';
        } else {
            button.style.background = 'var(--light)';
            button.style.color = 'var(--dark)';
        }
    });
}

function animateButtons() {
    // Add a subtle pulse animation to main action buttons
    const actionButtons = document.querySelectorAll('button:not(nav button):not(.tab-buttons button):not(.advanced-tab-buttons button)');
    
    actionButtons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.animate([
                { transform: 'scale(1)' },
                { transform: 'scale(1.05)' },
                { transform: 'scale(1)' }
            ], {
                duration: 400,
                iterations: 1
            });
        });
    });
}

function searchHousehold() {
    let hh_id = document.getElementById("hh_id").value;
    if (!hh_id) {
        showNotification("Please enter a valid HCES ID", "error");
        return;
    }
    
    // Add loading indicator
    const resultsContainer = document.getElementById("search-results");
    resultsContainer.innerHTML = '<div class="loading-spinner"></div>';
    resultsContainer.style.opacity = '1';

    fetch("http://127.0.0.1:5500/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ HH_ID: hh_id })
    })
    .then(response => response.json())
    .then(data => {
        // Hide results first for animation
        resultsContainer.style.opacity = '0';
        resultsContainer.style.transform = 'scale(0.98)';
        
        setTimeout(() => {
            if (data.error) {
                resultsContainer.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
            } else {
                resultsContainer.innerHTML = generateVerticalData(data[0]);
            }
            
            // Show results with animation
            setTimeout(() => {
                resultsContainer.style.opacity = '1';
                resultsContainer.style.transform = 'scale(1)';
            }, 50);
            
        }, 300);
    })
    .catch(error => {
        console.error("Error fetching household data:", error);
        showNotification("Error connecting to the server. Please try again.", "error");
    });
}

function generateVerticalData(data) {
    let html = `<table>`;
    
    Object.keys(data).forEach(key => {
        let value = data[key] !== null ? (data[key] == 1 ? 'Yes' : data[key] == 0.0 ? 'No' : data[key]) : '';
        html += `<tr><th>${key.replace(/_/g, ' ')}</th><td>${value}</td></tr>`;
    });
    
    html += `</table>`;
    return html;
}

// Global array to store person-level entries
let personEntries = [];

// Mapping objects for dropdown values
const relationMapping = {
  1: "Self",
  2: "Spouse of head",
  3: "Married child",
  4: "Spouse of married child",
  5: "Unmarried child",
  6: "Grandchild",
  7: "Father/Mother/Father-in-law/Mother-in-law",
  8: "Brother/Sister/Brother-in-law/Sister-in-law/Other relatives",
  9: "Servants/Employees/Other non-relatives"
};

const maritalMapping = {
  1: "Never married",
  2: "Currently married",
  3: "Widowed",
  4: "Divorced/Separated"
};

const educationMapping = {
  "01": "Not literate",
  "02": "Literate with non-formal education",
  "03": "Literate with formal education or below primary",
  "04": "Primary",
  "05": "Upper primary/middle",
  "06": "Secondary",
  "07": "Higher secondary",
  "08": "Diploma/certificate course (up to secondary)",
  "10": "Diploma/certificate course (higher secondary)",
  "11": "Diploma/certificate course (graduation & above)",
  "12": "Graduate",
  "13": "Post graduate and above"
};

// Clear person-level input fields
function clearPersonInputs() {
  document.getElementById("person_serial").value = "";
  document.getElementById("person_relation").value = "";
  document.getElementById("person_gender").value = "";
  document.getElementById("person_age").value = "";
  document.getElementById("person_marital").value = "";
  document.getElementById("person_education_level").value = "";
  document.getElementById("person_education_years").value = "";
  document.getElementById("person_internet").value = "";
  document.getElementById("person_days_away").value = "";
  document.getElementById("person_meals_usual").value = "";
  document.getElementById("person_meals_school").value = "";
  document.getElementById("person_meals_employer").value = "";
  document.getElementById("person_meals_others").value = "";
  document.getElementById("person_meals_payment").value = "";
  document.getElementById("person_meals_home").value = "";
}

// Render the person table from personEntries array
function renderPersonTable() {
  const tbody = document.getElementById("personTable").querySelector("tbody");
  tbody.innerHTML = "";
  personEntries.forEach((entry, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${entry.serial}</td>
      <td>${relationMapping[entry.relation] || entry.relation}</td>
      <td>${entry.gender == 1 ? "Male" : entry.gender == 2 ? "Female" : "Other"}</td>
      <td>${entry.age}</td>
      <td>${maritalMapping[entry.marital] || entry.marital}</td>
      <td>${educationMapping[entry.education_level] || entry.education_level}</td>
      <td>${entry.education_years}</td>
      <td>${entry.internet == 1 ? "Yes" : "No"}</td>
      <td>${entry.days_away}</td>
      <td>${entry.meals_usual}</td>
      <td>${entry.meals_school}</td>
      <td>${entry.meals_employer}</td>
      <td>${entry.meals_others}</td>
      <td>${entry.meals_payment}</td>
      <td>${entry.meals_home}</td>
      <td>
        <button onclick="editPerson(${index})">Edit</button>
        <button onclick="removePerson(${index})">Remove</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Function to add a person entry
function addPerson() {
  const serial = document.getElementById("person_serial").value.trim();
  const relation = document.getElementById("person_relation").value;
  const gender = document.getElementById("person_gender").value;
  const age = document.getElementById("person_age").value.trim();
  const marital = document.getElementById("person_marital").value;
  const education_level = document.getElementById("person_education_level").value;
  const education_years = document.getElementById("person_education_years").value.trim();
  const internet = document.getElementById("person_internet").value;
  const days_away = document.getElementById("person_days_away").value.trim();
  const meals_usual = document.getElementById("person_meals_usual").value.trim();
  const meals_school = document.getElementById("person_meals_school").value.trim();
  const meals_employer = document.getElementById("person_meals_employer").value.trim();
  const meals_others = document.getElementById("person_meals_others").value.trim();
  const meals_payment = document.getElementById("person_meals_payment").value.trim();
  const meals_home = document.getElementById("person_meals_home").value.trim();

  if (!serial || !relation || !gender || !age || !marital || !education_level ||
      !education_years || !internet || !days_away || !meals_usual) {
    showNotification("Please fill in all required person fields", "error");
    return;
  }

  const person = {
    serial: serial,
    relation: Number(relation),
    gender: Number(gender),
    age: Number(age),
    marital: Number(marital),
    education_level: education_level,
    education_years: Number(education_years),
    internet: Number(internet),
    days_away: Number(days_away),
    meals_usual: Number(meals_usual),
    meals_school: Number(meals_school),
    meals_employer: Number(meals_employer),
    meals_others: Number(meals_others),
    meals_payment: Number(meals_payment),
    meals_home: Number(meals_home)
  };

  personEntries.push(person);
  renderPersonTable();
  clearPersonInputs();
}

// Function to remove a person entry
function removePerson(index) {
  personEntries.splice(index, 1);
  renderPersonTable();
}

// Function to edit a person entry
function editPerson(index) {
  const entry = personEntries[index];
  document.getElementById("person_serial").value = entry.serial;
  document.getElementById("person_relation").value = entry.relation;
  document.getElementById("person_gender").value = entry.gender;
  document.getElementById("person_age").value = entry.age;
  document.getElementById("person_marital").value = entry.marital;
  document.getElementById("person_education_level").value = entry.education_level;
  document.getElementById("person_education_years").value = entry.education_years;
  document.getElementById("person_internet").value = entry.internet;
  document.getElementById("person_days_away").value = entry.days_away;
  document.getElementById("person_meals_usual").value = entry.meals_usual;
  document.getElementById("person_meals_school").value = entry.meals_school;
  document.getElementById("person_meals_employer").value = entry.meals_employer;
  document.getElementById("person_meals_others").value = entry.meals_others;
  document.getElementById("person_meals_payment").value = entry.meals_payment;
  document.getElementById("person_meals_home").value = entry.meals_home;

  personEntries.splice(index, 1);
  renderPersonTable();
}

function predictExpense() {
    const resultContainer = document.getElementById("prediction-results");
    resultContainer.innerHTML = '<div class="loading-spinner"></div>';
    resultContainer.style.opacity = '1';

    const getCheckboxValue = (id) => document.getElementById(id).checked ? 1 : 0;

    // Collect all data from the form
    let data = {
        "Sector": document.getElementById("sector").value,
        "State": document.getElementById("state-select").value,
        "NSS Region": document.getElementById("nss_region").value,
        "District": document.getElementById("district").value,
        "Household Size": document.getElementById("household_size").value || "1",
        "NCO 3D": document.getElementById("nco_3d").value,
        "NIC 5D": document.getElementById("nic_5d").value,

        // Online Purchases
        "Clothing": getCheckboxValue("clothing"),
        "Footwear": getCheckboxValue("footwear"),
        "Furniture & Fixtures": getCheckboxValue("furniture"),
        "Mobile Handset": getCheckboxValue("mobile"),
        "Personal Goods": getCheckboxValue("personal_goods"),
        "Recreation Goods": getCheckboxValue("recreation"),
        "Household Appliances": getCheckboxValue("household_appliances"),
        "Crockery & Utensils": getCheckboxValue("crockery"),
        "Sports Goods": getCheckboxValue("sports_goods"),
        "Medical Equipment": getCheckboxValue("medical"),
        "Bedding": getCheckboxValue("bedding"),

        // Household Possessions
        "Television": getCheckboxValue("television"),
        "Radio": getCheckboxValue("radio"),
        "Laptop/PC": getCheckboxValue("laptop"),
        "Bicycle": getCheckboxValue("bicycle"),
        "Motorcycle/Scooter": getCheckboxValue("motorcycle"),
        "Motorcar/Jeep/Van": getCheckboxValue("motorcar"),
        "Trucks": getCheckboxValue("trucks"),
        "Animal Cart": getCheckboxValue("animal_cart"),
        "Refrigerator": getCheckboxValue("refrigerator"),
        "Washing Machine": getCheckboxValue("washing_machine"),
        "Air Conditioner": getCheckboxValue("airconditioner"),
        "entry" : personEntries
    };

    fetch("http://127.0.0.1:5500/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        resultContainer.style.opacity = '0';
        resultContainer.style.transform = 'scale(0.98)';
        
        setTimeout(() => {
            resultContainer.innerHTML = `
                <div class="result-card">
                    <div class="result-item">
                        <h3>Total Expense</h3>
                        <div class="amount">₹${data["Total Expense"].toLocaleString()}</div>
                    </div>
                    <div class="result-item">
                        <h3>MPCE</h3>
                        <div class="amount">₹${data["MPCE"].toLocaleString()}</div>
                    </div>
                </div>
            `;
            setTimeout(() => {
                resultContainer.style.opacity = '1';
                resultContainer.style.transform = 'scale(1)';
                animateCountUp(resultContainer.querySelectorAll('.amount'));
            }, 50);
        }, 300);
    })
    .catch(error => {
        console.error("Error predicting expense:", error);
        showNotification("Error connecting to the server. Please try again.", "error");
    });
}


function animateCountUp(elements) {
    elements.forEach(element => {
        const finalValue = parseInt(element.textContent.replace(/[^\d]/g, ''));
        let startValue = 0;
        const duration = 1500;
        const increment = Math.ceil(finalValue / (duration / 20));
        
        const timer = setInterval(() => {
            startValue += increment;
            if (startValue >= finalValue) {
                clearInterval(timer);
                startValue = finalValue;
            }
            element.textContent = '₹' + startValue.toLocaleString();
        }, 20);
    });
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to body
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateY(0)';
        notification.style.opacity = '1';
    }, 10);
    
    // Animate out after delay
    setTimeout(() => {
        notification.style.transform = 'translateY(-20px)';
        notification.style.opacity = '0';
        
        // Remove from DOM after animation
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

function resetChartsAndData() {
    if (window.pieChart) {
        window.pieChart.destroy();
        window.pieChart = null;
    }
    if (window.barChart) {
        window.barChart.destroy();
        window.barChart = null;
    }
    document.getElementById("average-info").innerHTML = "";
}

function fetchDistribution() {
    const state = document.getElementById("state-select").value;
    const sector = document.getElementById("state-sector").value;
    const finalSector = sector === "Both" ? "" : sector;

    fetch("http://127.0.0.1:5500/analyze_state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, sector: finalSector })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showNotification(data.error, "error");
        } else {
            drawExpenseCharts(data.local_distribution, data.india_distribution, data.avg_local, data.avg_india, "state");
        }
    });
}

function submitDistrictAnalysis() {
    const district = document.getElementById("district-input").value;
    const sector = document.getElementById("district-sector").value;
    const finalSector = sector === "Both" ? "" : sector;

    fetch("http://127.0.0.1:5500/analyze_district", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ district, sector: finalSector })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showNotification(data.error, "error");
        } else {
            drawExpenseCharts(data.local_distribution, data.india_distribution, data.avg_local, data.avg_india, "district");
        }
    });
}

function submitNssAnalysis() {
    const nss_region = document.getElementById("nss-input").value;
    const sector = document.getElementById("nss-sector").value;
    const finalSector = sector === "Both" ? "" : sector;

    fetch("http://127.0.0.1:5500/analyze_nss_region", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nss_region, sector: finalSector })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showNotification(data.error, "error");
        } else {
            drawExpenseCharts(data.local_distribution, data.india_distribution, data.avg_local, data.avg_india, "nss");
        }
    });
}

function drawExpenseCharts(localData, indiaData, avgLocal, avgIndia, prefix) {
    const pieId = prefix + "-expensePieChart";
    const barId = prefix + "-compareBarChart";
    const avgId = prefix + "-average-info";

    const labels = Object.keys(localData);
    const pieCtx = document.getElementById(pieId).getContext("2d");
    const barCtx = document.getElementById(barId).getContext("2d");

    if (window[prefix + "PieChart"]) window[prefix + "PieChart"].destroy();
    if (window[prefix + "BarChart"]) window[prefix + "BarChart"].destroy();

    window[prefix + "PieChart"] = new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                label: 'Expense Class Distribution',
                data: Object.values(localData),
                backgroundColor: ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099', '#0099C6', '#DD4477', '#66AA00'],
                borderColor: '#fff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                tooltip: { enabled: true },
                legend: { position: 'bottom' }
            }
        }
    });

    window[prefix + "BarChart"] = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Selected Region',
                    data: Object.values(localData),
                    backgroundColor: '#42a5f5'
                },
                {
                    label: 'All India',
                    data: Object.values(indiaData),
                    backgroundColor: '#ffca28'
                }
            ]
        },
        options: {
            responsive: true,
            aspectRatio: 3/2,
            scales: {
                y: { beginAtZero: true },
                x: {
                    ticks: {
                        font: { size: 14 },
                        maxRotation: 45,
                        minRotation: 30
                    }
                }
            },
            plugins: {
                tooltip: { enabled: true },
                legend: {
                    position: 'top',
                    labels: {
                        font: { size: 14 }
                    }
                }
            }
        }        
    });

    document.getElementById(avgId).innerHTML = `
        <div class="result-card">
            <div class="result-item"><h3>Avg MPCE (Region)</h3><div class="amount">₹${avgLocal.MPCE.toLocaleString()}</div></div>
            <div class="result-item"><h3>Avg Total Expense (Region)</h3><div class="amount">₹${avgLocal.TotalExpense.toLocaleString()}</div></div>
            <div class="result-item"><h3>Avg MPCE (India)</h3><div class="amount">₹${avgIndia.MPCE.toLocaleString()}</div></div>
            <div class="result-item"><h3>Avg Total Expense (India)</h3><div class="amount">₹${avgIndia.TotalExpense.toLocaleString()}</div></div>
        </div>
    `;
}

document.getElementById("expensePieChart").scrollIntoView({ behavior: "smooth", block: "center" });

// Chatbot logic
// Chatbot Toggle & Close
document.getElementById("chatbot-toggle").addEventListener("click", () => {
    const windowEl = document.getElementById("chatbot-window");
    const messagesEl = document.getElementById("chat-messages");
    if (windowEl.classList.contains("hidden")) {
        windowEl.classList.remove("hidden");
        messagesEl.innerHTML = `<div class="bot">👋 Hello! I'm your data assistant. How can I help you today?</div>`;
    } else {
        windowEl.classList.add("hidden");
    }
});

document.getElementById("chat-close").addEventListener("click", () => {
    document.getElementById("chatbot-window").classList.add("hidden");
});

// Chat Input Enter
document.getElementById("chat-input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
        const input = this.value.trim();
        if (!input) return;
        const messages = document.getElementById("chat-messages");

        // User message
        const userMsg = document.createElement("div");
        userMsg.className = "user";
        userMsg.textContent = input;
        messages.appendChild(userMsg);

        // Clear input
        this.value = "";

        // Show loading
        const loadingMsg = document.createElement("div");
        loadingMsg.className = "bot";
        loadingMsg.textContent = "Thinking...";
        messages.appendChild(loadingMsg);
        messages.scrollTop = messages.scrollHeight;

        fetch("http://127.0.0.1:5500/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: input })
        })
        .then(res => res.json())
        .then(data => {
            loadingMsg.remove();
            const botMsg = document.createElement("div");
            botMsg.className = "bot";
            botMsg.textContent = data.answer || data.error || "❌ Failed to respond.";
            messages.appendChild(botMsg);
            messages.scrollTop = messages.scrollHeight;
        })
        .catch(() => {
            loadingMsg.remove();
            const botMsg = document.createElement("div");
            botMsg.className = "bot";
            botMsg.textContent = "❌ Something went wrong.";
            messages.appendChild(botMsg);
        });
    }
});
