// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Authentication related elements
    const signupForm = document.getElementById('signupForm');
    const loginForm = document.getElementById('loginForm');
    const showLoginModal = document.getElementById('showLoginModal');
    const showSignupModal = document.getElementById('showSignupModal');
    
    // Prediction form
    const predictionForm = document.getElementById('predictionForm');
    const resultsCard = document.getElementById('resultsCard');
    
    // Chart objects for later updates
    let confidenceChart = null;
    let modelComparisonChart = null;
    let featureImportanceChart = null;
    
    // Toggle between sign up and login modals
    if (showLoginModal) {
        showLoginModal.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Hide signup modal and show login modal
            const signupModal = bootstrap.Modal.getInstance(document.getElementById('signupModal'));
            signupModal.hide();
            
            const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
            loginModal.show();
        });
    }
    
    if (showSignupModal) {
        showSignupModal.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Hide login modal and show signup modal
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            loginModal.hide();
            
            const signupModal = new bootstrap.Modal(document.getElementById('signupModal'));
            signupModal.show();
        });
    }
    
    // Handle signup form submission
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const email = document.getElementById('signupEmail').value;
            const password = document.getElementById('signupPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            
            // Validate passwords match
            if (password !== confirmPassword) {
                alert("Passwords don't match!");
                return;
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('email', email);
            formData.append('password', password);
            
            // Send signup request
            fetch('/signup', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Refresh page on successful signup
                    window.location.href = '/';
                } else {
                    // Show error message
                    alert(data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during sign up');
            });
        });
    }
    
    // Handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            
            // Create form data
            const formData = new FormData();
            formData.append('email', email);
            formData.append('password', password);
            
            // Send login request
            fetch('/login', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Refresh page on successful login
                    window.location.href = '/';
                } else {
                    // Show error message
                    alert(data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during login');
            });
        });
    }
    
    // Handle prediction form submission
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Create form data from all form fields
            const formData = new FormData(predictionForm);
            
            // Convert form data to object for easier processing
            const formDataObj = {};
            formData.forEach((value, key) => {
                formDataObj[key] = value;
            });
            
            // Show loading state
            const submitBtn = predictionForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            submitBtn.disabled = true;
            
            // Simulate processing time (1-2 seconds)
            setTimeout(() => {
                // Generate prediction results locally
                const predictionResults = generatePredictions(formDataObj);
                
                // Reset button state
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
                
                // Display results
                displayResults(predictionResults);
            }, 1500);
        });
    }
    
    // Function to generate local predictions based on form data
    function generatePredictions(formData) {
        // Parse numeric values
        const numericFeatures = [
            'Src_Bytes', 'Dst_Bytes', 'Count', 'Serror_Rate', 
            'Same_Srv_Rate', 'Diff_Srv_Rate', 'Dst_Host_Srv_Count',
            'Dst_Host_Same_Srv_Count', 'Dst_Host_Diff_Srv_Count', 
            'Dst_Host_Serror_Rate'
        ];
        
        numericFeatures.forEach(feature => {
            formData[feature] = parseFloat(formData[feature]);
        });
        
        // Attack types
        const attackTypes = [
            "Normal", "DDoS", "Probe", "R2L", "U2R"
        ];
        
        // Calculate a simple risk score based on input values
        // This is just a simple heuristic for demonstration
        let riskScore = 0;
        
        // Service and flag based risks
        const highRiskServices = ['ftp', 'ftp_data', 'exec', 'login', 'telnet'];
        const highRiskFlags = ['RSTO', 'RSTOS0', 'REJ', 'S0'];
        
        if (highRiskServices.includes(formData['Service'].toLowerCase())) riskScore += 20;
        if (highRiskFlags.includes(formData['Flag'].toUpperCase())) riskScore += 20;
        
        // Numeric feature based risks
        if (formData['Src_Bytes'] > 1000) riskScore += 5;
        if (formData['Dst_Bytes'] > 1000) riskScore += 5;
        if (formData['Count'] > 10) riskScore += 10;
        if (formData['Serror_Rate'] > 0.5) riskScore += 15;
        if (formData['Same_Srv_Rate'] < 0.3) riskScore += 10;
        if (formData['Diff_Srv_Rate'] > 0.7) riskScore += 10;
        if (formData['Dst_Host_Serror_Rate'] > 0.5) riskScore += 15;
        
        // Generate model predictions with some variation
        // Each model will have slightly different behavior
        
        // MLP model - sensitive to error rates
        const mlpPrediction = {
            model: "MLP",
            attackIndex: riskScore > 45 ? 
                (formData['Serror_Rate'] > 0.7 ? 1 : // DoS
                (formData['Diff_Srv_Rate'] > 0.5 ? 2 : 3)) : 0, // Probe or R2L if not DoS, otherwise Normal
            confidence: Math.min(0.55 + (riskScore / 200), 0.98)
        };
        
        // Passive Aggressive - sensitive to bytes and counts
        const paScore = riskScore + (formData['Src_Bytes'] > 2000 ? 15 : 0) + (formData['Count'] > 15 ? 15 : 0);
        const paPrediction = {
            model: "Passive Aggressive",
            attackIndex: paScore > 50 ? 
                (formData['Count'] > 10 ? 1 : // DoS
                (formData['Dst_Host_Diff_Srv_Count'] > 5 ? 2 : 3)) : 0, // Probe or R2L if not DoS, otherwise Normal
            confidence: Math.min(0.6 + (paScore / 190), 0.95)
        };
        
        // Naive Bayes - more conservative with predictions
        const nbScore = riskScore * 0.8;
        const nbPrediction = {
            model: "Naive Bayes",
            attackIndex: nbScore > 55 ? 
                (formData['Service'].toLowerCase() === 'http' ? 1 : // DoS for HTTP
                (formData['Dst_Host_Serror_Rate'] > 0.6 ? 2 : 4)) : 0, // Probe or U2R if not DoS, otherwise Normal
            confidence: Math.min(0.5 + (nbScore / 220), 0.9)
        };
        
        // Stacking Classifier - ensemble approach
        // Weight the other models based on their typical strengths
        const scPrediction = {
            model: "Stacking Classifier",
            attackIndex: 0, // Will be determined by voting
            confidence: 0   // Will be calculated based on agreement
        };
        
        // Simple weighted voting ensemble for stacking classifier
        const votes = Array(5).fill(0); // [Normal, DoS, Probe, R2L, U2R]
        
        // Add weighted votes
        votes[mlpPrediction.attackIndex] += mlpPrediction.confidence * 0.35;
        votes[paPrediction.attackIndex] += paPrediction.confidence * 0.35;
        votes[nbPrediction.attackIndex] += nbPrediction.confidence * 0.3;
        
        // Find highest voted class
        let maxVote = 0;
        let maxIndex = 0;
        votes.forEach((vote, index) => {
            if (vote > maxVote) {
                maxVote = vote;
                maxIndex = index;
            }
        });
        
        scPrediction.attackIndex = maxIndex;
        
        // Calculate confidence based on agreement level
        const totalVotes = votes.reduce((sum, vote) => sum + vote, 0);
        scPrediction.confidence = maxVote / totalVotes;
        
        // Create model results array
        const modelResults = [
            {
                ...mlpPrediction,
                attackType: attackTypes[mlpPrediction.attackIndex]
            },
            {
                ...paPrediction,
                attackType: attackTypes[paPrediction.attackIndex]
            },
            {
                ...nbPrediction,
                attackType: attackTypes[nbPrediction.attackIndex]
            },
            {
                ...scPrediction,
                attackType: attackTypes[scPrediction.attackIndex]
            }
        ];
        
        // Determine the final prediction (using stacking classifier)
        const finalPrediction = {
            status: 'success',
            prediction: scPrediction.attackIndex === 0 ? 0 : 1, // 0 for Normal, 1 for Attack
            attack_type: attackTypes[scPrediction.attackIndex],
            confidence: scPrediction.confidence,
            model_results: modelResults,
            feature_importance: generateFeatureImportance(formData)
        };
        
        return finalPrediction;
    }
    
    // Generate feature importance analysis
    function generateFeatureImportance(formData) {
        // Identify top features based on the input values
        const features = [
            { name: 'Service', value: formData['Service'], importance: 0 },
            { name: 'Flag', value: formData['Flag'], importance: 0 },
            { name: 'Src_Bytes', value: formData['Src_Bytes'], importance: 0 },
            { name: 'Dst_Bytes', value: formData['Dst_Bytes'], importance: 0 },
            { name: 'Count', value: formData['Count'], importance: 0 },
            { name: 'Serror_Rate', value: formData['Serror_Rate'], importance: 0 },
            { name: 'Same_Srv_Rate', value: formData['Same_Srv_Rate'], importance: 0 },
            { name: 'Diff_Srv_Rate', value: formData['Diff_Srv_Rate'], importance: 0 },
            { name: 'Dst_Host_Srv_Count', value: formData['Dst_Host_Srv_Count'], importance: 0 },
            { name: 'Dst_Host_Same_Srv_Count', value: formData['Dst_Host_Same_Srv_Count'], importance: 0 },
            { name: 'Dst_Host_Diff_Srv_Count', value: formData['Dst_Host_Diff_Srv_Count'], importance: 0 },
            { name: 'Dst_Host_Serror_Rate', value: formData['Dst_Host_Serror_Rate'], importance: 0 }
        ];
        
        // Calculate importance (this is simplified for demonstration)
        // High importance to flag for certain values
        if (['S0', 'REJ', 'RSTO', 'RSTOS0'].includes(formData['Flag'].toUpperCase())) {
            features.find(f => f.name === 'Flag').importance = 0.85;
        }
        
        // Service importance
        if (['ftp', 'http', 'smtp', 'ssh'].includes(formData['Service'].toLowerCase())) {
            features.find(f => f.name === 'Service').importance = 0.75;
        }
        
        // Error rates are typically important
        features.find(f => f.name === 'Serror_Rate').importance = 0.6 + (formData['Serror_Rate'] * 0.3);
        features.find(f => f.name === 'Dst_Host_Serror_Rate').importance = 0.55 + (formData['Dst_Host_Serror_Rate'] * 0.3);
        
        // Bytes importance
        const srcBytesFeature = features.find(f => f.name === 'Src_Bytes');
        srcBytesFeature.importance = formData['Src_Bytes'] > 1000 ? 0.7 : 0.4;
        
        const dstBytesFeature = features.find(f => f.name === 'Dst_Bytes');
        dstBytesFeature.importance = formData['Dst_Bytes'] > 1000 ? 0.65 : 0.35;
        
        // Count importance
        features.find(f => f.name === 'Count').importance = 0.5 + (Math.min(formData['Count'], 20) / 40);
        
        // Service rate importance
        features.find(f => f.name === 'Same_Srv_Rate').importance = 0.3 + (Math.abs(0.5 - formData['Same_Srv_Rate']) * 0.8);
        features.find(f => f.name === 'Diff_Srv_Rate').importance = 0.3 + (formData['Diff_Srv_Rate'] * 0.6);
        
        // Normalize other features
        features.forEach(feature => {
            // Make sure importance is between 0 and 1
            feature.importance = Math.min(Math.max(feature.importance, 0), 1);
            
            // Format values for display
            if (typeof feature.value === 'number') {
                if (feature.value < 1 && feature.value > 0) {
                    feature.value = feature.value.toFixed(2);
                } else {
                    feature.value = feature.value.toString();
                }
            }
        });
        
        // Sort by importance (descending)
        return features.sort((a, b) => b.importance - a.importance);
    }
    
    // Function to display prediction results with enhanced visualizations
function displayResults(data) {
    // Make results card visible
    resultsCard.style.display = 'block';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth' });
    
    // Get elements
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const detailedResult = document.getElementById('detailedResult');
    const resultAlert = document.getElementById('resultAlert');
    const resultsHeader = document.getElementById('resultsHeader');
    const modelResults = document.getElementById('modelResults');
    const featureDetails = document.getElementById('featureDetails');
    const resultIcon = document.getElementById('resultIcon');
    const combinedResultBox = document.getElementById('combinedResultBox');
    
    // Set result text
    predictionResult.textContent = data.attack_type;
    confidenceResult.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    
    // Clear previous classes
    resultsHeader.className = 'card-header';
    resultAlert.className = 'alert w-100';
    resultIcon.className = 'bi result-icon';
    
    // Set appropriate styling based on prediction
    if (data.prediction === 0) { // Normal traffic
        resultsHeader.classList.add('bg-success', 'text-white');
        resultAlert.classList.add('alert-success');
        resultIcon.classList.add('bi-shield-check');
        detailedResult.textContent = 'The traffic pattern appears to be normal with no signs of malicious activity.';
        combinedResultBox.style.borderColor = '#198754';
    } else { // Attack detected
        resultsHeader.classList.add('bg-danger', 'text-white');
        resultAlert.classList.add('alert-danger');
        resultIcon.classList.add('bi-shield-exclamation');
        
        // Set detailed result text based on attack type
        switch(data.attack_type) {
            case 'DDoS':
                detailedResult.textContent = 'Distributed Denial of Service attack detected. The system may be under resource exhaustion attack.';
                break;
            case 'Probe':
                detailedResult.textContent = 'Probe attack detected. Scanning activity to discover vulnerabilities.';
                break;
            case 'R2L':
                detailedResult.textContent = 'Remote to Local attack detected. Unauthorized access from remote machine.';
                break;
            case 'U2R':
                detailedResult.textContent = 'User to Root attack detected. Attempt to gain root/admin privileges.';
                break;
            default:
                detailedResult.textContent = 'Suspicious traffic detected. Review details below.';
        }
        
        combinedResultBox.style.borderColor = '#dc3545';
    }
    
    // Create confidence gauge chart
    drawConfidenceChart(data.confidence);
    
    // Create model comparison chart
    drawModelComparisonChart(data.model_results);
    
    // Create feature importance chart
    drawFeatureImportanceChart(data.feature_importance);
    
    // Display individual model results
    displayModelResults(data.model_results);
    
    // Display feature details
    displayFeatureDetails(data.feature_importance);
}

function displayFeatureDetails(features) {
    const featureDetailsEl = document.getElementById('featureDetails');
    featureDetailsEl.innerHTML = ''; // Clear old content

    features.forEach(feature => {
        const li = document.createElement('li');
        li.className = 'feature-item bg-light p-2 border';
        li.innerHTML = `
            <strong>${feature.name}</strong>: Value = <code>${feature.value}</code>, Importance = <strong>${(feature.importance * 100).toFixed(1)}%</strong>
        `;
        featureDetailsEl.appendChild(li);
    });
}

// Create confidence gauge chart
function drawConfidenceChart(confidence) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    // Create new chart
    confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Confidence', 'Uncertainty'],
            datasets: [{
                data: [confidence, 1 - confidence],
                backgroundColor: [
                    confidence > 0.8 ? '#198754' : confidence > 0.6 ? '#ffc107' : '#dc3545',
                    '#e9ecef'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Add center text
    Chart.pluginService?.register({
        beforeDraw: function(chart) {
            if (chart.config.type === 'doughnut') {
                const width = chart.width;
                const height = chart.height;
                const ctx = chart.ctx;
                ctx.restore();
                const fontSize = (height / 114).toFixed(2);
                ctx.font = fontSize + 'em sans-serif';
                ctx.textBaseline = 'middle';
                
                const text = `${(confidence * 100).toFixed(0)}%`;
                const textX = Math.round((width - ctx.measureText(text).width) / 2);
                const textY = height / 2;
                
                ctx.fillText(text, textX, textY);
                ctx.save();
            }
        }
    });
}

// Create model comparison chart
function drawModelComparisonChart(modelResults) {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (modelComparisonChart) {
        modelComparisonChart.destroy();
    }
    
    // Extract data from model results
    const labels = modelResults.map(result => result.model);
    const confidences = modelResults.map(result => result.confidence);
    const attackTypes = modelResults.map(result => result.attackType);
    
    // Define colors for attack types
    const attackColors = {
        'Normal': '#198754',
        'DDoS': '#dc3545',
        'Probe': '#ffc107',
        'R2L': '#fd7e14',
        'U2R': '#6f42c1'
    };
    
    // Create colors array based on attack types
    const colors = attackTypes.map(type => attackColors[type] || '#6c757d');
    
    // Create new chart
    modelComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Confidence',
                data: confidences.map(conf => conf * 100),
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            return `Attack Type: ${attackTypes[context.dataIndex]}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    }
                }
            }
        }
    });
}

// Create feature importance chart
function drawFeatureImportanceChart(featureImportance) {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    // Get top 5 features
    const topFeatures = featureImportance.slice(0, 5);
    
    // Extract data
    const labels = topFeatures.map(feature => feature.name);
    const importances = topFeatures.map(feature => feature.importance);
    
    // Create gradient colors based on importance
    const colors = importances.map(importance => {
        const value = Math.floor(importance * 255);
        return `rgba(${255-value}, ${value}, 100, 0.7)`;
    });
    
    // Create new chart
    featureImportanceChart = new Chart(ctx, {
        type: 'bar', // Changed from 'horizontalBar' which is deprecated
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: importances.map(imp => imp * 100),
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y', // This makes the bar chart horizontal
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Importance (%)'
                    }
                }
            }
        }
    });
}

// Fixed the center text plugin for the confidence chart
// Chart.pluginService is deprecated in newer versions
const centerTextPlugin = {
    id: 'centerText',
    beforeDraw: function(chart) {
        if (chart.config.type === 'doughnut') {
            const width = chart.width;
            const height = chart.height;
            const ctx = chart.ctx;
            ctx.restore();
            const fontSize = (height / 114).toFixed(2);
            ctx.font = fontSize + 'em sans-serif';
            ctx.textBaseline = 'middle';
            ctx.textAlign = 'center';
            
            const text = `${(chart.data.datasets[0].data[0] * 100).toFixed(0)}%`;
            ctx.fillText(text, width/2, height/2);
            ctx.save();
        }
    }
};

// Register the plugin
Chart.register(centerTextPlugin);

// Display individual model results in cards
function displayModelResults(modelResults) {
    // Clear previous content
    const modelResultsEl = document.getElementById('modelResults');
    modelResultsEl.innerHTML = '';
    
    // Create cards for each model
    modelResults.forEach(model => {
        // Create card element
        const cardDiv = document.createElement('div');
        cardDiv.className = 'col-md-3 mb-3';
        
        // Define color based on attack type
        let bgColor, textColor;
        
        switch(model.attackType) {
            case 'Normal':
                bgColor = 'bg-success';
                textColor = 'text-white';
                break;
            case 'DDoS':
                bgColor = 'bg-danger';
                textColor = 'text-white';
                break;
            case 'Probe':
                bgColor = 'bg-warning';
                textColor = 'text-dark';
                break;
            case 'R2L':
                bgColor = 'bg-orange';
                textColor = 'text-white';
                break;
            case 'U2R':
                bgColor = 'bg-purple';
                textColor = 'text-white';
                break;
            default:
                bgColor = 'bg-secondary';
                textColor = 'text-white';
        }
        
        // Format confidence percentage
        const confidencePercent = (model.confidence * 100).toFixed(1);
        
        // Define confidence color
        let confidenceColor;
        if (model.confidence > 0.8) {
            confidenceColor = 'bg-success text-white';
        } else if (model.confidence > 0.6) {
            confidenceColor = 'bg-warning text-dark';
        } else {
            confidenceColor = 'bg-danger text-white';
        }
        
        // Set card content
        cardDiv.innerHTML = `
            <div class="model-card shadow">
                <div class="model-header ${bgColor} ${textColor}">
                    ${model.model}
                </div>
                <div class="model-body">
                    <h5 class="mb-3">${model.attackType}</h5>
                    <div class="confidence-pill ${confidenceColor}">
                        ${confidencePercent}% Confidence
                    </div>
                </div>
            </div>
        `;
        
        // Append card to container
        modelResultsEl.appendChild(cardDiv);
    });
}
});
