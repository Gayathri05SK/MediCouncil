const API_BASE = '/api';

document.getElementById('triageForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsContent').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    // Disable submit button
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    try {
        // Collect form data
        const formData = {
            symptoms_text: document.getElementById('symptoms').value.trim(),
            age: document.getElementById('age').value ? parseInt(document.getElementById('age').value) : null,
            sex: document.getElementById('sex').value || null,
            chronic_conditions: document.getElementById('chronic').value 
                ? document.getElementById('chronic').value.split(',').map(s => s.trim()).filter(s => s)
                : [],
            red_flags: Array.from(document.querySelectorAll('input[name="red_flags"]:checked'))
                .map(cb => cb.value)
        };
        
        // Call API
        const response = await fetch(`${API_BASE}/triage`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        // Hide loading
        document.getElementById('loadingSpinner').style.display = 'none';
        
        if (result.status === 'success') {
            displayResults(result);
        } else {
            displayError(result.error || result.message || 'An error occurred');
        }
        
    } catch (error) {
        document.getElementById('loadingSpinner').style.display = 'none';
        displayError('Failed to connect to server. Please try again.');
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Symptoms';
    }
});

function displayResults(result) {
    document.getElementById('resultsContent').style.display = 'block';
    
    // Urgency
    const urgencyCard = document.getElementById('urgencyCard');
    const urgencyBadge = document.getElementById('urgencyBadge');
    urgencyBadge.textContent = result.urgency.replace('_', '-').toUpperCase();
    urgencyBadge.className = `urgency-badge urgency-${result.urgency}`;
    
    const urgencyTexts = {
        'emergency': '🚨 Seek IMMEDIATE emergency medical attention. Call emergency services now.',
        'urgent': '⚠️ See a healthcare provider within 24 hours.',
        'routine': '📅 Schedule an appointment with your doctor within 1-2 weeks.',
        'self_care': '✅ Self-care measures may be sufficient. Monitor symptoms.'
    };
    document.getElementById('urgencyText').textContent = urgencyTexts[result.urgency] || '';
    
    // Risk
    document.getElementById('riskLevel').textContent = result.risk_level;
    document.getElementById('riskScore').textContent = result.risk_score;
    document.getElementById('riskScoreFill').style.width = `${result.risk_score}%`;
    
    const riskColors = {
        'Critical': '#dc2626',
        'High': '#ef4444',
        'Medium': '#f59e0b',
        'Low': '#10b981'
    };
    document.getElementById('riskLevel').style.color = riskColors[result.risk_level] || '#1f2937';
    
    // Confidence
    document.getElementById('confidence').textContent = result.confidence;
    document.getElementById('confidenceLevel').textContent = result.confidence_level;
    document.getElementById('agreement').textContent = 
        `${(result.agent_agreement.rate * 100).toFixed(0)}% (${result.agent_agreement.level})`;
    document.getElementById('safetyOverride').textContent = result.safety_override ? 'YES' : 'NO';
    document.getElementById('safetyOverride').style.color = result.safety_override ? '#dc2626' : '#10b981';
    
    // Explanation
    document.getElementById('explanation').innerHTML = result.explanation.replace(/\n/g, '<br>');
    
    // Key symptoms
    if (result.key_symptoms && result.key_symptoms.length > 0) {
        document.getElementById('keySymptomsCard').style.display = 'block';
        document.getElementById('keySymptoms').innerHTML = result.key_symptoms
            .map(s => `<span class="symptom-tag">${s}</span>`)
            .join('');
    } else {
        document.getElementById('keySymptomsCard').style.display = 'none';
    }
    
    // ML Baseline
    if (result.ml_baseline) {
        document.getElementById('mlBaselineCard').style.display = 'block';
        const ml = result.ml_baseline.predictions;
        document.getElementById('mlBaseline').innerHTML = `
            <p><strong>Naive Bayes:</strong> ${ml.naive_bayes}</p>
            <p><strong>Logistic Regression:</strong> ${ml.logistic_regression}</p>
            <p><strong>Random Forest:</strong> ${ml.random_forest}</p>
        `;
    } else {
        document.getElementById('mlBaselineCard').style.display = 'none';
    }
    
    // Disclaimer
    document.getElementById('disclaimer').innerHTML = `<strong>Disclaimer:</strong> ${result.disclaimer}`;
}

function displayError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function resetForm() {
    document.getElementById('triageForm').reset();
    document.getElementById('resultsSection').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
