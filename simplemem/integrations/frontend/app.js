// SimpleMem Frontend Application

const API_BASE = window.location.origin;

// DOM Elements
const registerSection = document.getElementById('register-section');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');
const registerForm = document.getElementById('register-form');
const apiKeyInput = document.getElementById('api-key');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoading = submitBtn.querySelector('.btn-loading');

// State
let currentToken = null;
let currentUserId = null;

// Event Listeners
registerForm.addEventListener('submit', handleRegister);

// Handlers
async function handleRegister(e) {
    e.preventDefault();

    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        showError('Please enter your OpenRouter API key.');
        return;
    }

    // Show loading state
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                openrouter_api_key: apiKey,
            }),
        });

        const data = await response.json();

        if (data.success) {
            currentToken = data.token;
            currentUserId = data.user_id;
            showResult(data);
        } else {
            showError(data.error || 'Registration failed. Please check your API key.');
        }
    } catch (err) {
        console.error('Registration error:', err);
        showError('Network error. Please check your connection and try again.');
    } finally {
        setLoading(false);
    }
}

function showResult(data) {
    // Hide other sections
    registerSection.style.display = 'none';
    errorSection.style.display = 'none';

    // Populate result values
    document.getElementById('token-value').textContent = data.token;
    document.getElementById('endpoint-value').textContent = `${API_BASE}/mcp`;
    document.getElementById('user-id-value').textContent = data.user_id;

    // Generate Claude Desktop config (MCP over HTTP)
    const config = {
        "mcpServers": {
            "simplemem": {
                "type": "http",
                "url": `${API_BASE}/mcp`,
                "headers": {
                    "Authorization": `Bearer ${data.token}`
                }
            }
        }
    };
    document.getElementById('claude-config').textContent = JSON.stringify(config, null, 2);

    // Show result section with animation
    resultSection.style.display = 'block';
    resultSection.classList.add('fade-in');
}

function showError(message) {
    // Hide other sections
    resultSection.style.display = 'none';

    // Show error
    document.getElementById('error-message').textContent = message;
    errorSection.style.display = 'block';
    errorSection.classList.add('fade-in');
}

function hideError() {
    errorSection.style.display = 'none';
    registerSection.style.display = 'block';
}

function resetForm() {
    // Clear form
    apiKeyInput.value = '';
    currentToken = null;
    currentUserId = null;

    // Show register section
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    registerSection.style.display = 'block';
    registerSection.classList.add('fade-in');
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    btnText.style.display = isLoading ? 'none' : 'inline-flex';
    btnLoading.style.display = isLoading ? 'inline-flex' : 'none';
}

// Clipboard Functions
function fallbackCopyToClipboard(text) {
    // Fallback for non-HTTPS or older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    textarea.style.top = '0';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    try {
        document.execCommand('copy');
        return true;
    } catch (err) {
        console.error('Fallback copy failed:', err);
        return false;
    } finally {
        document.body.removeChild(textarea);
    }
}

function showCopySuccess(btn) {
    if (!btn) return;
    const originalText = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');

    setTimeout(() => {
        btn.textContent = originalText;
        btn.classList.remove('copied');
    }, 2000);
}

function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const text = element.textContent;
    const btn = element.parentElement.querySelector('.copy-btn');

    // Check if clipboard API is available (requires HTTPS or localhost)
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showCopySuccess(btn);
        }).catch(err => {
            console.error('Clipboard API failed:', err);
            if (fallbackCopyToClipboard(text)) {
                showCopySuccess(btn);
            }
        });
    } else {
        // Use fallback for HTTP environments
        if (fallbackCopyToClipboard(text)) {
            showCopySuccess(btn);
        }
    }
}

function copyConfig() {
    const configElement = document.getElementById('claude-config');
    if (!configElement) return;

    const text = configElement.textContent;
    const btn = document.querySelector('.config-section .copy-btn');

    // Check if clipboard API is available
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showCopySuccess(btn);
        }).catch(err => {
            console.error('Clipboard API failed:', err);
            if (fallbackCopyToClipboard(text)) {
                showCopySuccess(btn);
            }
        });
    } else {
        // Use fallback for HTTP environments
        if (fallbackCopyToClipboard(text)) {
            showCopySuccess(btn);
        }
    }
}

// Check for existing token in URL
function checkUrlToken() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');

    if (token) {
        // Verify token
        fetch(`${API_BASE}/api/auth/verify?token=${token}`)
            .then(res => res.json())
            .then(data => {
                if (data.valid) {
                    currentToken = token;
                    currentUserId = data.user_id;

                    // Show result with existing token
                    showResult({
                        token: token,
                        user_id: data.user_id,
                        mcp_endpoint: `/mcp`,
                    });
                }
            })
            .catch(err => {
                console.error('Token verification failed:', err);
            });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkUrlToken();
});
