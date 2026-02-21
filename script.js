// ============================================
// 🐍 Python Masterclass — Tech Theme Scripts
// ============================================

// --- Content Unlock Configuration ---
// Set the date and time when specific pages should unlock for students.
// Format: 'YYYY-MM-DDTHH:MM:SS'
const UNLOCK_DATES = {
    // Example: Locks OOP and NumPy until March 2026. Change these dates as needed!
    'oop.html': new Date('2026-02-22T14:00:00').getTime(),
    'numpy.html': new Date('2026-02-25T14:00:00').getTime()
};

// --- Teacher/Admin Bypass Key ---
// Anyone with this key can bypass the lock screen. It resets on page refresh!
const UNLOCK_SECRET = "galaxy";

// --- Particle Network Background ---
(function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    const PARTICLE_COUNT = 60;
    const CONNECTION_DIST = 150;
    let mouse = { x: null, y: null };

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 1.5 + 0.5;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(56, 189, 248, 0.5)';
            ctx.fill();
        }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONNECTION_DIST) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(56, 189, 248, ${0.12 * (1 - dist / CONNECTION_DIST)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
            // Mouse interaction
            if (mouse.x !== null) {
                const dx = particles[i].x - mouse.x;
                const dy = particles[i].y - mouse.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 200) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.strokeStyle = `rgba(167, 139, 250, ${0.15 * (1 - dist / 200)})`;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
})();


// --- Card Mouse Tracking Glow ---
document.querySelectorAll('.task-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
        card.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
    });
});


// --- Toggle Code Answers ---
function toggleCode(taskId, btnElement) {
    const answerBlock = document.getElementById(taskId);
    const isHidden = window.getComputedStyle(answerBlock).display === 'none';

    if (isHidden) {
        answerBlock.style.display = 'block';
        btnElement.innerText = "Hide Answer";
        btnElement.classList.add('hide-mode');
        answerBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
        answerBlock.style.display = 'none';
        btnElement.innerText = "Show Answer";
        btnElement.classList.remove('hide-mode');
    }
}


// --- Copy Code to Clipboard ---
function copyToClipboard(button) {
    const container = button.closest('.answer-container');
    const codeBlock = container.querySelector('code');
    const codeText = codeBlock.innerText;

    navigator.clipboard.writeText(codeText).then(() => {
        const originalText = button.innerText;
        button.innerText = "✓ Copied!";
        button.style.color = "#34d399";
        button.style.borderColor = "#34d399";

        setTimeout(() => {
            button.innerText = originalText;
            button.style.color = "";
            button.style.borderColor = "";
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// --- Toggle Output Content ---
function toggleOutput(id, btn) {
    const el = document.getElementById(id);
    if (el.classList.contains('output-visible')) {
        el.classList.remove('output-visible');
        btn.innerText = 'Show Output';
    } else {
        el.classList.add('output-visible');
        btn.innerText = 'Hide Output';
    }
}

// --- Mobile Navigation Toggle & Content Lock ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Content Lock Check (Direct Access via URL)
    const currentTime = new Date().getTime();
    const pathArray = window.location.pathname.split('/');
    let currentPage = pathArray[pathArray.length - 1];
    if (currentPage === '') currentPage = 'index.html'; // Fallback for root

    if (UNLOCK_DATES[currentPage] && currentTime < UNLOCK_DATES[currentPage]) {
        const unlockDateObj = new Date(UNLOCK_DATES[currentPage]);
        const unlockDateStr = unlockDateObj.toLocaleString(undefined, {
            weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });

        // Hide main content elements completely
        const elementsToHide = document.querySelectorAll('header, .storybook-intro, .task-card, .module-title');
        elementsToHide.forEach(el => el.style.display = 'none');

        // Inject high-end lock screen overlay into the container
        const lockScreen = document.createElement('div');
        lockScreen.className = 'locked-screen';
        lockScreen.innerHTML = `
            <div class="lock-icon" id="secret-unlock-btn" style="cursor: pointer;" title="Teacher Access">🔒</div>
            <h2>Content Locked</h2>
            <p style="color: #a1a1aa; max-width: 400px; margin: 0 auto;">The materials and tasks for this module have not yet been released. Please return on the scheduled date.</p>
            <div class="unlock-date">Unlocks: ${unlockDateStr}</div>
            <br>
            <a href="index.html" class="btn-output" style="display: inline-block; text-decoration: none; margin-top: 2.5rem;">Return to Dashboard</a>
        `;
        const container = document.querySelector('.container');
        if (container) {
            container.appendChild(lockScreen);
            
            // --- Secret Bypass Logic ---
            const secretBtn = document.getElementById('secret-unlock-btn');
            secretBtn.addEventListener('click', () => {
                const password = prompt("Teacher Access Required:");
                if (password === UNLOCK_SECRET) {
                    // Unlock success! Remove lock screen and reveal content.
                    lockScreen.remove();
                    elementsToHide.forEach(el => el.style.removeProperty('display'));
                    showToast("🔓 Teacher Access Granted!");
                } else if (password !== null) {
                    alert("Incorrect Access Key.");
                }
            });
        }
    }

    // 2. Navigation Link Intercept & Mobile Toggle
    const nav = document.querySelector('.top-nav');
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            // Mobile menu toggle (Only applies if it's the active toggle button)
            if (link.classList.contains('active') && window.innerWidth <= 640 && nav) {
                e.preventDefault();
                nav.classList.toggle('open');
            }
        });
    });
});

// Toast Notification System (for locked links)
function showToast(message) {
    let toast = document.getElementById('lock-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'lock-toast';
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    
    // Force browser reflow to reset animation
    void toast.offsetWidth; 
    
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}
