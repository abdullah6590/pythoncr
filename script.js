// ============================================
// 🐍 Python Masterclass — Tech Theme Scripts
// ============================================

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

// --- Mobile Navigation Toggle ---
document.addEventListener('DOMContentLoaded', () => {
    const nav = document.querySelector('.top-nav');
    const activeLink = document.querySelector('.nav-link.active');
    
    if (nav && activeLink) {
        activeLink.addEventListener('click', (e) => {
            // Only toggle menu on mobile screens
            if (window.innerWidth <= 640) {
                e.preventDefault(); // Prevent navigating to same page immediately
                nav.classList.toggle('open');
            }
        });
    }
});
