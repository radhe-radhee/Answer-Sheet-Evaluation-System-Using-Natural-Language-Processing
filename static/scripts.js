
document.addEventListener('DOMContentLoaded', function() {
    // Hit counter simulation
    if (document.getElementById('counter')) {
        let visits = localStorage.getItem('visitCount') || 0;
        visits++;
        localStorage.setItem('visitCount', visits);
        document.getElementById('counter').textContent = visits;
    }

    // Blinking text effect
    const blinkElements = document.querySelectorAll('.blink');
    blinkElements.forEach(el => {
        setInterval(() => {
            el.style.visibility = el.style.visibility === 'hidden' ? 'visible' : 'hidden';
        }, 500);
    });

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const fileInputs = form.querySelectorAll('input[type="file"]');
            let valid = true;

            fileInputs.forEach(input => {
                if (input.required && !input.value) {
                    alert(`Please select a ${input.name.replace('_', ' ')} file!`);
                    valid = false;
                }
            });

            if (!valid) {
                e.preventDefault();
            }
        });
    });

    // Confirm before deletion
    const deleteButtons = document.querySelectorAll('.delete-button');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this evaluation?')) {
                e.preventDefault();
            }
        });
    });

    // Auto-close print window
    if (window.location.pathname.includes('/print_result/')) {
        window.onafterprint = function() {
            window.close();
        };
    }
});
