// Example Animation: Bouncing button on hover
document.addEventListener('DOMContentLoaded', function () {
    const buttons = document.querySelectorAll('button');

    buttons.forEach((button) => {
        button.addEventListener('mouseover', function () {
            button.classList.add('button-animate');
        });

        button.addEventListener('mouseout', function () {
            button.classList.remove('button-animate');
        });
    });
});
