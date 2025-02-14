document.addEventListener("DOMContentLoaded", function () {
    window.addEventListener("scroll", function () {
        let header = document.querySelector(".header");
        if (window.scrollY > 50) {
            header.classList.add("scrolled");
        } else {
            header.classList.remove("scrolled");
        }
    });
});
document.addEventListener("DOMContentLoaded", function () {
    function animateCountUp(element) {
        let target = parseInt(element.getAttribute("data-count"));
        let count = 0;
        let speed = Math.max(10, target / 100);  // Adjust speed based on number size
        
        let updateCounter = setInterval(() => {
            count += Math.ceil(target / 100);  // Increment smoothly
            if (count >= target) {
                count = target;  // Ensure final value is exact
                clearInterval(updateCounter);
            }
            element.innerText = count;
        }, speed);
    }

    // Find all count-up elements and apply animation
    document.querySelectorAll(".count-up").forEach(animateCountUp);
});
